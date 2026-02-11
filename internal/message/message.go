package message

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/charmbracelet/crush/internal/db"
	"github.com/charmbracelet/crush/internal/pubsub"
	"github.com/google/uuid"
)

// publishInterval is the minimum interval between pubsub notifications
// for the same message during streaming. This prevents the TUI from being
// overwhelmed by per-token updates while still keeping the UI responsive.
const publishInterval = 33 * time.Millisecond // ~30 Hz

type CreateMessageParams struct {
	Role             MessageRole
	Parts            []ContentPart
	Model            string
	Provider         string
	IsSummaryMessage bool
}

type Service interface {
	pubsub.Subscriber[Message]
	Create(ctx context.Context, sessionID string, params CreateMessageParams) (Message, error)
	Update(ctx context.Context, message Message) error
	Get(ctx context.Context, id string) (Message, error)
	List(ctx context.Context, sessionID string) ([]Message, error)
	ListUserMessages(ctx context.Context, sessionID string) ([]Message, error)
	ListAllUserMessages(ctx context.Context) ([]Message, error)
	Delete(ctx context.Context, id string) error
	DeleteSessionMessages(ctx context.Context, sessionID string) error
}

type service struct {
	*pubsub.Broker[Message]
	q db.Querier

	// pendingUpdates stores the latest message state for throttled
	// publishing. When a message is updated rapidly (e.g., per-token
	// during streaming), only the latest state is published at most
	// once per publishInterval.
	pendingUpdates map[string]Message
	pendingTimers  map[string]*time.Timer
	mu             sync.Mutex
}

func NewService(q db.Querier) Service {
	return &service{
		Broker:         pubsub.NewBroker[Message](),
		q:              q,
		pendingUpdates: make(map[string]Message),
		pendingTimers:  make(map[string]*time.Timer),
	}
}

func (s *service) Delete(ctx context.Context, id string) error {
	message, err := s.Get(ctx, id)
	if err != nil {
		return err
	}
	err = s.q.DeleteMessage(ctx, message.ID)
	if err != nil {
		return err
	}
	s.cancelPendingUpdate(id)
	// Clone the message before publishing to avoid race conditions with
	// concurrent modifications to the Parts slice.
	s.Publish(pubsub.DeletedEvent, message.Clone())
	return nil
}

func (s *service) Create(ctx context.Context, sessionID string, params CreateMessageParams) (Message, error) {
	if params.Role != Assistant {
		params.Parts = append(params.Parts, Finish{
			Reason: "stop",
		})
	}
	partsJSON, err := marshalParts(params.Parts)
	if err != nil {
		return Message{}, err
	}
	isSummary := int64(0)
	if params.IsSummaryMessage {
		isSummary = 1
	}
	dbMessage, err := s.q.CreateMessage(ctx, db.CreateMessageParams{
		ID:               uuid.New().String(),
		SessionID:        sessionID,
		Role:             string(params.Role),
		Parts:            string(partsJSON),
		Model:            sql.NullString{String: string(params.Model), Valid: true},
		Provider:         sql.NullString{String: params.Provider, Valid: params.Provider != ""},
		IsSummaryMessage: isSummary,
	})
	if err != nil {
		return Message{}, err
	}
	message, err := s.fromDBItem(dbMessage)
	if err != nil {
		return Message{}, err
	}
	// Clone the message before publishing to avoid race conditions with
	// concurrent modifications to the Parts slice.
	s.Publish(pubsub.CreatedEvent, message.Clone())
	return message, nil
}

func (s *service) DeleteSessionMessages(ctx context.Context, sessionID string) error {
	messages, err := s.List(ctx, sessionID)
	if err != nil {
		return err
	}
	for _, message := range messages {
		if message.SessionID == sessionID {
			err = s.Delete(ctx, message.ID)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *service) Update(ctx context.Context, message Message) error {
	parts, err := marshalParts(message.Parts)
	if err != nil {
		return err
	}
	finishedAt := sql.NullInt64{}
	if f := message.FinishPart(); f != nil {
		finishedAt.Int64 = f.Time
		finishedAt.Valid = true
	}
	err = s.q.UpdateMessage(ctx, db.UpdateMessageParams{
		ID:         message.ID,
		Parts:      string(parts),
		FinishedAt: finishedAt,
	})
	if err != nil {
		return err
	}
	message.UpdatedAt = time.Now().Unix()

	// Publish immediately for finished messages (tool calls completing,
	// end of turn, etc.) so the UI updates without delay. For streaming
	// deltas, throttle to avoid overwhelming the TUI.
	cloned := message.Clone()
	if message.FinishPart() != nil || len(message.ToolCalls()) > 0 && message.ToolCalls()[len(message.ToolCalls())-1].Finished {
		s.flushPendingUpdate(message.ID)
		s.Publish(pubsub.UpdatedEvent, cloned)
	} else {
		s.throttledPublish(message.ID, cloned)
	}
	return nil
}

// throttledPublish coalesces rapid update events for the same message.
// It stores the latest state and schedules a publish after
// publishInterval. If another update arrives before the timer fires,
// the stored state is replaced but no additional timer is created.
func (s *service) throttledPublish(id string, msg Message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.pendingUpdates[id] = msg

	if _, exists := s.pendingTimers[id]; !exists {
		s.pendingTimers[id] = time.AfterFunc(publishInterval, func() {
			s.mu.Lock()
			latest, ok := s.pendingUpdates[id]
			delete(s.pendingUpdates, id)
			delete(s.pendingTimers, id)
			s.mu.Unlock()

			if ok {
				s.Publish(pubsub.UpdatedEvent, latest)
			}
		})
	}
}

// flushPendingUpdate immediately publishes any pending throttled update
// for the given message ID and cancels its timer.
func (s *service) flushPendingUpdate(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if timer, ok := s.pendingTimers[id]; ok {
		timer.Stop()
		delete(s.pendingTimers, id)
	}

	if pending, ok := s.pendingUpdates[id]; ok {
		delete(s.pendingUpdates, id)
		s.Publish(pubsub.UpdatedEvent, pending)
	}
}

// cancelPendingUpdate cancels any pending throttled update for the
// given message ID without publishing.
func (s *service) cancelPendingUpdate(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if timer, ok := s.pendingTimers[id]; ok {
		timer.Stop()
		delete(s.pendingTimers, id)
	}
	delete(s.pendingUpdates, id)
}

func (s *service) Get(ctx context.Context, id string) (Message, error) {
	dbMessage, err := s.q.GetMessage(ctx, id)
	if err != nil {
		return Message{}, err
	}
	return s.fromDBItem(dbMessage)
}

func (s *service) List(ctx context.Context, sessionID string) ([]Message, error) {
	dbMessages, err := s.q.ListMessagesBySession(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	messages := make([]Message, len(dbMessages))
	for i, dbMessage := range dbMessages {
		messages[i], err = s.fromDBItem(dbMessage)
		if err != nil {
			return nil, err
		}
	}
	return messages, nil
}

func (s *service) ListUserMessages(ctx context.Context, sessionID string) ([]Message, error) {
	dbMessages, err := s.q.ListUserMessagesBySession(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	messages := make([]Message, len(dbMessages))
	for i, dbMessage := range dbMessages {
		messages[i], err = s.fromDBItem(dbMessage)
		if err != nil {
			return nil, err
		}
	}
	return messages, nil
}

func (s *service) ListAllUserMessages(ctx context.Context) ([]Message, error) {
	dbMessages, err := s.q.ListAllUserMessages(ctx)
	if err != nil {
		return nil, err
	}
	messages := make([]Message, len(dbMessages))
	for i, dbMessage := range dbMessages {
		messages[i], err = s.fromDBItem(dbMessage)
		if err != nil {
			return nil, err
		}
	}
	return messages, nil
}

func (s *service) fromDBItem(item db.Message) (Message, error) {
	parts, err := unmarshalParts([]byte(item.Parts))
	if err != nil {
		return Message{}, err
	}
	return Message{
		ID:               item.ID,
		SessionID:        item.SessionID,
		Role:             MessageRole(item.Role),
		Parts:            parts,
		Model:            item.Model.String,
		Provider:         item.Provider.String,
		CreatedAt:        item.CreatedAt,
		UpdatedAt:        item.UpdatedAt,
		IsSummaryMessage: item.IsSummaryMessage != 0,
	}, nil
}

type partType string

const (
	reasoningType  partType = "reasoning"
	textType       partType = "text"
	imageURLType   partType = "image_url"
	binaryType     partType = "binary"
	toolCallType   partType = "tool_call"
	toolResultType partType = "tool_result"
	finishType     partType = "finish"
)

type partWrapper struct {
	Type partType    `json:"type"`
	Data ContentPart `json:"data"`
}

func marshalParts(parts []ContentPart) ([]byte, error) {
	wrappedParts := make([]partWrapper, len(parts))

	for i, part := range parts {
		var typ partType

		switch part.(type) {
		case ReasoningContent:
			typ = reasoningType
		case TextContent:
			typ = textType
		case ImageURLContent:
			typ = imageURLType
		case BinaryContent:
			typ = binaryType
		case ToolCall:
			typ = toolCallType
		case ToolResult:
			typ = toolResultType
		case Finish:
			typ = finishType
		default:
			return nil, fmt.Errorf("unknown part type: %T", part)
		}

		wrappedParts[i] = partWrapper{
			Type: typ,
			Data: part,
		}
	}
	return json.Marshal(wrappedParts)
}

func unmarshalParts(data []byte) ([]ContentPart, error) {
	temp := []json.RawMessage{}

	if err := json.Unmarshal(data, &temp); err != nil {
		return nil, err
	}

	parts := make([]ContentPart, 0)

	for _, rawPart := range temp {
		var wrapper struct {
			Type partType        `json:"type"`
			Data json.RawMessage `json:"data"`
		}

		if err := json.Unmarshal(rawPart, &wrapper); err != nil {
			return nil, err
		}

		switch wrapper.Type {
		case reasoningType:
			part := ReasoningContent{}
			if err := json.Unmarshal(wrapper.Data, &part); err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case textType:
			part := TextContent{}
			if err := json.Unmarshal(wrapper.Data, &part); err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case imageURLType:
			part := ImageURLContent{}
			if err := json.Unmarshal(wrapper.Data, &part); err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case binaryType:
			part := BinaryContent{}
			if err := json.Unmarshal(wrapper.Data, &part); err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case toolCallType:
			part := ToolCall{}
			if err := json.Unmarshal(wrapper.Data, &part); err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case toolResultType:
			part := ToolResult{}
			if err := json.Unmarshal(wrapper.Data, &part); err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case finishType:
			part := Finish{}
			if err := json.Unmarshal(wrapper.Data, &part); err != nil {
				return nil, err
			}
			parts = append(parts, part)
		default:
			return nil, fmt.Errorf("unknown part type: %s", wrapper.Type)
		}
	}

	return parts, nil
}
