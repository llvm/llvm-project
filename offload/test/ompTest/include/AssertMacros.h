#ifndef OFFLOAD_TEST_OMPTEST_INCLUDE_ASSERTMACROS_H
#define OFFLOAD_TEST_OMPTEST_INCLUDE_ASSERTMACROS_H

#define OMPTEST_EXCLUDED_EVENT omptest::ObserveState::never
#define OMPTEST_REQUIRED_EVENT omptest::ObserveState::always

/// ASSERT MACROS TO BE USED BY THE USER

#define OMPT_GENERATE_EVENTS(NumberOfCopies, EventMacro)                       \
  for (size_t i = 0; i < NumberOfCopies; ++i) {                                \
    EventMacro                                                                 \
  }

// Handle a minimum unordered set of events
// Required events
#define OMPT_ASSERT_SET_EVENT(Name, Group, EventTy, ...)                       \
  SetAsserter->insert(OmptAssertEvent::EventTy(                                \
      Name, Group, OMPTEST_REQUIRED_EVENT, __VA_ARGS__));
#define OMPT_ASSERT_SET(EventTy, ...)                                          \
  OMPT_ASSERT_SET_EVENT("", "", EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SET_GROUPED(Group, EventTy, ...)                           \
  OMPT_ASSERT_SET_EVENT("", Group, EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SET_NAMED(Name, EventTy, ...)                              \
  OMPT_ASSERT_SET_EVENT(Name, "", EventTy, __VA_ARGS__)
// Excluded ("NOT") events
#define OMPT_ASSERT_SET_EVENT_NOT(Name, Group, EventTy, ...)                   \
  SetAsserter->insert(OmptAssertEvent::EventTy(                                \
      Name, Group, OMPTEST_EXCLUDED_EVENT, __VA_ARGS__));
#define OMPT_ASSERT_SET_NOT(EventTy, ...)                                      \
  OMPT_ASSERT_SET_EVENT_NOT("", "", EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SET_GROUPED_NOT(Group, EventTy, ...)                       \
  OMPT_ASSERT_SET_EVENT_NOT("", Group, EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SET_NAMED_NOT(Name, EventTy, ...)                          \
  OMPT_ASSERT_SET_EVENT_NOT(Name, "", EventTy, __VA_ARGS__)

// Handle an exact sequence of events
// Required events
#define OMPT_ASSERT_SEQUENCE_EVENT(Name, Group, EventTy, ...)                  \
  SequenceAsserter->insert(OmptAssertEvent::EventTy(                           \
      Name, Group, OMPTEST_REQUIRED_EVENT, __VA_ARGS__));
#define OMPT_ASSERT_SEQUENCE(EventTy, ...)                                     \
  OMPT_ASSERT_SEQUENCE_EVENT("", "", EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SEQUENCE_GROUPED(Group, EventTy, ...)                      \
  OMPT_ASSERT_SEQUENCE_EVENT("", Group, EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SEQUENCE_NAMED(Name, EventTy, ...)                         \
  OMPT_ASSERT_SEQUENCE_EVENT(Name, "", EventTy, __VA_ARGS__)
// Excluded ("NOT") events
#define OMPT_ASSERT_SEQUENCE_EVENT_NOT(Name, Group, EventTy, ...)              \
  SequenceAsserter->insert(OmptAssertEvent::EventTy(                           \
      Name, Group, OMPTEST_EXCLUDED_EVENT, __VA_ARGS__));
#define OMPT_ASSERT_SEQUENCE_NOT(EventTy, ...)                                 \
  OMPT_ASSERT_SEQUENCE_EVENT_NOT("", "", EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SEQUENCE_GROUPED_NOT(Group, EventTy, ...)                  \
  OMPT_ASSERT_SEQUENCE_EVENT_NOT("", Group, EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SEQUENCE_NAMED_NOT(Name, EventTy, ...)                     \
  OMPT_ASSERT_SEQUENCE_EVENT_NOT(Name, "", EventTy, __VA_ARGS__)
// Special command: suspend active assertion
// The created event is not correlated to any observed event
#define OMPT_ASSERT_SEQUENCE_SUSPEND()                                         \
  SequenceAsserter->insert(                                                    \
      OmptAssertEvent::AssertionSuspend("", "", OMPTEST_EXCLUDED_EVENT));
#define OMPT_ASSERT_SEQUENCE_ONLY(EventTy, ...)                                \
  OMPT_ASSERT_SEQUENCE_SUSPEND()                                               \
  OMPT_ASSERT_SEQUENCE_EVENT("", "", EventTy, __VA_ARGS__)                     \
  OMPT_ASSERT_SEQUENCE_SUSPEND()
#define OMPT_ASSERT_SEQUENCE_GROUPED_ONLY(Group, EventTy, ...)                 \
  OMPT_ASSERT_SEQUENCE_SUSPEND()                                               \
  OMPT_ASSERT_SEQUENCE_EVENT("", Group, EventTy, __VA_ARGS__)                  \
  OMPT_ASSERT_SEQUENCE_SUSPEND()
#define OMPT_ASSERT_SEQUENCE_NAMED_ONLY(Name, EventTy, ...)                    \
  OMPT_ASSERT_SEQUENCE_SUSPEND()                                               \
  OMPT_ASSERT_SEQUENCE_EVENT(Name, "", EventTy, __VA_ARGS__)                   \
  OMPT_ASSERT_SEQUENCE_SUSPEND()

#define OMPT_ASSERTER_MODE_STRICT(Asserter)                                    \
  Asserter->setOperationMode(AssertMode::strict);
#define OMPT_ASSERTER_MODE_RELAXED(Asserter)                                   \
  Asserter->setOperationMode(AssertMode::relaxed);
#define OMPT_ASSERT_SEQUENCE_MODE_STRICT()                                     \
  OMPT_ASSERTER_MODE_STRICT(SequenceAsserter)
#define OMPT_ASSERT_SEQUENCE_MODE_RELAXED()                                    \
  OMPT_ASSERTER_MODE_RELAXED(SequenceAsserter)
#define OMPT_ASSERT_SET_MODE_STRICT() OMPT_ASSERTER_MODE_STRICT(SetAsserter)
#define OMPT_ASSERT_SET_MODE_RELAXED() OMPT_ASSERTER_MODE_RELAXED(SetAsserter)

// Enable / disable asserters entirely
#define OMPT_ASSERTER_DISABLE(Asserter) Asserter->setActive(false);
#define OMPT_ASSERTER_ENABLE(Asserter) Asserter->setActive(true);
#define OMPT_ASSERT_SET_DISABLE() OMPT_ASSERTER_DISABLE(SetAsserter)
#define OMPT_ASSERT_SET_ENABLE() OMPT_ASSERTER_ENABLE(SetAsserter)
#define OMPT_ASSERT_SEQUENCE_DISABLE() OMPT_ASSERTER_DISABLE(SequenceAsserter)
#define OMPT_ASSERT_SEQUENCE_ENABLE() OMPT_ASSERTER_ENABLE(SequenceAsserter)
#define OMPT_REPORT_EVENT_DISABLE() OMPT_ASSERTER_DISABLE(EventReporter)
#define OMPT_REPORT_EVENT_ENABLE() OMPT_ASSERTER_ENABLE(EventReporter)

// Enable / disable certain event types for asserters
#define OMPT_ASSERTER_PERMIT_EVENT(Asserter, EventTy)                          \
  Asserter->permitEvent(EventTy);
#define OMPT_ASSERTER_SUPPRESS_EVENT(Asserter, EventTy)                        \
  Asserter->suppressEvent(EventTy);
#define OMPT_PERMIT_EVENT(EventTy)                                             \
  OMPT_ASSERTER_PERMIT_EVENT(SetAsserter, EventTy);                            \
  OMPT_ASSERTER_PERMIT_EVENT(EventReporter, EventTy);                          \
  OMPT_ASSERTER_PERMIT_EVENT(SequenceAsserter, EventTy);
#define OMPT_SUPPRESS_EVENT(EventTy)                                           \
  OMPT_ASSERTER_SUPPRESS_EVENT(SetAsserter, EventTy);                          \
  OMPT_ASSERTER_SUPPRESS_EVENT(EventReporter, EventTy);                        \
  OMPT_ASSERTER_SUPPRESS_EVENT(SequenceAsserter, EventTy);

#define OMPT_ASSERT_SYNC_POINT(SyncPointName)                                  \
  flush_traced_devices();                                                      \
  OmptCallbackHandler::get().handleAssertionSyncPoint(SyncPointName);

#endif
