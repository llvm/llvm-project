#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_ASSERTMACROS_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_ASSERTMACROS_H

#define OMPTEST_EXCLUDE_EVENT omptest::ObserveState::never
#define OMPTEST_REQUIRE_EVENT omptest::ObserveState::always

/// ASSERT MACROS TO BE USED BY THE USER

// Not implemented yet
#define OMPT_ASSERT_EVENT(Event, ...)

// Handle a minimum unordered set of events
// Required events
#define OMPT_ASSERT_SET_EVENT(Name, Group, EventTy, ...)                       \
  this->SetAsserter.insert(OmptAssertEvent::EventTy(                           \
      Name, Group, OMPTEST_REQUIRE_EVENT, __VA_ARGS__));
#define OMPT_ASSERT_SET(EventTy, ...)                                          \
  OMPT_ASSERT_SET_EVENT("", "", EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SET_GROUP(Group, EventTy, ...)                             \
  OMPT_ASSERT_SET_EVENT("", Group, EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SET_NAMED(Name, EventTy, ...)                              \
  OMPT_ASSERT_SET_EVENT(Name, "", EventTy, __VA_ARGS__)
// Excluded ("NOT") events
#define OMPT_ASSERT_SET_EVENT_NOT(Name, Group, EventTy, ...)                   \
  this->SetAsserter.insert(OmptAssertEvent::EventTy(                           \
      Name, Group, OMPTEST_EXCLUDE_EVENT, __VA_ARGS__));
#define OMPT_ASSERT_SET_NOT(EventTy, ...)                                      \
  OMPT_ASSERT_SET_EVENT_NOT("", "", EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SET_GROUP_NOT(Group, EventTy, ...)                         \
  OMPT_ASSERT_SET_EVENT_NOT("", Group, EventTy, __VA_ARGS__)
#define OMPT_ASSERT_SET_NAMED_NOT(Name, EventTy, ...)                          \
  OMPT_ASSERT_SET_EVENT_NOT(Name, "", EventTy, __VA_ARGS__)

// Handle an exact sequence of events
// Required events
#define OMPT_ASSERT_SEQUENCE_EVENT(Name, Group, EventTy, ...)                  \
  this->SequenceAsserter.insert(OmptAssertEvent::EventTy(                      \
      Name, Group, OMPTEST_REQUIRE_EVENT, __VA_ARGS__));
#define OMPT_ASSERT_SEQUENCE(EventTy, ...)                                     \
  OMPT_ASSERT_SEQUENCE_EVENT("", "", EventTy, __VA_ARGS__)
#define OMPT_ASSERT_GROUPED_SEQUENCE(Group, EventTy, ...)                      \
  OMPT_ASSERT_SEQUENCE_EVENT("", Group, EventTy, __VA_ARGS__)
#define OMPT_ASSERT_NAMED_SEQUENCE(Name, EventTy, ...)                         \
  OMPT_ASSERT_SEQUENCE_EVENT(Name, "", EventTy, __VA_ARGS__)
// Excluded ("NOT") events
#define OMPT_ASSERT_SEQUENCE_EVENT_NOT(Name, Group, EventTy, ...)              \
  this->SequenceAsserter.insert(OmptAssertEvent::EventTy(                      \
      Name, Group, OMPTEST_EXCLUDE_EVENT, __VA_ARGS__));
#define OMPT_ASSERT_SEQUENCE_NOT(EventTy, ...)                                 \
  OMPT_ASSERT_SEQUENCE_EVENT_NOT("", "", EventTy, __VA_ARGS__)
#define OMPT_ASSERT_GROUPED_SEQUENCE_NOT(Group, EventTy, ...)                  \
  OMPT_ASSERT_SEQUENCE_EVENT_NOT("", Group, EventTy, __VA_ARGS__)
#define OMPT_ASSERT_NAMED_SEQUENCE_NOT(Name, EventTy, ...)                     \
  OMPT_ASSERT_SEQUENCE_EVENT_NOT(Name, "", EventTy, __VA_ARGS__)
// Special command: suspend active assertion
#define OMPT_ASSERT_SEQUENCE_SUSPEND()                                         \
  this->SequenceAsserter.insert(                                               \
      OmptAssertEvent::Asserter("", "", OMPTEST_REQUIRE_EVENT));
#define OMPT_ASSERT_SEQUENCE_ONLY(EventTy, ...)                                \
  OMPT_ASSERT_SEQUENCE_SUSPEND()                                               \
  OMPT_ASSERT_SEQUENCE_EVENT("", "", EventTy, __VA_ARGS__)                     \
  OMPT_ASSERT_SEQUENCE_SUSPEND()
#define OMPT_ASSERT_GROUPED_SEQUENCE_ONLY(Group, EventTy, ...)                 \
  OMPT_ASSERT_SEQUENCE_SUSPEND()                                               \
  OMPT_ASSERT_SEQUENCE_EVENT("", Group, EventTy, __VA_ARGS__)                  \
  OMPT_ASSERT_SEQUENCE_SUSPEND()
#define OMPT_ASSERT_NAMED_SEQUENCE_ONLY(Name, EventTy, ...)                    \
  OMPT_ASSERT_SEQUENCE_SUSPEND()                                               \
  OMPT_ASSERT_SEQUENCE_EVENT(Name, "", EventTy, __VA_ARGS__)                   \
  OMPT_ASSERT_SEQUENCE_SUSPEND()

// Enable / disable asserters entirely
#define OMPT_ASSERTER_DISABLE(AsserterName) this->AsserterName.setActive(false);
#define OMPT_ASSERTER_ENABLE(AsserterName) this->AsserterName.setActive(true);
#define OMPT_ASSERT_SET_DISABLE() OMPT_ASSERTER_DISABLE(SetAsserter)
#define OMPT_ASSERT_SET_ENABLE() OMPT_ASSERTER_ENABLE(SetAsserter)
#define OMPT_REPORT_EVENT_DISABLE() OMPT_ASSERTER_DISABLE(EventReporter)
#define OMPT_REPORT_EVENT_ENABLE() OMPT_ASSERTER_ENABLE(EventReporter)
#define OMPT_ASSERT_SEQUENCE_DISABLE() OMPT_ASSERTER_DISABLE(SequenceAsserter)
#define OMPT_ASSERT_SEQUENCE_ENABLE() OMPT_ASSERTER_ENABLE(SequenceAsserter)

// Enable / disable certain event types for asserters
#define OMPT_ASSERTER_PERMIT_EVENT(Asserter, EventTy)                          \
  this->Asserter.permitEvent(EventTy);
#define OMPT_ASSERTER_SUPPRESS_EVENT(Asserter, EventTy)                        \
  this->Asserter.suppressEvent(EventTy);
#define OMPT_PERMIT_EVENT(EventTy)                                             \
  OMPT_ASSERTER_PERMIT_EVENT(SetAsserter, EventTy);                            \
  OMPT_ASSERTER_PERMIT_EVENT(EventReporter, EventTy);                          \
  OMPT_ASSERTER_PERMIT_EVENT(SequenceAsserter, EventTy);
#define OMPT_SUPPRESS_EVENT(EventTy)                                           \
  OMPT_ASSERTER_SUPPRESS_EVENT(SetAsserter, EventTy);                          \
  OMPT_ASSERTER_SUPPRESS_EVENT(EventReporter, EventTy);                        \
  OMPT_ASSERTER_SUPPRESS_EVENT(SequenceAsserter, EventTy);

#endif // include guard