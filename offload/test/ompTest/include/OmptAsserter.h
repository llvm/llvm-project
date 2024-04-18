#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTER_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTER_H

#include "OmptAssertEvent.h"

#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <vector>

/// General base class for the subscriber/notification pattern in
/// OmptCallbachHandler. Derived classes need to implement the notify method.
class OmptListener {
public:
  virtual ~OmptListener() = default;

  /// Called for each registered OMPT event of the OmptCallbackHandler
  virtual void notify(omptest::OmptAssertEvent &&AE) = 0;

  /// Control whether this asserter should be considered 'active'.
  void setActive(bool Enabled);

  /// Check if this asserter is considered 'active'.
  bool isActive();

  /// Check if the given event type is in the set of suppressed event types.
  bool isSuppressedEventType(omptest::internal::EventTy EvTy);

  /// Remove the given event type to the set of suppressed events.
  void permitEvent(omptest::internal::EventTy EvTy);

  /// Add the given event type to the set of suppressed events.
  void suppressEvent(omptest::internal::EventTy EvTy);

private:
  bool Active{true};

  // For now we add event types to the set of suppressed events by default.
  // This is necessary because AOMP currently does not handle these events.
  std::set<omptest::internal::EventTy> SuppressedEvents{
      omptest::internal::EventTy::ParallelBegin,
      omptest::internal::EventTy::ParallelEnd,
      omptest::internal::EventTy::ThreadBegin,
      omptest::internal::EventTy::ThreadEnd,
      omptest::internal::EventTy::ImplicitTask,
      omptest::internal::EventTy::TaskCreate,
      omptest::internal::EventTy::TaskSchedule};
};

/// Base class for asserting on OMPT events
class OmptAsserter : public OmptListener {
public:
  virtual void insert(omptest::OmptAssertEvent &&AE);

  // Called from the CallbackHandler with a corresponding AssertEvent to which
  // callback was handled.
  void notify(omptest::OmptAssertEvent &&AE) override;

  /// Implemented in subclasses to implement what should actually be done with
  /// the notification.
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) = 0;

  /// Report an error for a single event
  void reportError(const omptest::OmptAssertEvent &OffendingEvent,
                   const std::string &Message);

  void reportError(const omptest::OmptAssertEvent &AwaitedEvent,
                   const omptest::OmptAssertEvent &OffendingEvent,
                   const std::string &Message);

  virtual omptest::AssertState getState();

  bool verifyEventGroups(const omptest::OmptAssertEvent &ExpectedEvent,
                         const omptest::OmptAssertEvent &ObservedEvent);

protected:
  omptest::AssertState State{omptest::AssertState::pass};

  // This map stores an AssertEventGroup under the given groupname as key.
  // Using these groups allows to verify e.g. if a given operation belongs to
  // certain target regions -- i.e. if the group was specified.
  std::map<std::string, omptest::AssertEventGroup> EventGroups{};
};

/// Class that can assert in a sequenced fashion, i.e., events have to occur in
/// the order they were registered
struct OmptSequencedAsserter : public OmptAsserter {
  OmptSequencedAsserter() : NextEvent(0), Events() {}

  /// Add the event to the in-sequence set of events that the asserter should
  /// check for.
  void insert(omptest::OmptAssertEvent &&AE) override;

  /// Implements the asserter's actual logic
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) override;

  omptest::AssertState getState() override;

  bool ActiveMode{true};
  int NumAssertSuccesses{0};
  int NumNotifications{0};
  size_t NextEvent{0};
  std::vector<omptest::OmptAssertEvent> Events;
};

/// Class that asserts with set semantics, i.e., unordered
struct OmptEventAsserter : public OmptAsserter {
  void insert(omptest::OmptAssertEvent &&AE) override;

  /// Implements the asserter's logic
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) override;

  omptest::AssertState getState() override;

  int NumAssertSuccesses{0};
  int NumNotifications{0};
  size_t NumEvents{0};

  /// For now use vector (but do set semantics)
  std::vector<omptest::OmptAssertEvent> Events; // TODO std::unordered_set?
};

/// Class that reports the occurred events
class OmptEventReporter : public OmptListener {
public:
  OmptEventReporter(std::ostream &OutStream = std::cout)
      : OutStream(OutStream) {}

  // Called from the CallbackHandler with a corresponding AssertEvent to which
  // callback was handled.
  void notify(omptest::OmptAssertEvent &&AE) override;

private:
  std::ostream &OutStream;
};

#endif
