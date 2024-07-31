#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTER_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTER_H

#include "OmptAssertEvent.h"

#include <cassert>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace omptest {

// Forward declaration.
class OmptEventGroupInterface;

enum class AssertMode { strict, relaxed };
enum class AssertState { pass, fail };

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
  OmptAsserter(OmptEventGroupInterface *ParentTC) : TC(ParentTC) {}

  /// Add an event to the asserter's internal data structure.
  virtual void insert(omptest::OmptAssertEvent &&AE);

  /// Called from the CallbackHandler with a corresponding AssertEvent to which
  /// callback was handled.
  void notify(omptest::OmptAssertEvent &&AE) override;

  /// Implemented in subclasses to implement what should actually be done with
  /// the notification.
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) = 0;

  /// Report an error for a single event.
  void reportError(const omptest::OmptAssertEvent &OffendingEvent,
                   const std::string &Message);

  void reportError(const omptest::OmptAssertEvent &AwaitedEvent,
                   const omptest::OmptAssertEvent &OffendingEvent,
                   const std::string &Message);

  /// Get the number of currently remaining events, with: ObserveState::always.
  virtual size_t getRemainingEventCount() = 0;

  /// Determine and return the asserter's current state.
  virtual omptest::AssertState getState();

  /// Check the given events' group association. If the event indicates the
  /// begin/end of an OpenMP target region, we will create/deprecate the
  /// expected event's group. Return true if the expected event group exists
  /// (and is active), otherwise: false. Note: BufferRecords may also match with
  /// deprecated groups as they may be delivered asynchronously.
  bool verifyEventGroups(const omptest::OmptAssertEvent &ExpectedEvent,
                         const omptest::OmptAssertEvent &ObservedEvent);

  /// Set the asserter's mode of operation w.r.t. assertion.
  void setOperationMode(AssertMode Mode);

protected:
  /// The asserter's current state.
  omptest::AssertState State{omptest::AssertState::pass};

  /// Mutex to avoid data races w.r.t. event notifications and/or insertions.
  std::mutex AssertMutex;

  /// Pointer to the parent TestCase.
  OmptEventGroupInterface *TC{nullptr};

  /// Operation mode during assertion / notification.
  AssertMode OperationMode{AssertMode::strict};
};

/// Class that can assert in a sequenced fashion, i.e., events have to occur in
/// the order they were registered
struct OmptSequencedAsserter : public OmptAsserter {
  OmptSequencedAsserter(OmptEventGroupInterface *ParentTC = nullptr)
      : OmptAsserter(ParentTC), NextEvent(0) {}

  /// Add the event to the in-sequence set of events that the asserter should
  /// check for.
  void insert(omptest::OmptAssertEvent &&AE) override;

  /// Implements the asserter's actual logic
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) override;

  size_t getRemainingEventCount() override;

  omptest::AssertState getState() override;

  bool AssertionSuspended{false};
  int NumAssertSuccesses{0};
  int NumNotifications{0};
  size_t NextEvent{0};
  std::vector<omptest::OmptAssertEvent> Events{};
};

/// Class that asserts with set semantics, i.e., unordered
struct OmptEventAsserter : public OmptAsserter {
  OmptEventAsserter(OmptEventGroupInterface *ParentTC = nullptr)
      : OmptAsserter(ParentTC), NumEvents(0), Events() {}

  /// Add the event to the set of events that the asserter should check for.
  void insert(omptest::OmptAssertEvent &&AE) override;

  /// Implements the asserter's logic
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) override;

  size_t getRemainingEventCount() override;

  omptest::AssertState getState() override;

  int NumAssertSuccesses{0};
  int NumNotifications{0};
  size_t NumEvents{0};

  /// For now use vector (but do set semantics)
  // TODO std::unordered_set?
  std::vector<omptest::OmptAssertEvent> Events{};
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

/// This class provides the members and methods to manage event groups and
/// SyncPoints in conjunction with asserters. Most importantly it maintains a
/// coherent view of active and past events or SyncPoints.
class OmptEventGroupInterface {
public:
  OmptEventGroupInterface()
      : SequenceAsserter{std::make_unique<OmptSequencedAsserter>(this)},
        SetAsserter{std::make_unique<OmptEventAsserter>(this)},
        EventReporter{std::make_unique<OmptEventReporter>()} {}

  // Add given group to the set of active event groups.
  bool addActiveEventGroup(const std::string &GroupName,
                           omptest::AssertEventGroup Group);

  // Move given group from the set of active event groups to the set of
  // previously active event groups.
  bool deprecateActiveEventGroup(const std::string &GroupName);

  // Check if given group is currently part of the active event groups.
  bool checkActiveEventGroups(const std::string &GroupName,
                              omptest::AssertEventGroup Group);

  // Check if given group is currently part of the deprecated event groups.
  bool checkDeprecatedEventGroups(const std::string &GroupName,
                                  omptest::AssertEventGroup Group);

  std::unique_ptr<OmptSequencedAsserter> SequenceAsserter;
  std::unique_ptr<OmptEventAsserter> SetAsserter;
  std::unique_ptr<OmptEventReporter> EventReporter;

private:
  std::mutex GroupMutex;
  std::map<std::string, omptest::AssertEventGroup> ActiveEventGroups{};
  std::map<std::string, omptest::AssertEventGroup> DeprecatedEventGroups{};
  std::set<std::string> EncounteredSyncPoints{};
};

} // namespace omptest

#endif
