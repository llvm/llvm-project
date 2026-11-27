//===- OmptAsserter.h - Asserter-related classes, enums, etc. ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Contains all asserter-related class declarations and important enums.
///
//===----------------------------------------------------------------------===//

#ifndef OPENMP_TOOLS_OMPTEST_INCLUDE_OMPTASSERTER_H
#define OPENMP_TOOLS_OMPTEST_INCLUDE_OMPTASSERTER_H

#include "Logging.h"
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

enum class AssertMode { Strict, Relaxed };
enum class AssertState { Pass, Fail };

/// General base class for the subscriber/notification pattern in
/// OmptCallbackHandler. Derived classes need to implement the notify method.
class OmptListener {
public:
  virtual ~OmptListener() = default;

  /// Called for each registered OMPT event of the OmptCallbackHandler
  virtual void notify(omptest::OmptAssertEvent &&AE) = 0;

  /// Control whether this asserter should be considered 'active'.
  void setActive(bool Enabled);

  /// Check if this asserter is considered 'active'.
  bool isActive();

  /// Check if the given event type is from the set of suppressed event types.
  bool isSuppressedEventType(omptest::internal::EventTy EvTy);

  /// Remove the given event type to the set of suppressed events.
  void permitEvent(omptest::internal::EventTy EvTy);

  /// Add the given event type to the set of suppressed events.
  void suppressEvent(omptest::internal::EventTy EvTy);

private:
  bool Active{true};

  // Add event types to the set of suppressed events by default.
  std::set<omptest::internal::EventTy> SuppressedEvents{
      omptest::internal::EventTy::ThreadBegin,
      omptest::internal::EventTy::ThreadEnd,
      omptest::internal::EventTy::ParallelBegin,
      omptest::internal::EventTy::ParallelEnd,
      omptest::internal::EventTy::Work,
      omptest::internal::EventTy::Dispatch,
      omptest::internal::EventTy::TaskCreate,
      omptest::internal::EventTy::Dependences,
      omptest::internal::EventTy::TaskDependence,
      omptest::internal::EventTy::TaskSchedule,
      omptest::internal::EventTy::ImplicitTask,
      omptest::internal::EventTy::Masked,
      omptest::internal::EventTy::SyncRegion,
      omptest::internal::EventTy::MutexAcquire,
      omptest::internal::EventTy::Mutex,
      omptest::internal::EventTy::NestLock,
      omptest::internal::EventTy::Flush,
      omptest::internal::EventTy::Cancel};
};

/// Base class for asserting on OMPT events
class OmptAsserter : public OmptListener {
public:
  OmptAsserter();
  virtual ~OmptAsserter() = default;

  /// Add an event to the asserter's internal data structure.
  virtual void insert(omptest::OmptAssertEvent &&AE);

  /// Called from the CallbackHandler with a corresponding AssertEvent to which
  /// callback was handled.
  void notify(omptest::OmptAssertEvent &&AE) override;

  /// Implemented in subclasses to implement what should actually be done with
  /// the notification.
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) = 0;

  /// Get the number of currently remaining events, with: ObserveState::Always.
  virtual size_t getRemainingEventCount() = 0;

  /// Get the total number of received, effective notifications.
  int getNotificationCount() { return NumNotifications; }

  /// Get the total number of successful assertion checks.
  int getSuccessfulAssertionCount() { return NumSuccessfulAsserts; }

  /// Get the asserter's current operationmode: e.g.: Strict or Relaxed.
  AssertMode getOperationMode() { return OperationMode; }

  /// Return the asserter's current state.
  omptest::AssertState getState() { return State; }

  /// Determine and return the asserter's state.
  virtual omptest::AssertState checkState();

  /// Accessor for the event group interface.
  std::shared_ptr<OmptEventGroupInterface> getEventGroups() const {
    return EventGroups;
  }

  /// Accessor for the event group interface.
  std::shared_ptr<logging::Logger> getLog() const { return Log; }

  /// Check the observed events' group association. If the event indicates the
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
  omptest::AssertState State{omptest::AssertState::Pass};

  /// Mutex to avoid data races w.r.t. event notifications and/or insertions.
  std::mutex AssertMutex;

  /// Pointer to the OmptEventGroupInterface.
  std::shared_ptr<OmptEventGroupInterface> EventGroups{nullptr};

  /// Pointer to the logging instance.
  std::shared_ptr<logging::Logger> Log{nullptr};

  /// Operation mode during assertion / notification.
  AssertMode OperationMode{AssertMode::Strict};

  /// The total number of effective notifications. For example, if specific
  /// notifications are to be ignored, they will not count towards this total.
  int NumNotifications{0};

  /// The number of successful assertion checks.
  int NumSuccessfulAsserts{0};

private:
  /// Mutex for creating/accessing the singleton members
  static std::mutex StaticMemberAccessMutex;

  /// Static member to manage the singleton event group interface instance
  static std::weak_ptr<OmptEventGroupInterface> EventGroupInterfaceInstance;

  /// Static member to manage the singleton logging instance
  static std::weak_ptr<logging::Logger> LoggingInstance;
};

/// Class that can assert in a sequenced fashion, i.e., events have to occur in
/// the order they were registered
class OmptSequencedAsserter : public OmptAsserter {
public:
  OmptSequencedAsserter() : OmptAsserter(), NextEvent(0) {}

  /// Add the event to the in-sequence set of events that the asserter should
  /// check for.
  void insert(omptest::OmptAssertEvent &&AE) override;

  /// Implements the asserter's actual logic
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) override;

  size_t getRemainingEventCount() override;

  omptest::AssertState checkState() override;

  bool AssertionSuspended{false};

protected:
  /// Notification helper function, implementing SyncPoint logic. Returns true
  /// in case of consumed event, indicating early exit of notification.
  bool consumeSyncPoint(const omptest::OmptAssertEvent &AE);

  /// Notification helper function, implementing excess event notification
  /// logic. Returns true when no more events were expected, indicating early
  /// exit of notification.
  bool checkExcessNotify(const omptest::OmptAssertEvent &AE);

  /// Notification helper function, implementing Suspend logic. Returns true
  /// in case of consumed event, indicating early exit of notification.
  bool consumeSuspend();

  /// Notification helper function, implementing regular event notification
  /// logic. Returns true when a matching event was encountered, indicating
  /// early exit of notification.
  bool consumeRegularEvent(const omptest::OmptAssertEvent &AE);

public:
  /// Index of the next, expected event.
  size_t NextEvent{0};
  std::vector<omptest::OmptAssertEvent> Events{};
};

/// Class that asserts with set semantics, i.e., unordered
struct OmptEventAsserter : public OmptAsserter {
  OmptEventAsserter() : OmptAsserter(), NumEvents(0), Events() {}

  /// Add the event to the set of events that the asserter should check for.
  void insert(omptest::OmptAssertEvent &&AE) override;

  /// Implements the asserter's logic
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) override;

  size_t getRemainingEventCount() override;

  omptest::AssertState checkState() override;

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

  /// Called from the CallbackHandler with a corresponding AssertEvent to which
  /// callback was handled.
  void notify(omptest::OmptAssertEvent &&AE) override;

private:
  std::ostream &OutStream;
};

/// This class provides the members and methods to manage event groups and
/// SyncPoints in conjunction with asserters. Most importantly it maintains a
/// coherent view of active and past events or SyncPoints.
class OmptEventGroupInterface {
public:
  OmptEventGroupInterface() = default;
  ~OmptEventGroupInterface() = default;

  /// Non-copyable and non-movable
  OmptEventGroupInterface(const OmptEventGroupInterface &) = delete;
  OmptEventGroupInterface &operator=(const OmptEventGroupInterface &) = delete;
  OmptEventGroupInterface(OmptEventGroupInterface &&) = delete;
  OmptEventGroupInterface &operator=(OmptEventGroupInterface &&) = delete;

  /// Add given group to the set of active event groups. Effectively connecting
  /// the given groupname (expected) with a target region id (observed).
  bool addActiveEventGroup(const std::string &GroupName,
                           omptest::AssertEventGroup Group);

  /// Move given group from the set of active event groups to the set of
  /// previously active event groups.
  bool deprecateActiveEventGroup(const std::string &GroupName);

  /// Check if given group is currently part of the active event groups.
  bool checkActiveEventGroups(const std::string &GroupName,
                              omptest::AssertEventGroup Group);

  /// Check if given group is currently part of the deprecated event groups.
  bool checkDeprecatedEventGroups(const std::string &GroupName,
                                  omptest::AssertEventGroup Group);

private:
  mutable std::mutex GroupMutex;
  std::map<std::string, omptest::AssertEventGroup> ActiveEventGroups{};
  std::map<std::string, omptest::AssertEventGroup> DeprecatedEventGroups{};
  std::set<std::string> EncounteredSyncPoints{};
};

} // namespace omptest

#endif
