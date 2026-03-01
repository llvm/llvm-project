//===- OmptAsserter.cpp - Asserter-related implementations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements all asserter-related class methods, like: notifications, handling
/// of groups or determination of the testcase state.
///
//===----------------------------------------------------------------------===//

#include "OmptAsserter.h"
#include "Logging.h"

#include <algorithm>

using namespace omptest;
using namespace internal;

// Initialize static members
std::mutex OmptAsserter::StaticMemberAccessMutex;
std::weak_ptr<OmptEventGroupInterface>
    OmptAsserter::EventGroupInterfaceInstance;
std::weak_ptr<logging::Logger> OmptAsserter::LoggingInstance;

OmptAsserter::OmptAsserter() {
  // Protect static members access
  std::lock_guard<std::mutex> Lock(StaticMemberAccessMutex);

  // Upgrade OmptEventGroupInterface weak_ptr to shared_ptr
  {
    EventGroups = EventGroupInterfaceInstance.lock();
    if (!EventGroups) {
      // Coordinator doesn't exist or was previously destroyed, create a new
      // one.
      EventGroups = std::make_shared<OmptEventGroupInterface>();
      // Store a weak reference to it
      EventGroupInterfaceInstance = EventGroups;
    }
    // EventGroups is now a valid shared_ptr, either to a new or existing
    // instance.
  }

  // Upgrade logging::Logger weak_ptr to shared_ptr
  {
    Log = LoggingInstance.lock();
    if (!Log) {
      // Coordinator doesn't exist or was previously destroyed, create a new
      // one.
      Log = std::make_shared<logging::Logger>();
      // Store a weak reference to it
      LoggingInstance = Log;
    }
    // Log is now a valid shared_ptr, either to a new or existing instance.
  }
}

void OmptListener::setActive(bool Enabled) { Active = Enabled; }

bool OmptListener::isActive() { return Active; }

bool OmptListener::isSuppressedEventType(EventTy EvTy) {
  return SuppressedEvents.find(EvTy) != SuppressedEvents.end();
}

void OmptListener::permitEvent(EventTy EvTy) { SuppressedEvents.erase(EvTy); }

void OmptListener::suppressEvent(EventTy EvTy) {
  SuppressedEvents.insert(EvTy);
}

void OmptAsserter::insert(OmptAssertEvent &&AE) {
  assert(false && "Base class 'insert' has undefined semantics.");
}

void OmptAsserter::notify(OmptAssertEvent &&AE) {
  // Ignore notifications while inactive
  if (!isActive() || isSuppressedEventType(AE.getEventType()))
    return;

  this->notifyImpl(std::move(AE));
}

AssertState OmptAsserter::checkState() { return State; }

bool OmptAsserter::verifyEventGroups(const OmptAssertEvent &ExpectedEvent,
                                     const OmptAssertEvent &ObservedEvent) {
  assert(ExpectedEvent.getEventType() == ObservedEvent.getEventType() &&
         "Type mismatch: Expected != Observed event type");
  assert(EventGroups && "Missing EventGroups interface");

  // Ignore all events within "default" group
  auto GroupName = ExpectedEvent.getEventGroup();

  if (GroupName == "default")
    return true;

  // Get a pointer to the observed internal event
  auto Event = ObservedEvent.getEvent();

  switch (Event->Type) {
  case EventTy::Target:
    if (auto E = static_cast<const internal::Target *>(Event)) {
      if (E->Endpoint == ompt_scope_begin) {
        // Add new group since we entered a Target Region
        EventGroups->addActiveEventGroup(GroupName,
                                         AssertEventGroup{E->TargetId});
      } else if (E->Endpoint == ompt_scope_end) {
        // Deprecate group since we return from a Target Region
        EventGroups->deprecateActiveEventGroup(GroupName);
      }
      return true;
    }
    return false;
  case EventTy::TargetEmi:
    if (auto E = static_cast<const internal::TargetEmi *>(Event)) {
      if (E->Endpoint == ompt_scope_begin) {
        // Add new group since we entered a Target Region
        EventGroups->addActiveEventGroup(
            GroupName, AssertEventGroup{E->TargetData->value});
      } else if (E->Endpoint == ompt_scope_end) {
        // Deprecate group since we return from a Target Region
        EventGroups->deprecateActiveEventGroup(GroupName);
      }
      return true;
    }
    return false;
  case EventTy::TargetDataOp:
    if (auto E = static_cast<const internal::TargetDataOp *>(Event))
      return EventGroups->checkActiveEventGroups(GroupName,
                                                 AssertEventGroup{E->TargetId});

    return false;
  case EventTy::TargetDataOpEmi:
    if (auto E = static_cast<const internal::TargetDataOpEmi *>(Event))
      return EventGroups->checkActiveEventGroups(
          GroupName, AssertEventGroup{E->TargetData->value});

    return false;
  case EventTy::TargetSubmit:
    if (auto E = static_cast<const internal::TargetSubmit *>(Event))
      return EventGroups->checkActiveEventGroups(GroupName,
                                                 AssertEventGroup{E->TargetId});

    return false;
  case EventTy::TargetSubmitEmi:
    if (auto E = static_cast<const internal::TargetSubmitEmi *>(Event))
      return EventGroups->checkActiveEventGroups(
          GroupName, AssertEventGroup{E->TargetData->value});

    return false;
  case EventTy::BufferRecord:
    // BufferRecords are delivered asynchronously: also check deprecated groups.
    if (auto E = static_cast<const internal::BufferRecord *>(Event))
      return (EventGroups->checkActiveEventGroups(
                  GroupName, AssertEventGroup{E->Record.target_id}) ||
              EventGroups->checkDeprecatedEventGroups(
                  GroupName, AssertEventGroup{E->Record.target_id}));
    return false;
  // Some event types do not need any handling
  case EventTy::ThreadBegin:
  case EventTy::ThreadEnd:
  case EventTy::ParallelBegin:
  case EventTy::ParallelEnd:
  case EventTy::Work:
  case EventTy::Dispatch:
  case EventTy::TaskCreate:
  case EventTy::Dependences:
  case EventTy::TaskDependence:
  case EventTy::TaskSchedule:
  case EventTy::ImplicitTask:
  case EventTy::Masked:
  case EventTy::SyncRegion:
  case EventTy::MutexAcquire:
  case EventTy::Mutex:
  case EventTy::NestLock:
  case EventTy::Flush:
  case EventTy::Cancel:
  case EventTy::DeviceInitialize:
  case EventTy::DeviceFinalize:
  case EventTy::DeviceLoad:
  case EventTy::DeviceUnload:
  case EventTy::BufferRequest:
  case EventTy::BufferComplete:
  case EventTy::BufferRecordDeallocation:
    return true;
  // Some event types must not be encountered
  case EventTy::None:
  case EventTy::AssertionSyncPoint:
  case EventTy::AssertionSuspend:
  default:
    Log->log("Observed invalid event type: " + Event->toString(),
             logging::Level::Critical);
    __builtin_unreachable();
  }

  return true;
}

void OmptAsserter::setOperationMode(AssertMode Mode) { OperationMode = Mode; }

void OmptSequencedAsserter::insert(OmptAssertEvent &&AE) {
  std::lock_guard<std::mutex> Lock(AssertMutex);
  Events.emplace_back(std::move(AE));
}

void OmptSequencedAsserter::notifyImpl(OmptAssertEvent &&AE) {
  std::lock_guard<std::mutex> Lock(AssertMutex);
  // Ignore notifications while inactive, or for suppressed events
  if (Events.empty() || !isActive() || isSuppressedEventType(AE.getEventType()))
    return;

  ++NumNotifications;

  // Note: Order of these checks has semantic meaning.
  // (1) Synchronization points should fail if there are remaining events,
  // otherwise pass. (2) Regular notification while no further events are
  // expected: fail. (3) Assertion suspension relies on a next expected event
  // being available. (4) All other cases are considered 'regular' and match the
  // next expected against the observed event. (5+6) Depending on the state /
  // mode we signal failure if no other check has done already, or signaled pass
  // by early-exit.
  if (consumeSyncPoint(AE) ||               // Handle observed SyncPoint event
      checkExcessNotify(AE) ||              // Check for remaining expected
      consumeSuspend() ||                   // Handle requested suspend
      consumeRegularEvent(AE) ||            // Handle regular event
      AssertionSuspended ||                 // Ignore fail, if suspended
      OperationMode == AssertMode::Relaxed) // Ignore fail, if Relaxed op-mode
    return;

  Log->logEventMismatch("[OmptSequencedAsserter] The events are not equal",
                        Events[NextEvent], AE);
  State = AssertState::Fail;
}

bool OmptSequencedAsserter::consumeSyncPoint(
    const omptest::OmptAssertEvent &AE) {
  if (AE.getEventType() == EventTy::AssertionSyncPoint) {
    auto NumRemainingEvents = getRemainingEventCount();
    // Upon encountering a SyncPoint, all events should have been processed
    if (NumRemainingEvents == 0)
      return true;

    Log->logEventMismatch(
        "[OmptSequencedAsserter] Encountered SyncPoint while still awaiting " +
            std::to_string(NumRemainingEvents) + " events. Asserted " +
            std::to_string(NumSuccessfulAsserts) + "/" +
            std::to_string(Events.size()) + " events successfully.",
        AE);
    State = AssertState::Fail;
    return true;
  }

  // Nothing to process: continue.
  return false;
}

bool OmptSequencedAsserter::checkExcessNotify(
    const omptest::OmptAssertEvent &AE) {
  if (NextEvent >= Events.size()) {
    // If we are not expecting any more events and passively asserting: return
    if (AssertionSuspended)
      return true;

    Log->logEventMismatch(
        "[OmptSequencedAsserter] Too many events to check (" +
            std::to_string(NumNotifications) + "). Asserted " +
            std::to_string(NumSuccessfulAsserts) + "/" +
            std::to_string(Events.size()) + " events successfully.",
        AE);
    State = AssertState::Fail;
    return true;
  }

  // Remaining expected events present: continue.
  return false;
}

bool OmptSequencedAsserter::consumeSuspend() {
  // On AssertionSuspend -- enter 'passive' assertion.
  // Since we may encounter multiple, successive AssertionSuspend events, loop
  // until we hit the next non-AssertionSuspend event.
  while (Events[NextEvent].getEventType() == EventTy::AssertionSuspend) {
    AssertionSuspended = true;
    // We just hit the very last event: indicate early exit.
    if (++NextEvent >= Events.size())
      return true;
  }

  // Continue with remaining notification logic.
  return false;
}

bool OmptSequencedAsserter::consumeRegularEvent(
    const omptest::OmptAssertEvent &AE) {
  // If we are actively asserting, increment the event counter.
  // Otherwise: If passively asserting, we will keep waiting for a match.
  auto &E = Events[NextEvent];
  if (E == AE && verifyEventGroups(E, AE)) {
    if (E.getEventExpectedState() == ObserveState::Always) {
      ++NumSuccessfulAsserts;
    } else if (E.getEventExpectedState() == ObserveState::Never) {
      Log->logEventMismatch(
          "[OmptSequencedAsserter] Encountered forbidden event", E, AE);
      State = AssertState::Fail;
    }

    // Return to active assertion
    if (AssertionSuspended)
      AssertionSuspended = false;

    // Match found, increment index and indicate early exit (success).
    ++NextEvent;
    return true;
  }

  // Continue with remaining notification logic.
  return false;
}

size_t OmptSequencedAsserter::getRemainingEventCount() {
  return std::count_if(Events.begin(), Events.end(),
                       [](const omptest::OmptAssertEvent &E) {
                         return E.getEventExpectedState() ==
                                ObserveState::Always;
                       }) -
         NumSuccessfulAsserts;
}

AssertState OmptSequencedAsserter::checkState() {
  // This is called after the testcase executed.
  // Once reached the number of successful notifications should be equal to the
  // number of expected events. However, there may still be excluded as well as
  // special asserter events remaining in the sequence.
  for (size_t i = NextEvent; i < Events.size(); ++i) {
    auto &E = Events[i];
    if (E.getEventExpectedState() == ObserveState::Always) {
      State = AssertState::Fail;
      Log->logEventMismatch("[OmptSequencedAsserter] Expected event was not "
                            "encountered (Remaining events: " +
                                std::to_string(getRemainingEventCount()) + ")",
                            E);
      break;
    }
  }

  return State;
}

void OmptEventAsserter::insert(OmptAssertEvent &&AE) {
  std::lock_guard<std::mutex> Lock(AssertMutex);
  Events.emplace_back(std::move(AE));
}

void OmptEventAsserter::notifyImpl(OmptAssertEvent &&AE) {
  std::lock_guard<std::mutex> Lock(AssertMutex);
  if (Events.empty() || !isActive() || isSuppressedEventType(AE.getEventType()))
    return;

  if (NumEvents == 0)
    NumEvents = Events.size();

  ++NumNotifications;

  if (AE.getEventType() == EventTy::AssertionSyncPoint) {
    auto NumRemainingEvents = getRemainingEventCount();
    // Upon encountering a SyncPoint, all events should have been processed
    if (NumRemainingEvents == 0)
      return;

    Log->logEventMismatch(
        "[OmptEventAsserter] Encountered SyncPoint while still awaiting " +
            std::to_string(NumRemainingEvents) + " events. Asserted " +
            std::to_string(NumSuccessfulAsserts) + " events successfully.",
        AE);
    State = AssertState::Fail;
    return;
  }

  for (size_t i = 0; i < Events.size(); ++i) {
    auto &E = Events[i];
    if (E == AE && verifyEventGroups(E, AE)) {
      if (E.getEventExpectedState() == ObserveState::Always) {
        Events.erase(Events.begin() + i);
        ++NumSuccessfulAsserts;
      } else if (E.getEventExpectedState() == ObserveState::Never) {
        Log->logEventMismatch("[OmptEventAsserter] Encountered forbidden event",
                              E, AE);
        State = AssertState::Fail;
      }
      return;
    }
  }

  if (OperationMode == AssertMode::Strict) {
    Log->logEventMismatch("[OmptEventAsserter] Too many events to check (" +
                              std::to_string(NumNotifications) +
                              "). Asserted " +
                              std::to_string(NumSuccessfulAsserts) +
                              " events successfully. (Remaining events: " +
                              std::to_string(getRemainingEventCount()) + ")",
                          AE);
    State = AssertState::Fail;
    return;
  }
}

size_t OmptEventAsserter::getRemainingEventCount() {
  return std::count_if(
      Events.begin(), Events.end(), [](const omptest::OmptAssertEvent &E) {
        return E.getEventExpectedState() == ObserveState::Always;
      });
}

AssertState OmptEventAsserter::checkState() {
  // This is called after the testcase executed.
  // Once reached no more expected events should be in the queue
  for (const auto &E : Events) {
    // Check if any of the remaining events were expected to be observed
    if (E.getEventExpectedState() == ObserveState::Always) {
      State = AssertState::Fail;
      Log->logEventMismatch("[OmptEventAsserter] Expected event was not "
                            "encountered (Remaining events: " +
                                std::to_string(getRemainingEventCount()) + ")",
                            E);
      break;
    }
  }

  return State;
}

void OmptEventReporter::notify(OmptAssertEvent &&AE) {
  if (!isActive() || isSuppressedEventType(AE.getEventType()))
    return;

  // Prepare notification, containing the newline to avoid stream interleaving.
  auto Notification{AE.toString()};
  Notification.push_back('\n');
  OutStream << Notification;
}

bool OmptEventGroupInterface::addActiveEventGroup(
    const std::string &GroupName, omptest::AssertEventGroup Group) {
  std::lock_guard<std::mutex> Lock(GroupMutex);
  auto EventGroup = ActiveEventGroups.find(GroupName);
  if (EventGroup != ActiveEventGroups.end() &&
      EventGroup->second.TargetRegion == Group.TargetRegion)
    return false;
  ActiveEventGroups.emplace(GroupName, Group);
  return true;
}

bool OmptEventGroupInterface::deprecateActiveEventGroup(
    const std::string &GroupName) {
  std::lock_guard<std::mutex> Lock(GroupMutex);
  auto EventGroup = ActiveEventGroups.find(GroupName);
  auto DeprecatedEventGroup = DeprecatedEventGroups.find(GroupName);
  if (EventGroup == ActiveEventGroups.end() &&
      DeprecatedEventGroup != DeprecatedEventGroups.end())
    return false;
  DeprecatedEventGroups.emplace(GroupName, EventGroup->second);
  ActiveEventGroups.erase(GroupName);
  return true;
}

bool OmptEventGroupInterface::checkActiveEventGroups(
    const std::string &GroupName, omptest::AssertEventGroup Group) {
  std::lock_guard<std::mutex> Lock(GroupMutex);
  auto EventGroup = ActiveEventGroups.find(GroupName);
  return (EventGroup != ActiveEventGroups.end() &&
          EventGroup->second.TargetRegion == Group.TargetRegion);
}

bool OmptEventGroupInterface::checkDeprecatedEventGroups(
    const std::string &GroupName, omptest::AssertEventGroup Group) {
  std::lock_guard<std::mutex> Lock(GroupMutex);
  auto EventGroup = DeprecatedEventGroups.find(GroupName);
  return (EventGroup != DeprecatedEventGroups.end() &&
          EventGroup->second.TargetRegion == Group.TargetRegion);
}
