#include "OmptAsserter.h"

#include <algorithm>

using namespace omptest;
using namespace internal;

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

void OmptAsserter::reportError(const OmptAssertEvent &OffendingEvent,
                               const std::string &Message) {
  std::cerr << "[Error] " << Message << "\nOffending Event: name='"
            << OffendingEvent.getEventName() << "' toString='"
            << OffendingEvent.toString() << "'" << std::endl;
}

void OmptAsserter::reportError(const OmptAssertEvent &AwaitedEvent,
                               const OmptAssertEvent &OffendingEvent,
                               const std::string &Message) {
  std::cerr << "[Assert Error]: Awaited event name='"
            << AwaitedEvent.getEventName() << "' toString='"
            << AwaitedEvent.toString() << "'\nGot: name='"
            << OffendingEvent.getEventName() << "' toString='"
            << OffendingEvent.toString() << "'\n"
            << Message << std::endl;
}

AssertState OmptAsserter::getState() { return State; }

bool OmptAsserter::verifyEventGroups(const OmptAssertEvent &ExpectedEvent,
                                     const OmptAssertEvent &ObservedEvent) {
  assert(ExpectedEvent.getEventType() == ObservedEvent.getEventType() &&
         "Type mismatch: Expected != Observed event type");
  assert(TC && "Missing parent TestCase");

  // Ignore all events within "default" group
  auto GroupName = ExpectedEvent.getEventGroup();
  if (GroupName == "default")
    return true;

  // Get a pointer to the observed internal event
  auto Event = ObservedEvent.getEvent();

  switch (Event->getType()) {
  case EventTy::Target:
    if (auto E = static_cast<const internal::Target *>(Event)) {
      if (E->Endpoint == ompt_scope_begin) {
        // Add new group since we entered a Target Region
        TC->addActiveEventGroup(GroupName, AssertEventGroup{E->TargetId});
      } else if (E->Endpoint == ompt_scope_end) {
        // Deprecate group since we return from a Target Region
        TC->deprecateActiveEventGroup(GroupName);
      }
      return true;
    }
    return false;
  case EventTy::TargetEmi:
    if (auto E = static_cast<const internal::TargetEmi *>(Event)) {
      if (E->Endpoint == ompt_scope_begin) {
        // Add new group since we entered a Target Region
        TC->addActiveEventGroup(GroupName,
                                AssertEventGroup{E->TargetData->value});
      } else if (E->Endpoint == ompt_scope_end) {
        // Deprecate group since we return from a Target Region
        TC->deprecateActiveEventGroup(GroupName);
      }
      return true;
    }
    return false;
  case EventTy::TargetDataOp:
    if (auto E = static_cast<const internal::TargetDataOp *>(Event))
      return TC->checkActiveEventGroups(GroupName,
                                        AssertEventGroup{E->TargetId});

    return false;
  case EventTy::TargetDataOpEmi:
    if (auto E = static_cast<const internal::TargetDataOpEmi *>(Event))
      return TC->checkActiveEventGroups(GroupName,
                                        AssertEventGroup{E->TargetData->value});

    return false;
  case EventTy::TargetSubmit:
    if (auto E = static_cast<const internal::TargetSubmit *>(Event))
      return TC->checkActiveEventGroups(GroupName,
                                        AssertEventGroup{E->TargetId});

    return false;
  case EventTy::TargetSubmitEmi:
    if (auto E = static_cast<const internal::TargetSubmitEmi *>(Event))
      return TC->checkActiveEventGroups(GroupName,
                                        AssertEventGroup{E->TargetData->value});

    return false;
  case EventTy::BufferRecord:
    // BufferRecords are delivered asynchronously: also check deprecated groups.
    if (auto E = static_cast<const internal::BufferRecord *>(Event))
      return (TC->checkActiveEventGroups(
                  GroupName, AssertEventGroup{E->Record.target_id}) ||
              TC->checkDeprecatedEventGroups(
                  GroupName, AssertEventGroup{E->Record.target_id}));
    return false;
  // Some event types do not need any handling
  case EventTy::ThreadBegin:
  case EventTy::ThreadEnd:
  case EventTy::ParallelBegin:
  case EventTy::ParallelEnd:
  case EventTy::TaskCreate:
  case EventTy::TaskSchedule:
  case EventTy::ImplicitTask:
  case EventTy::DeviceInitialize:
  case EventTy::DeviceFinalize:
  case EventTy::DeviceLoad:
  case EventTy::DeviceUnload:
  case EventTy::BufferRequest:
  case EventTy::BufferComplete:
  case EventTy::BufferRecordDeallocation:
    return true;
  // Observed events should be part of the OpenMP spec
  case EventTy::None:
  default:
    assert(false && "Encountered invalid event type");
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
  // Ignore notifications while inactive
  if (Events.empty() || !isActive() || isSuppressedEventType(AE.getEventType()))
    return;

  ++NumNotifications;

  if (AE.getEventType() == EventTy::AssertionSyncPoint) {
    auto NumRemainingEvents = getRemainingEventCount();
    // Upon encountering a SyncPoint, all events should have been processed
    if (NumRemainingEvents == 0)
      return;

    reportError(
        AE,
        "[OmptSequencedAsserter] Encountered SyncPoint while still awaiting " +
            std::to_string(NumRemainingEvents) + " events. Asserted " +
            std::to_string(NumAssertSuccesses) + "/" +
            std::to_string(Events.size()) + " events successfully.");
    State = AssertState::fail;
    return;
  }

  if (NextEvent >= Events.size()) {
    // If we are not expecting any more events and passively asserting: return
    if (AssertionSuspended)
      return;

    reportError(AE, "[OmptSequencedAsserter] Too many events to check (" +
                        std::to_string(NumNotifications) + "). Asserted " +
                        std::to_string(NumAssertSuccesses) + "/" +
                        std::to_string(Events.size()) +
                        " events successfully.");
    State = AssertState::fail;
    return;
  }

  // On AssertionSuspend -- enter 'passive' assertion.
  // Since we may encounter multiple, successive AssertionSuspend events, loop
  // until we hit the next non-AssertionSuspend event.
  while (Events[NextEvent].getEventType() == EventTy::AssertionSuspend) {
    AssertionSuspended = true;
    // We just hit the very last event: return
    if (++NextEvent >= Events.size())
      return;
  }

  // If we are actively asserting, increment the event counter.
  // Otherwise: If passively asserting, we will keep waiting for a match.
  auto &E = Events[NextEvent];
  if (E == AE && verifyEventGroups(E, AE)) {
    if (E.getEventExpectedState() == ObserveState::always) {
      ++NumAssertSuccesses;
    } else if (E.getEventExpectedState() == ObserveState::never) {
      reportError(E, AE, "[OmptSequencedAsserter] Encountered forbidden event");
      State = AssertState::fail;
    }

    // Return to active assertion
    if (AssertionSuspended)
      AssertionSuspended = false;

    // Match found, increment index
    ++NextEvent;
    return;
  }

  if (AssertionSuspended || OperationMode == AssertMode::relaxed)
    return;

  reportError(E, AE, "[OmptSequencedAsserter] The events are not equal");
  State = AssertState::fail;
}

size_t OmptSequencedAsserter::getRemainingEventCount() {
  return std::count_if(Events.begin(), Events.end(),
                       [](const omptest::OmptAssertEvent &E) {
                         return E.getEventExpectedState() ==
                                ObserveState::always;
                       }) -
         NumAssertSuccesses;
}

AssertState OmptSequencedAsserter::getState() {
  // This is called after the testcase executed.
  // Once reached the number of successful notifications should be equal to the
  // number of expected events. However, there may still be excluded as well as
  // special asserter events remaining in the sequence.
  for (size_t i = NextEvent; i < Events.size(); ++i) {
    auto &E = Events[i];
    if (E.getEventExpectedState() == ObserveState::always) {
      State = AssertState::fail;
      reportError(E, "[OmptSequencedAsserter] Expected event was not "
                     "encountered (Remaining events: " +
                         std::to_string(getRemainingEventCount()) + ")");
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

    reportError(
        AE, "[OmptEventAsserter] Encountered SyncPoint while still awaiting " +
                std::to_string(NumRemainingEvents) + " events. Asserted " +
                std::to_string(NumAssertSuccesses) + " events successfully.");
    State = AssertState::fail;
    return;
  }

  for (size_t i = 0; i < Events.size(); ++i) {
    auto &E = Events[i];
    if (E == AE && verifyEventGroups(E, AE)) {
      if (E.getEventExpectedState() == ObserveState::always) {
        Events.erase(Events.begin() + i);
        ++NumAssertSuccesses;
      } else if (E.getEventExpectedState() == ObserveState::never) {
        reportError(E, AE, "[OmptEventAsserter] Encountered forbidden event");
        State = AssertState::fail;
      }
      return;
    }
  }

  if (OperationMode == AssertMode::strict) {
    reportError(AE, "[OmptEventAsserter] Too many events to check (" +
                        std::to_string(NumNotifications) + "). Asserted " +
                        std::to_string(NumAssertSuccesses) +
                        " events successfully. (Remaining events: " +
                        std::to_string(getRemainingEventCount()) + ")");
    State = AssertState::fail;
    return;
  }
}

size_t OmptEventAsserter::getRemainingEventCount() {
  // size_t EventCount = std::count_if(Events.begin(), Events.end(), [](const
  // omptest::OmptAssertEvent &E) { return E.getEventExpectedState() ==
  // ObserveState::always; });
  return std::count_if(
      Events.begin(), Events.end(), [](const omptest::OmptAssertEvent &E) {
        return E.getEventExpectedState() == ObserveState::always;
      });
}

AssertState OmptEventAsserter::getState() {
  // This is called after the testcase executed.
  // Once reached no more expected events should be in the queue
  for (const auto &E : Events) {
    // Check if any of the remaining events were expected to be observed
    if (E.getEventExpectedState() == ObserveState::always) {
      State = AssertState::fail;
      reportError(E, "[OmptEventAsserter] Expected event was not "
                     "encountered (Remaining events: " +
                         std::to_string(getRemainingEventCount()) + ")");
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
