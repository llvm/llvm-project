#include "OmptAsserter.h"

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

  // Ignore all events within "default" group
  auto GroupName = ExpectedEvent.getEventGroup();
  if (GroupName == "default")
    return true;

  // Get the expected event group and a pointer to the observed internal event
  auto EventGroup = EventGroups.find(GroupName);
  auto Event = ObservedEvent.getEvent();

  switch (Event->getType()) {
  case EventTy::Target:
    if (auto E = static_cast<const internal::Target *>(Event)) {
      if (E->Endpoint == ompt_scope_begin) {
        assert(EventGroup == EventGroups.end() && "Group already exists");
        // Add new group since we entered a Target Region
        EventGroups.emplace(GroupName, E->TargetId);
      } else if (E->Endpoint == ompt_scope_end) {
        assert(EventGroup != EventGroups.end() && "Group does not exist");
        // Erase group since we return from a Target Region
        EventGroups.erase(GroupName);
      }
      return true;
    }
    return false;
  case EventTy::TargetEmi:
    if (auto E = static_cast<const internal::TargetEmi *>(Event)) {
      if (E->Endpoint == ompt_scope_begin) {
        assert(EventGroup == EventGroups.end() && "Group already exists");
        // Add new group since we entered a Target Region
        EventGroups.emplace(GroupName, E->TargetData->value);
      } else if (E->Endpoint == ompt_scope_end) {
        assert(EventGroup != EventGroups.end() && "Group does not exist");
        // Erase group since we return from a Target Region
        EventGroups.erase(GroupName);
      }
      return true;
    }
    return false;
  case EventTy::TargetDataOp:
    if (auto E = static_cast<const internal::TargetDataOp *>(Event)) {
      if (EventGroup != EventGroups.end())
        return E->TargetId == EventGroup->second.TargetRegion;
    }
    return false;
  case EventTy::TargetDataOpEmi:
    if (auto E = static_cast<const internal::TargetDataOpEmi *>(Event)) {
      if (EventGroup != EventGroups.end())
        return E->TargetData->value == EventGroup->second.TargetRegion;
    }
    return false;
  case EventTy::TargetSubmit:
    if (auto E = static_cast<const internal::TargetSubmit *>(Event)) {
      if (EventGroup != EventGroups.end())
        return E->TargetId == EventGroup->second.TargetRegion;
    }
    return false;
  case EventTy::TargetSubmitEmi:
    if (auto E = static_cast<const internal::TargetSubmitEmi *>(Event)) {
      if (EventGroup != EventGroups.end())
        return E->TargetData->value == EventGroup->second.TargetRegion;
    }
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
    return true;
  // Some event types are not implemented yet
  case EventTy::BufferRequest:
  case EventTy::BufferComplete:
    assert(false && "Encountered unimplemented event type");
  // Observed events should be part of the OpenMP spec
  case EventTy::None:
  case EventTy::BufferRecord:
  default:
    assert(false && "Encountered invalid event type");
  }

  return true;
}

void OmptSequencedAsserter::insert(OmptAssertEvent &&AE) {
  Events.emplace_back(std::move(AE));
}

void OmptSequencedAsserter::notifyImpl(OmptAssertEvent &&AE) {
  // Ignore notifications while inactive
  if (Events.empty() || !isActive() || isSuppressedEventType(AE.getEventType()))
    return;

  ++NumNotifications;

  if (NextEvent >= Events.size()) {
    // If we are not expecting any more events and passively asserting: return
    if (!ActiveMode)
      return;

    reportError(AE, "[OmptSequencedAsserter] Too many events to check (" +
                        std::to_string(NumNotifications) + "). Asserted " +
                        std::to_string(NumAssertSuccesses) + "/" +
                        std::to_string(Events.size()) +
                        " events successfully.");
    State = AssertState::fail;
    return;
  }

  // If the event is meant for the Asserter itself -- enter 'passive' assertion.
  // Since we may encounter multiple successive Asserter events, loop until we
  // hit the next non-Asserter event.
  while (Events[NextEvent].getEventType() == EventTy::Asserter) {
    ActiveMode = false;
    ++NumAssertSuccesses;
    // We just hit the very last event: return
    if (++NextEvent >= Events.size())
      return;
  }

  // If we are actively asserting, increment the event counter.
  // Otherwise: If passively asserting, we will keep waiting for a match.
  auto &E = ActiveMode ? Events[NextEvent++] : Events[NextEvent];
  if (E == AE && verifyEventGroups(E, AE)) {
    if (E.getEventExpectedState() == ObserveState::always) {
      ++NumAssertSuccesses;
    } else if (E.getEventExpectedState() == ObserveState::never) {
      reportError(E, AE, "[OmptSequencedAsserter] Encountered forbidden event");
      State = AssertState::fail;
    }
    // Return to active assertion
    if (!ActiveMode) {
      ActiveMode = true;
      ++NextEvent;
    }
    return;
  }

  if (!ActiveMode)
    return;

  reportError(E, AE, "[OmptSequencedAsserter] The events are not equal");
  State = AssertState::fail;
}

AssertState OmptSequencedAsserter::getState() {
  // This is called after the testcase executed.
  // Once reached the number of successful notifications should be equal to the
  // number of expected events. However, there may still be excluded as well as
  // special asserter events remaining in the sequence.
  for (size_t i = NextEvent; i < Events.size(); ++i) {
    auto &E = Events[i];
    if (E.getEventExpectedState() == ObserveState::always &&
        E.getEventType() != EventTy::Asserter) {
      State = AssertState::fail;
      break;
    }
  }

  return State;
}

void OmptEventAsserter::insert(OmptAssertEvent &&AE) {
  Events.emplace_back(std::move(AE));
}

void OmptEventAsserter::notifyImpl(OmptAssertEvent &&AE) {
  if (Events.empty() || !isActive() || isSuppressedEventType(AE.getEventType()))
    return;

  if (NumEvents == 0)
    NumEvents = Events.size();

  ++NumNotifications;

  for (size_t I = 0; I < Events.size(); ++I) {
    auto &E = Events[I];
    if (E == AE) {
      if (E.getEventExpectedState() == ObserveState::always) {
        Events.erase(Events.begin() + I);
        ++NumAssertSuccesses;
      } else if (E.getEventExpectedState() == ObserveState::never) {
        reportError(E, AE, "[OmptEventAsserter] Encountered forbidden event");
        State = AssertState::fail;
      }
      return;
    }
  }
}

AssertState OmptEventAsserter::getState() {
  // This is called after the testcase executed.
  // Once reached no more expected events should be in the queue
  if (!Events.empty())
    for (const auto &E : Events) {
      // Check if any of the remaining events were expected to be observed
      if (E.getEventExpectedState() == ObserveState::always) {
        State = AssertState::fail;
        break;
      }
    }

  return State;
}

void OmptEventReporter::notify(OmptAssertEvent &&AE) {
  if (!isActive() || isSuppressedEventType(AE.getEventType()))
    return;

  OutStream << AE.toString() << std::endl;
}
