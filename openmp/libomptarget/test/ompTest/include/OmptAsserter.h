#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTER_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTER_H

#include "OmptAssertEvent.h"

#include <cassert>
#include <iostream>
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
  void setActive(bool Enabled) { Active = Enabled; }

  /// Check if this asserter is considered 'active'.
  bool isActive() { return Active; }

  /// Check if the given event type is in the set of suppressed event types.
  bool isSuppressedEventType(omptest::internal::EventTy EvTy) {
    return SuppressedEvents.find(EvTy) != SuppressedEvents.end();
  }

  /// Remove the given event type to the set of suppressed events.
  void permitEvent(omptest::internal::EventTy EvTy) {
    SuppressedEvents.erase(EvTy);
  }

  /// Add the given event type to the set of suppressed events.
  void suppressEvent(omptest::internal::EventTy EvTy) {
    SuppressedEvents.insert(EvTy);
  }

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
  virtual void insert(omptest::OmptAssertEvent &&AE) {
    assert(false && "Base class 'insert' has undefined semantics.");
  }

  // Called from the CallbackHandler with a corresponding AssertEvent to which
  // callback was handled.
  void notify(omptest::OmptAssertEvent &&AE) override {
    // Ignore notifications while inactive
    if (!isActive() || isSuppressedEventType(AE.getEventType()))
      return;

    this->notifyImpl(std::move(AE));
  }

  /// Implemented in subclasses to implement what should actually be done with
  /// the notification.
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) = 0;

  /// Report an error for a single event
  void reportError(const omptest::OmptAssertEvent &OffendingEvent,
                   const std::string &Message) {
    std::cerr << "[Error] " << Message
              << "\nOffending Event: " << OffendingEvent.getEventName()
              << std::endl;
  }

  void reportError(const omptest::OmptAssertEvent &AwaitedEvent,
                   const omptest::OmptAssertEvent &OffendingEvent,
                   const std::string &Message) {
    std::cerr << "[Assert Error]: Awaited event name='"
              << AwaitedEvent.getEventName() << "' toString='"
              << AwaitedEvent.toString() << ")\nGot: name='"
              << OffendingEvent.getEventName() << "' toString='"
              << OffendingEvent.toString() << "'\n"
              << Message << std::endl;
  }

  virtual omptest::AssertState getState() { return State; }

protected:
  omptest::AssertState State{omptest::AssertState::pass};
};

/// Class that can assert in a sequenced fashion, i.e., events hace to occur in
/// the order they were registered
struct OmptSequencedAsserter : public OmptAsserter {
  OmptSequencedAsserter() : NextEvent(0), Events() {}

  /// Add the event to the in-sequence set of events that the asserter should
  /// check for.
  void insert(omptest::OmptAssertEvent &&AE) override {
    Events.emplace_back(std::move(AE));
  }

  /// Implements the asserter's actual logic
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) override {
    // Ignore notifications while inactive
    if (!isActive() || isSuppressedEventType(AE.getEventType()))
      return;

    ++NumNotifications;

    if (NextEvent >= Events.size()) {
      reportError(AE, "[OmptSequencedAsserter] Too many events to check (" +
                          std::to_string(NumNotifications) + "). Asserted " +
                          std::to_string(NumAssertSuccesses) + "/" +
                          std::to_string(Events.size()) +
                          " events successfully.");
      State = omptest::AssertState::fail;
      return;
    }

    auto &E = Events[NextEvent++];
    if (E == AE) {
      ++NumAssertSuccesses;
      return;
    }

    reportError(E, AE, "[OmptSequencedAsserter] The events are not equal");
    State = omptest::AssertState::fail;
  }

  omptest::AssertState getState() override {
    // This is called after the testcase executed.
    // Once, reached, no more events should be in the queue
    if (NextEvent < Events.size())
      State = omptest::AssertState::fail;

    return State;
  }

  int NumAssertSuccesses{0};
  int NumNotifications{0};
  size_t NextEvent{0};
  std::vector<omptest::OmptAssertEvent> Events;
};

/// Class that asserts with set semantics, i.e., unordered
struct OmptEventAsserter : public OmptAsserter {

  void insert(omptest::OmptAssertEvent &&AE) override {
    Events.emplace_back(std::move(AE));
  }

  /// Implements the asserter's logic
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) override {
    if (!isActive())
      return;

    for (size_t I = 0; I < Events.size(); ++I) {
      if (Events[I] == AE) {
        Events.erase(Events.begin() + I);
        break;
      }
    }
  }

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
  void notify(omptest::OmptAssertEvent &&AE) override {
    if (!isActive() || isSuppressedEventType(AE.getEventType()))
      return;

    OutStream << AE.toString() << std::endl;
  }

private:
  std::ostream &OutStream;
};

#endif
