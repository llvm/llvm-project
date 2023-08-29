#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTER_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTER_H

#include "OmptAssertEvent.h"

#include <cassert>
#include <set>
#include <vector>

#include <iostream>

/// Base class for asserting on OMPT events
struct OmptAsserter {
  virtual void insert(omptest::OmptAssertEvent &&AE) {
    assert(false && "Base class 'insert' has undefined semantics.");
  }

  // Called from the CallbackHandler with a corresponding AssertEvent to which
  // callback was handled.
  void notify(omptest::OmptAssertEvent &&AE) {
    this->notifyImpl(std::move(AE));
  }

  /// Implemented in subclasses to implement what should actually be done with
  /// the notification
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) = 0;
};

/// Class that can assert in a sequenced fashion, i.e., events hace to occur in
/// the order they were registered
struct OmptSequencedAsserter : public OmptAsserter {
  void insert(omptest::OmptAssertEvent &&AE) override {
    Events.emplace_back(std::move(AE));
  }

  /// Implements the asserter's logic
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) override {
    std::cout << "OmptSequencedAsserter::notifyImpl called w/ " << Events.size()
              << " Events to check.\nNext Check item: " << NextEvent
              << std::endl;
    if (NextEvent >= Events.size()) {
      std::cerr << "[Error] Too many events to check. Only asserted single "
                   "event.\nOffending event: "
                << AE.getEventName() << std::endl;
      exit(-1); // TODO: Make this reasonable assert error
    }

    auto &E = Events[NextEvent++];
    if (E == AE)
      return;

    // TODO: Implement assert error
    std::cout << "[ASSERT ERROR] The events are not equal.\n"
              << AE.getEventName() << " == " << E.getEventName() << std::endl;
    exit(-2);
  }

  int NextEvent{0};
  std::vector<omptest::OmptAssertEvent> Events;
};

/// Class that asserts with set semantics, i.e., unordered
struct OmptEventAsserter : public OmptAsserter {
  void insert(omptest::OmptAssertEvent &&AE) override {
    // TODO
  }
  /// Implements the asserter's logic
  virtual void notifyImpl(omptest::OmptAssertEvent &&AE) override {
    // TODO
  }
};

#endif
