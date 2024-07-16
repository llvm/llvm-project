#include "OmptTesterStandalone.h"
#include "OmptCallbackHandler.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace omptest;

Error TestCase::exec() {
  Error E;
  E.Fail = false;

  if (IsDisabled)
    return E;

  OmptCallbackHandler::get().subscribe(SequenceAsserter.get());
  OmptCallbackHandler::get().subscribe(SetAsserter.get());
  OmptCallbackHandler::get().subscribe(EventReporter.get());

  execImpl();

  // Actively flush potential in-flight trace records
  flush_traced_devices();

  // We remove subscribers to not be notified of events after our test case
  // finished.
  OmptCallbackHandler::get().clearSubscribers();
  omptest::AssertState SequenceResultState = SequenceAsserter->getState();
  omptest::AssertState SetResultState = SetAsserter->getState();
  bool AnyFail = SequenceResultState == omptest::AssertState::fail ||
                 SetResultState == omptest::AssertState::fail;
  bool AllPass = SequenceResultState == omptest::AssertState::pass &&
                 SetResultState == omptest::AssertState::pass;
  if (ExpectedState == omptest::AssertState::pass && AnyFail)
    E.Fail = true;
  else if (ExpectedState == omptest::AssertState::fail && AllPass)
    E.Fail = true;
  if (AnyFail)
    ResultState = omptest::AssertState::fail;
  return E;
}

TestSuite::TestSuite(TestSuite &&O) {
  Name = O.Name;
  TestCases.swap(O.TestCases);
}

void TestSuite::setup() {}

void TestSuite::teardown() {}

TestSuite::TestCaseVec::iterator TestSuite::begin() {
  return TestCases.begin();
}

TestSuite::TestCaseVec::iterator TestSuite::end() { return TestCases.end(); }

TestRegistrar &TestRegistrar::get() {
  static TestRegistrar TR;
  return TR;
}

std::vector<TestSuite> TestRegistrar::getTestSuites() {
  std::vector<TestSuite> TSs;
  for (auto &[k, v] : Tests)
    TSs.emplace_back(std::move(v));
  return TSs;
}

void TestRegistrar::addCaseToSuite(TestCase *TC, std::string TSName) {
  auto &TS = Tests[TSName];
  if (TS.Name.empty())
    TS.Name = TSName;
  TS.TestCases.emplace_back(TC);
}

Registerer::Registerer(TestCase *TC, const std::string SuiteName) {
  std::cout << "Adding " << TC->Name << " to " << SuiteName << std::endl;
  TestRegistrar::get().addCaseToSuite(TC, SuiteName);
}

int Runner::run() {
  int ErrorCount = 0;
  for (auto &TS : TestSuites) {
    std::cout << "\n======\nExecuting for " << TS.Name << std::endl;
    TS.setup();
    for (auto &TC : TS) {
      std::cout << "\nExecuting " << TC->Name << std::endl;
      if (Error Err = TC->exec()) {
        reportError(Err);
        abortOrKeepGoing();
        ++ErrorCount;
      }
    }
    TS.teardown();
  }
  printSummary();
  return ErrorCount;
}

void Runner::reportError(const Error &Err) {}

void Runner::abortOrKeepGoing() {}

void Runner::printSummary() {
  std::cout << "\n====== SUMMARY\n";
  for (auto &TS : TestSuites) {
    std::cout << "  - " << TS.Name;
    for (auto &TC : TS) {
      std::string Result;
      if (TC->IsDisabled) {
        Result = "-#-#-";
      } else if (TC->ResultState == TC->ExpectedState) {
        if (TC->ResultState == omptest::AssertState::pass)
          Result = "PASS";
        else if (TC->ResultState == omptest::AssertState::fail)
          Result = "XFAIL";
      } else {
        if (TC->ResultState == omptest::AssertState::fail)
          Result = "FAIL";
        else if (TC->ResultState == omptest::AssertState::pass)
          Result = "UPASS";
      }
      std::cout << "\n      " << std::setw(5) << Result << " : " << TC->Name;
    }
    std::cout << std::endl;
  }
}
