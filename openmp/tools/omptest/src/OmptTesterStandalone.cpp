//===- OmptTesterStandalone.cpp - Standalone unit testing impl. -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file represents the 'standalone' ompTest unit testing core
/// implementation, defining the general test suite and test case execution.
///
//===----------------------------------------------------------------------===//

#include "OmptTesterStandalone.h"
#include "OmptCallbackHandler.h"

#include <algorithm>
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
  omptest::AssertState SequenceResultState = SequenceAsserter->checkState();
  omptest::AssertState SetResultState = SetAsserter->checkState();
  bool AnyFail = SequenceResultState == omptest::AssertState::Fail ||
                 SetResultState == omptest::AssertState::Fail;
  bool AllPass = SequenceResultState == omptest::AssertState::Pass &&
                 SetResultState == omptest::AssertState::Pass;
  if (ExpectedState == omptest::AssertState::Pass && AnyFail)
    E.Fail = true;
  else if (ExpectedState == omptest::AssertState::Fail && AllPass)
    E.Fail = true;
  if (AnyFail)
    ResultState = omptest::AssertState::Fail;
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

void TestRegistrar::addCaseToSuite(TestCase *TC, const std::string &TSName) {
  // Search the test suites for a matching name
  auto It = std::find_if(Tests.begin(), Tests.end(),
                         [&](const auto &P) { return P.first == TSName; });

  if (It != Tests.end()) {
    // Test suite exists: add the test case
    It->second.TestCases.emplace_back(TC);
  } else {
    // Test suite does not exist: construct it and add the test case
    TestSuite TS(TSName);
    TS.TestCases.emplace_back(TC);
    // Move and emplace the suite
    Tests.emplace_back(TSName, std::move(TS));
  }
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
        if (TC->ResultState == omptest::AssertState::Pass)
          Result = "PASS";
        else if (TC->ResultState == omptest::AssertState::Fail)
          Result = "XFAIL";
      } else {
        if (TC->ResultState == omptest::AssertState::Fail)
          Result = "FAIL";
        else if (TC->ResultState == omptest::AssertState::Pass)
          Result = "UPASS";
      }
      std::cout << "\n      " << std::setw(5) << Result << " : " << TC->Name;
    }
    std::cout << std::endl;
  }
}
