//===- OmptTesterGoogleTest.h - GoogleTest header variant -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file represents the GoogleTest-based header variant, defining the
/// actual test classes and their behavior.
///
//===----------------------------------------------------------------------===//

#ifndef OPENMP_TOOLS_OMPTEST_INCLUDE_OMPTTESTERGOOGLETEST_H
#define OPENMP_TOOLS_OMPTEST_INCLUDE_OMPTTESTERGOOGLETEST_H

#include "AssertMacros.h"
#include "OmptAliases.h"
#include "OmptAssertEvent.h"
#include "OmptAsserter.h"
#include "OmptCallbackHandler.h"
#include "OmptTesterGlobals.h"

// This will allow us to override the "TEST" macro of gtest
#define GTEST_DONT_DEFINE_TEST 1
#include "gtest/gtest.h"

namespace testing {
class GTEST_API_ OmptTestCase : public testing::Test,
                                public omptest::OmptEventGroupInterface {
public:
  std::unique_ptr<omptest::OmptSequencedAsserter> SequenceAsserter =
      std::make_unique<omptest::OmptSequencedAsserter>();
  std::unique_ptr<omptest::OmptEventAsserter> SetAsserter =
      std::make_unique<omptest::OmptEventAsserter>();
  std::unique_ptr<omptest::OmptEventReporter> EventReporter =
      std::make_unique<omptest::OmptEventReporter>();

protected:
  void SetUp() override {
    omptest::OmptCallbackHandler::get().subscribe(SequenceAsserter.get());
    omptest::OmptCallbackHandler::get().subscribe(SetAsserter.get());
    omptest::OmptCallbackHandler::get().subscribe(EventReporter.get());
  }

  void TearDown() override {
    // Actively flush potential in-flight trace records
    flush_traced_devices();

    // Remove subscribers to not be notified of events after test execution.
    omptest::OmptCallbackHandler::get().clearSubscribers();

    // This common testcase must not encounter any failures.
    if (SequenceAsserter->checkState() == omptest::AssertState::Fail ||
        SetAsserter->checkState() == omptest::AssertState::Fail)
      ADD_FAILURE();
  }
};

class GTEST_API_ OmptTestCaseXFail : public testing::OmptTestCase {
protected:
  void TearDown() override {
    // Actively flush potential in-flight trace records
    flush_traced_devices();

    // Remove subscribers to not be notified of events after test execution.
    omptest::OmptCallbackHandler::get().clearSubscribers();

    // This eXpectedly failing testcase has to encounter at least one failure.
    if (SequenceAsserter->checkState() == omptest::AssertState::Pass &&
        SetAsserter->checkState() == omptest::AssertState::Pass)
      ADD_FAILURE();
  }
};
} // namespace testing

#define TEST(test_suite_name, test_name)                                       \
  GTEST_TEST_(test_suite_name, test_name, ::testing::OmptTestCase,             \
              ::testing::internal::GetTypeId<::testing::OmptTestCase>())

#define TEST_XFAIL(test_suite_name, test_name)                                 \
  GTEST_TEST_(test_suite_name, test_name, ::testing::OmptTestCaseXFail,        \
              ::testing::internal::GetTypeId<::testing::OmptTestCaseXFail>())

#endif // include guard
