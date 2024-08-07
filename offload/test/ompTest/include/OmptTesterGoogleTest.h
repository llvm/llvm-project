#ifndef OFFLOAD_TEST_OMPTEST_INCLUDE_OMPTTESTERGOOGLETEST_H
#define OFFLOAD_TEST_OMPTEST_INCLUDE_OMPTTESTERGOOGLETEST_H

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
    if (SequenceAsserter->getState() == omptest::AssertState::fail ||
        SetAsserter->getState() == omptest::AssertState::fail)
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
    if (SequenceAsserter->getState() == omptest::AssertState::pass &&
        SetAsserter->getState() == omptest::AssertState::pass)
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
