//===- ErrorMatchersTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the Error / Expected<T> matchers in ErrorMatchers.h.
//
// Each matcher is exercised on a value it should match and on one it should
// not. The latter run inside gtest's own failure interception (gtest-spi.h) and
// assert on the reported text, since a matcher's diagnostic is most of what it
// is for.
//
//===----------------------------------------------------------------------===//

#include "ErrorMatchers.h"

#include "gtest/gtest-spi.h"

using namespace orc_rt;
using namespace orc_rt::test;

using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::Property;

namespace {

class CustomError : public ErrorExtends<CustomError, ErrorInfoBase> {
public:
  static constexpr const char *RTTIName = "::CustomError";

  CustomError(int Info) : Info(Info) {}

  std::string toString() const noexcept override {
    return "CustomError (" + std::to_string(Info) + ")";
  }

  int getInfo() const { return Info; }

protected:
  int Info;
};

class OtherError : public ErrorExtends<OtherError, ErrorInfoBase> {
public:
  static constexpr const char *RTTIName = "::OtherError";

  std::string toString() const noexcept override { return "OtherError"; }
};

TEST(ErrorMatchersTest, SucceededMatchesSuccess) {
  EXPECT_THAT_ERROR(Error::success(), Succeeded());
}

TEST(ErrorMatchersTest, SucceededRejectsFailure) {
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THAT_ERROR(make_error<CustomError>(7), Succeeded()), "7");
}

TEST(ErrorMatchersTest, FailedMatchesFailure) {
  EXPECT_THAT_ERROR(make_error<CustomError>(7), Failed());
}

TEST(ErrorMatchersTest, FailedRejectsSuccess) {
  EXPECT_NONFATAL_FAILURE(EXPECT_THAT_ERROR(Error::success(), Failed()),
                          "Actual");
}

TEST(ErrorMatchersTest, LValueErrorIsHandedOverExplicitly) {
  Error E = make_error<CustomError>(7);
  EXPECT_THAT_ERROR(std::move(E), Failed());
}

TEST(ErrorMatchersTest, FailedOfTypeMatchesThatType) {
  EXPECT_THAT_ERROR(make_error<CustomError>(7), Failed<CustomError>());
}

TEST(ErrorMatchersTest, FailedOfTypeNamesTheTypeItFound) {
  // The point of the type matcher: on a mismatch it says what did turn up,
  // rather than only that the expectation was not met.
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THAT_ERROR(make_error<OtherError>(), Failed<CustomError>()),
      "failed with ::OtherError: OtherError");
}

TEST(ErrorMatchersTest, FailedOfTypeRejectsSuccess) {
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THAT_ERROR(Error::success(), Failed<CustomError>()), "succeeded");
}

TEST(ErrorMatchersTest, FailedOfTypeWithInnerMatcher) {
  EXPECT_THAT_ERROR(
      make_error<CustomError>(7),
      Failed<CustomError>(Property("getInfo", &CustomError::getInfo, 7)));
}

TEST(ErrorMatchersTest, FailedOfTypeRejectsFailingInnerMatcher) {
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THAT_ERROR(
          make_error<CustomError>(7),
          Failed<CustomError>(Property("getInfo", &CustomError::getInfo, 8))),
      "getInfo");
}

TEST(ErrorMatchersTest, FailedWithMessageMatchesExactMessage) {
  EXPECT_THAT_ERROR(make_error<CustomError>(7),
                    FailedWithMessage("CustomError (7)"));
}

TEST(ErrorMatchersTest, FailedWithMessageTakesAMatcher) {
  // The reason to prefer a matcher: a message that embeds errno text, a path,
  // or an address cannot be compared for equality across platforms.
  EXPECT_THAT_ERROR(make_error<CustomError>(7),
                    FailedWithMessage(HasSubstr("(7)")));
}

TEST(ErrorMatchersTest, FailedWithMessageRejectsOtherMessage) {
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THAT_ERROR(make_error<CustomError>(7),
                        FailedWithMessage("CustomError (8)")),
      "CustomError (7)");
}

TEST(ErrorMatchersTest, FailedWithMessageRejectsSuccess) {
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THAT_ERROR(Error::success(), FailedWithMessage("anything")),
      "succeeded");
}

TEST(ErrorMatchersTest, AssertThatErrorIsFatal) {
  EXPECT_FATAL_FAILURE(ASSERT_THAT_ERROR(Error::success(), Failed()), "Actual");
}

TEST(ExpectedMatchersTest, SucceededMatchesValue) {
  EXPECT_THAT_EXPECTED(Expected<int>(42), Succeeded());
}

TEST(ExpectedMatchersTest, FailedMatchesFailure) {
  Expected<int> V = make_error<CustomError>(7);
  EXPECT_THAT_EXPECTED(V, Failed());
}

TEST(ExpectedMatchersTest, HasValueMatchesValue) {
  EXPECT_THAT_EXPECTED(Expected<int>(42), HasValue(42));
}

TEST(ExpectedMatchersTest, HasValueTakesAMatcher) {
  EXPECT_THAT_EXPECTED(Expected<int>(42), HasValue(Gt(40)));
}

TEST(ExpectedMatchersTest, HasValueRejectsOtherValue) {
  EXPECT_NONFATAL_FAILURE(EXPECT_THAT_EXPECTED(Expected<int>(42), HasValue(43)),
                          "42");
}

TEST(ExpectedMatchersTest, HasValueRejectsFailureAndNamesTheError) {
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THAT_EXPECTED(Expected<int>(make_error<CustomError>(7)),
                           HasValue(42)),
      "failed with ::CustomError: CustomError (7)");
}

TEST(ExpectedMatchersTest, FailedRejectsValue) {
  EXPECT_NONFATAL_FAILURE(EXPECT_THAT_EXPECTED(Expected<int>(42), Failed()),
                          "Actual");
}

TEST(ExpectedMatchersTest, ValueSurvivesASucceededMatch) {
  Expected<int> V(42);
  ASSERT_THAT_EXPECTED(V, Succeeded());
  EXPECT_EQ(*V, 42);
}

TEST(ExpectedMatchersTest, FailureLeavesExpectedDestructible) {
  // The macro takes the Expected's error, which leaves it in a state where
  // re-checking it would arm the abort in its destructor. Nothing in the
  // matchers does that, so this scope exits cleanly rather than aborting.
  Expected<int> V = make_error<CustomError>(7);
  EXPECT_THAT_EXPECTED(V, Failed<CustomError>());
}

} // namespace
