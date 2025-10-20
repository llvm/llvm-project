//===- SessionTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's Session.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Session.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <optional>

using namespace orc_rt;
using ::testing::Eq;
using ::testing::Optional;

class MockResourceManager : public ResourceManager {
public:
  enum class Op { Detach, Shutdown };

  static Error alwaysSucceed(Op) { return Error::success(); }

  MockResourceManager(std::optional<size_t> &DetachOpIdx,
                      std::optional<size_t> &ShutdownOpIdx, size_t &OpIdx,
                      move_only_function<Error(Op)> GenResult = alwaysSucceed)
      : DetachOpIdx(DetachOpIdx), ShutdownOpIdx(ShutdownOpIdx), OpIdx(OpIdx),
        GenResult(std::move(GenResult)) {}

  void detach(OnCompleteFn OnComplete) override {
    DetachOpIdx = OpIdx++;
    OnComplete(GenResult(Op::Detach));
  }

  void shutdown(OnCompleteFn OnComplete) override {
    ShutdownOpIdx = OpIdx++;
    OnComplete(GenResult(Op::Shutdown));
  }

private:
  std::optional<size_t> &DetachOpIdx;
  std::optional<size_t> &ShutdownOpIdx;
  size_t &OpIdx;
  move_only_function<Error(Op)> GenResult;
};

// Non-overloaded version of cantFail: allows easy construction of
// move_only_functions<void(Error)>s.
static void noErrors(Error Err) { cantFail(std::move(Err)); }

TEST(SessionTest, TrivialConstructionAndDestruction) { Session S(noErrors); }

TEST(SessionTest, ReportError) {
  Error E = Error::success();
  cantFail(std::move(E)); // Force error into checked state.

  Session S([&](Error Err) { E = std::move(Err); });
  S.reportError(make_error<StringError>("foo"));

  if (E)
    EXPECT_EQ(toString(std::move(E)), "foo");
  else
    ADD_FAILURE() << "Missing error value";
}

TEST(SessionTest, SingleResourceManager) {
  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;

  {
    Session S(noErrors);
    S.addResourceManager(std::make_unique<MockResourceManager>(
        DetachOpIdx, ShutdownOpIdx, OpIdx));
  }

  EXPECT_EQ(OpIdx, 1U);
  EXPECT_EQ(DetachOpIdx, std::nullopt);
  EXPECT_THAT(ShutdownOpIdx, Optional(Eq(0)));
}

TEST(SessionTest, MultipleResourceManagers) {
  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx[3];
  std::optional<size_t> ShutdownOpIdx[3];

  {
    Session S(noErrors);
    for (size_t I = 0; I != 3; ++I)
      S.addResourceManager(std::make_unique<MockResourceManager>(
          DetachOpIdx[I], ShutdownOpIdx[I], OpIdx));
  }

  EXPECT_EQ(OpIdx, 3U);
  // Expect shutdown in reverse order.
  for (size_t I = 0; I != 3; ++I) {
    EXPECT_EQ(DetachOpIdx[I], std::nullopt);
    EXPECT_THAT(ShutdownOpIdx[I], Optional(Eq(2 - I)));
  }
}
