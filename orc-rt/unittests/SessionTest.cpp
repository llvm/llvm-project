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
#include "orc-rt/ThreadPoolTaskDispatcher.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <deque>
#include <future>
#include <optional>

#include <iostream>

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

class NoDispatcher : public TaskDispatcher {
public:
  void dispatch(std::unique_ptr<Task> T) override {
    assert(false && "strictly no dispatching!");
  }
  void shutdown() override {}
};

class EnqueueingDispatcher : public TaskDispatcher {
public:
  using OnShutdownRunFn = move_only_function<void()>;
  EnqueueingDispatcher(std::deque<std::unique_ptr<Task>> &Tasks,
                       OnShutdownRunFn OnShutdownRun = {})
      : Tasks(Tasks), OnShutdownRun(std::move(OnShutdownRun)) {}
  void dispatch(std::unique_ptr<Task> T) override {
    Tasks.push_back(std::move(T));
  }
  void shutdown() override {
    if (OnShutdownRun)
      OnShutdownRun();
  }

private:
  std::deque<std::unique_ptr<Task>> &Tasks;
  OnShutdownRunFn OnShutdownRun;
};

// Non-overloaded version of cantFail: allows easy construction of
// move_only_functions<void(Error)>s.
static void noErrors(Error Err) { cantFail(std::move(Err)); }

TEST(SessionTest, TrivialConstructionAndDestruction) {
  Session S(std::make_unique<NoDispatcher>(), noErrors);
}

TEST(SessionTest, ReportError) {
  Error E = Error::success();
  cantFail(std::move(E)); // Force error into checked state.

  Session S(std::make_unique<NoDispatcher>(),
            [&](Error Err) { E = std::move(Err); });
  S.reportError(make_error<StringError>("foo"));

  if (E)
    EXPECT_EQ(toString(std::move(E)), "foo");
  else
    ADD_FAILURE() << "Missing error value";
}

TEST(SessionTest, DispatchTask) {
  int X = 0;
  std::deque<std::unique_ptr<Task>> Tasks;
  Session S(std::make_unique<EnqueueingDispatcher>(Tasks), noErrors);

  EXPECT_EQ(Tasks.size(), 0U);
  S.dispatch(makeGenericTask([&]() { ++X; }));
  EXPECT_EQ(Tasks.size(), 1U);
  auto T = std::move(Tasks.front());
  Tasks.pop_front();
  T->run();
  EXPECT_EQ(X, 1);
}

TEST(SessionTest, SingleResourceManager) {
  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;

  {
    Session S(std::make_unique<NoDispatcher>(), noErrors);
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
    Session S(std::make_unique<NoDispatcher>(), noErrors);
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

TEST(SessionTest, ExpectedShutdownSequence) {
  // Check that Session shutdown results in...
  // 1. ResourceManagers being shut down.
  // 2. The TaskDispatcher being shut down.
  // 3. A call to OnShutdownComplete.

  size_t OpIdx = 0;
  std::optional<size_t> DetachOpIdx;
  std::optional<size_t> ShutdownOpIdx;

  bool DispatcherShutDown = false;
  bool SessionShutdownComplete = false;
  std::deque<std::unique_ptr<Task>> Tasks;
  Session S(std::make_unique<EnqueueingDispatcher>(
                Tasks,
                [&]() {
                  std::cerr << "Running dispatcher shutdown.\n";
                  EXPECT_TRUE(ShutdownOpIdx);
                  EXPECT_EQ(*ShutdownOpIdx, 0);
                  EXPECT_FALSE(SessionShutdownComplete);
                  DispatcherShutDown = true;
                }),
            noErrors);
  S.addResourceManager(
      std::make_unique<MockResourceManager>(DetachOpIdx, ShutdownOpIdx, OpIdx));

  S.shutdown([&]() {
    EXPECT_TRUE(DispatcherShutDown);
    std::cerr << "Running shutdown callback.\n";
    SessionShutdownComplete = true;
  });
  S.waitForShutdown();

  EXPECT_TRUE(SessionShutdownComplete);
}
