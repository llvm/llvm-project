//===- CooperativeFutureTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt/CooperativeFuture.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/CooperativeFuture.h"
#include "orc-rt/QueueingTaskDispatcher.h"
#include "gtest/gtest.h"

#include <memory>

using namespace orc_rt;

namespace {

class QueueingTaskDispatchRunner : public CooperativeFutureTaskRunner {
public:
  QueueingTaskDispatchRunner(QueueingTaskDispatcher &QTD) : QTD(QTD) {}

  Error runNextTask() override {
    ++RunNextTaskCalls;
    if (auto T = QTD.pop_back()) {
      T->run();
      return Error::success();
    }
    return make_error<StringError>("No tasks left");
  }
  size_t getRunNextTaskCalls() const { return RunNextTaskCalls; }

private:
  QueueingTaskDispatcher &QTD;
  size_t RunNextTaskCalls = 0;
};

} // anonymous namespace

using namespace orc_rt;

TEST(CooperativeFutureTest, PromiseDefaultConstructionAndDestruction) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  // Test that promise can be created and destroyed without issues
  CooperativePromise<Expected<int>> P(QTDRunner);
}

TEST(CooperativeFutureTest, FutureDefaultConstructionAndDestruction) {
  // Test that future can be created and destroyed without issues
  CooperativeFuture<Expected<int>> F;
}

TEST(CooperativeFutureTest, BasicPromiseFuturePairValue) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  // Create promise and get future
  CooperativePromise<Expected<int>> P(QTDRunner);
  auto F = P.get_future();

  // Set value on promise
  P.set_value(Expected<int>(42));

  // Get value from future
  auto Result = F.get();
  EXPECT_TRUE(!!Result);
  EXPECT_EQ(*Result, 42);
}

TEST(CooperativeFutureTest, BasicPromiseFuturePairError) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  CooperativePromise<Expected<int>> P(QTDRunner);
  auto F = P.get_future();

  // Set error value
  P.set_value(make_error<StringError>("Test error"));

  // Get error from future
  auto Result = F.get();
  EXPECT_FALSE(!!Result);
  EXPECT_EQ(toString(Result.takeError()), "Test error");
}

TEST(CooperativeFutureTest, WorkStealingDuringWait) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  CooperativePromise<Expected<int>> P(QTDRunner);
  auto F = P.get_future();

  // Dispatch a task that will set the promise value
  QTD.dispatch(makeGenericTask([&]() { P.set_value(Expected<int>(42)); }));

  // Getting the value should cause the task to be run
  auto Result = F.get();

  // Verify task ran and value was set
  EXPECT_TRUE(!!Result);
  EXPECT_EQ(*Result, 42);
}

TEST(CooperativeFutureTest, MultipleTasksBeforeValue) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  int TasksRun = 0;

  CooperativePromise<Expected<int>> P(QTDRunner);
  auto F = P.get_future();

  // Dispatch a task to set the promise. TaskWorkQueue runs tasks in LIFO
  // order, so this will run last.
  QTD.dispatch(makeGenericTask([&P]() { P.set_value(Expected<int>(42)); }));

  // Dispatch several other tasks to run before the one that sets the promise.
  for (int I = 0; I < 3; ++I)
    QTD.dispatch(makeGenericTask([&TasksRun]() { ++TasksRun; }));

  // Could get should run all tasks until value is set
  auto Result = F.get();

  EXPECT_EQ(TasksRun, 3);
  EXPECT_TRUE(!!Result);
  EXPECT_EQ(*Result, 42);
}

TEST(CooperativeFutureTest, PromiseMoveConstruction) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  CooperativePromise<Expected<int>> P1(QTDRunner);
  auto F = P1.get_future();

  // Move promise
  CooperativePromise<Expected<int>> P2 = std::move(P1);

  // Set value on moved promise
  P2.set_value(Expected<int>(42));

  auto Result = F.get();
  EXPECT_TRUE(!!Result);
  EXPECT_EQ(*Result, 42);
}

TEST(CooperativeFutureTest, FutureMoveConstruction) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  CooperativePromise<Expected<int>> P(QTDRunner);
  auto F1 = P.get_future();

  // Move future
  CooperativeFuture<Expected<int>> F2 = std::move(F1);

  P.set_value(Expected<int>(42));

  auto Result = F2.get();
  EXPECT_TRUE(!!Result);
  EXPECT_EQ(*Result, 42);
}

TEST(CooperativeFutureTest, PromiseAssignment) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  CooperativePromise<Expected<int>> P1(QTDRunner);
  CooperativePromise<Expected<int>> P2;
  auto F = P1.get_future();

  // Move-assign promise
  P2 = std::move(P1);

  P2.set_value(Expected<int>(42));

  auto Result = F.get();
  EXPECT_TRUE(!!Result);
  EXPECT_EQ(*Result, 42);
}

TEST(CooperativeFutureTest, FutureAssignment) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  CooperativePromise<Expected<int>> P(QTDRunner);
  CooperativeFuture<Expected<int>> F1 = P.get_future();
  CooperativeFuture<Expected<int>> F2;

  // Move-assign future
  F2 = std::move(F1);

  P.set_value(Expected<int>(42));

  auto Result = F2.get();
  EXPECT_TRUE(!!Result);
  EXPECT_EQ(*Result, 42);
}

TEST(CooperativeFutureTest, ErrorTypeValue) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  // Test with Error as the future type
  CooperativePromise<Error> P(QTDRunner);
  auto F = P.get_future();

  P.set_value(Error::success());

  auto Result = F.get();
  EXPECT_FALSE(!!Result); // Error::success() evaluates to false
}

TEST(CooperativeFutureTest, ErrorTypeError) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  CooperativePromise<Error> P(QTDRunner);
  auto F = P.get_future();

  P.set_value(make_error<StringError>("Error type test"));

  auto Result = F.get();
  EXPECT_TRUE(!!Result); // Error with content evaluates to true
  EXPECT_EQ(toString(std::move(Result)), "Error type test");
}

TEST(CooperativeFutureTest, WorkQueueFailure) {
  QueueingTaskDispatcher QTD;
  QueueingTaskDispatchRunner QTDRunner(QTD);

  CooperativePromise<Expected<int>> P(QTDRunner);
  auto F = P.get_future();

  // With no value set and no tasks to run available this should return / throw
  // an error.
#if ORC_RT_ENABLE_EXCEPTIONS
  try {
    auto Result = F.get();
    ADD_FAILURE() << "CooperativeFuture::get() did not throw on empty queue";
  } catch (ErrorInfoBase &EIB) {
    SUCCEED();
  }
#else
  auto Result = F.get();

  EXPECT_FALSE(!!Result);
  consumeError(Result.takeError());
  EXPECT_GT(QTDRunner.getRunNextTaskCalls(), 0u);
#endif // ORC_RT_ENABLE_EXCEPTIONS
}
