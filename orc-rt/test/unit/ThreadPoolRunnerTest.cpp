//===- ThreadPoolRunnerTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt/ThreadPoolRunner.h"
#include "orc-rt/SPSWrapperFunction.h"
#include "gtest/gtest.h"

#include <atomic>
#include <cstdint>
#include <future>
#include <thread>

using namespace orc_rt;

namespace {

inline orc_rt_SessionRef dummySession() noexcept {
  return reinterpret_cast<orc_rt_SessionRef>(uintptr_t{0xABCD});
}

inline orc_rt_WrapperFunctionReturn dummyReturn() noexcept {
  return [](orc_rt_SessionRef, uint64_t, orc_rt_WrapperFunctionBuffer) {};
}

template <typename T> WrapperFunctionBuffer serializePtr(T *Ptr) {
  auto Buf = WrapperFunctionSPSSerializer<void(SPSExecutorAddr)>::arguments()
                 .serialize(Ptr);
  assert(Buf && "failed to serialize pointer arg");
  return std::move(*Buf);
}

TEST(ThreadPoolRunnerTest, NoCalls) {
  // Check that immediate destruction works as expected.
  ThreadPoolRunner R(1);
}

static void signalPromise(orc_rt_SessionRef S, uint64_t CallId,
                          orc_rt_WrapperFunctionReturn Return,
                          orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<void(SPSExecutorAddr)>::handle(
      S, CallId, Return, ArgBytes,
      [](move_only_function<void()> Return, ExecutorAddr P) {
        P.toPtr<std::promise<void> *>()->set_value();
        Return();
      });
}

TEST(ThreadPoolRunnerTest, BasicCallExecution) {
  // Smoke test: dispatch one call on a single-threaded pool, wait for it to
  // run, then let the runner destruct.
  std::promise<void> Done;
  std::future<void> DoneF = Done.get_future();

  {
    ThreadPoolRunner R(1);
    R(dummySession(), 0, dummyReturn(), signalPromise, serializePtr(&Done));
    DoneF.get();
  }
}

static void incrementCounter(orc_rt_SessionRef S, uint64_t CallId,
                             orc_rt_WrapperFunctionReturn Return,
                             orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<void(SPSExecutorAddr)>::handle(
      S, CallId, Return, ArgBytes,
      [](move_only_function<void()> Return, ExecutorAddr P) {
        ++*P.toPtr<std::atomic<size_t> *>();
        Return();
      });
}

TEST(ThreadPoolRunnerTest, SingleThreadMultipleCalls) {
  // Dispatch multiple calls on a single-threaded pool, wait for all to run,
  // then let the runner destruct.
  size_t NumCallsToRun = 10;
  std::atomic<size_t> CallsRun = 0;

  {
    ThreadPoolRunner R(1);
    for (size_t I = 0; I != NumCallsToRun; ++I)
      R(dummySession(), I, dummyReturn(), incrementCounter,
        serializePtr(&CallsRun));

    // while (CallsRun.load() < NumCallsToRun)
    //   std::this_thread::yield();
  }

  EXPECT_EQ(CallsRun, NumCallsToRun);
}

struct ConcurrencyState {
  std::future<int> FInit;
  std::promise<int> P1;
  std::promise<int> P2;
  std::future<int> F1 = P1.get_future();
  std::future<int> F2 = P2.get_future();
  std::promise<int> PResult;
};

static void concurrencyTaskA(orc_rt_SessionRef S, uint64_t CallId,
                             orc_rt_WrapperFunctionReturn Return,
                             orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<void(SPSExecutorAddr)>::handle(
      S, CallId, Return, ArgBytes,
      [](move_only_function<void()> Return, ExecutorAddr P) {
        auto *State = P.toPtr<ConcurrencyState *>();
        State->P1.set_value(State->FInit.get());
        State->PResult.set_value(State->F2.get());
        Return();
      });
}

static void concurrencyTaskB(orc_rt_SessionRef S, uint64_t CallId,
                             orc_rt_WrapperFunctionReturn Return,
                             orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<void(SPSExecutorAddr)>::handle(
      S, CallId, Return, ArgBytes,
      [](move_only_function<void()> Return, ExecutorAddr P) {
        auto *State = P.toPtr<ConcurrencyState *>();
        State->P2.set_value(State->F1.get());
        Return();
      });
}

TEST(ThreadPoolRunnerTest, ConcurrentCalls) {
  // Check that calls run concurrently when multiple workers are available.
  // Two calls communicate values back and forth via futures; neither can
  // complete without the other having started. FResult.get() also serves
  // as the "all calls have run" wait point before destruction.
  std::promise<int> PInit;
  ConcurrencyState State;
  State.FInit = PInit.get_future();
  std::future<int> FResult = State.PResult.get_future();

  int ExpectedValue = 42;

  {
    ThreadPoolRunner R(2);
    R(dummySession(), 0, dummyReturn(), concurrencyTaskA, serializePtr(&State));
    R(dummySession(), 1, dummyReturn(), concurrencyTaskB, serializePtr(&State));

    PInit.set_value(ExpectedValue);

    EXPECT_EQ(FResult.get(), ExpectedValue);
  }
}

} // namespace
