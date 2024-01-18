//===- LazyAtomicPointerTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/LazyAtomicPointer.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/ThreadPool.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(LazyAtomicPointer, loadOrGenerate) {
  int Value = 0;
  LazyAtomicPointer<int> Ptr;
  ThreadPool Threads;
  for (unsigned I = 0; I < 4; ++I)
    Threads.async([&]() {
      Ptr.loadOrGenerate([&]() {
        // Make sure this is only called once.
        static std::atomic<bool> Once(false);
        bool Current = false;
        EXPECT_TRUE(Once.compare_exchange_strong(Current, true));
        return &Value;
      });
    });

  Threads.wait();
  EXPECT_EQ(Ptr.load(), &Value);
}

#if (LLVM_ENABLE_THREADS)
TEST(LazyAtomicPointer, BusyState) {
  int Value = 0;
  LazyAtomicPointer<int> Ptr;
  ThreadPool Threads;

  std::mutex BusyLock, EndLock;
  std::condition_variable Busy, End;
  bool IsBusy = false, IsEnd = false;
  Threads.async([&]() {
    Ptr.loadOrGenerate([&]() {
      // Notify busy state.
      {
        std::lock_guard<std::mutex> Lock(BusyLock);
        IsBusy = true;
      }
      Busy.notify_all();
      std::unique_lock<std::mutex> LEnd(EndLock);
      // Wait for end state.
      End.wait(LEnd, [&]() { return IsEnd; });
      return &Value;
    });
  });

  // Wait for busy state.
  std::unique_lock<std::mutex> LBusy(BusyLock);
  Busy.wait(LBusy, [&]() { return IsBusy; });
  int *ExistingValue = nullptr;
  // Busy state will not exchange the value.
  EXPECT_FALSE(Ptr.compare_exchange_weak(ExistingValue, nullptr));
  // Busy state return nullptr on load/compare_exchange_weak.
  EXPECT_EQ(ExistingValue, nullptr);
  EXPECT_EQ(Ptr.load(), nullptr);

  // End busy state.
  {
    std::lock_guard<std::mutex> Lock(EndLock);
    IsEnd = true;
  }
  End.notify_all();
  Threads.wait();
  EXPECT_EQ(Ptr.load(), &Value);
}
#endif

} // namespace
