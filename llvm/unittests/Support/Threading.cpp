//===- unittests/Threading.cpp - Thread tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Threading.h"
#include "llvm/Support/thread.h"
#include "gtest/gtest.h"

#include <atomic>
#include <condition_variable>

using namespace llvm;

namespace {

TEST(Threading, PhysicalConcurrency) {
  auto Num = heavyweight_hardware_concurrency();
  // Since Num is unsigned this will also catch us trying to
  // return -1.
  ASSERT_LE(Num.compute_thread_count(),
            hardware_concurrency().compute_thread_count());
}

#if LLVM_ENABLE_THREADS

class Notification {
public:
  void notify() {
    {
      std::lock_guard<std::mutex> Lock(M);
      Notified = true;
      // Broadcast with the lock held, so it's safe to destroy the Notification
      // after wait() returns.
      CV.notify_all();
    }
  }

  bool wait() {
    std::unique_lock<std::mutex> Lock(M);
    using steady_clock = std::chrono::steady_clock;
    auto Deadline = steady_clock::now() +
                    std::chrono::duration_cast<steady_clock::duration>(
                        std::chrono::duration<double>(5));
    return CV.wait_until(Lock, Deadline, [this] { return Notified; });
  }

private:
  bool Notified = false;
  mutable std::condition_variable CV;
  mutable std::mutex M;
};

TEST(Threading, RunOnThreadSyncAsync) {
  Notification ThreadStarted, ThreadAdvanced, ThreadFinished;

  auto ThreadFunc = [&] {
    ThreadStarted.notify();
    ASSERT_TRUE(ThreadAdvanced.wait());
    ThreadFinished.notify();
  };

  llvm::llvm_execute_on_thread_async(ThreadFunc);
  ASSERT_TRUE(ThreadStarted.wait());
  ThreadAdvanced.notify();
  ASSERT_TRUE(ThreadFinished.wait());
}

TEST(Threading, RunOnThreadSync) {
  std::atomic_bool Executed(false);
  llvm::llvm_execute_on_thread(
      [](void *Arg) { *static_cast<std::atomic_bool *>(Arg) = true; },
      &Executed);
  ASSERT_EQ(Executed, true);
}
#endif

} // end anon namespace
