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

  llvm::thread Thread(ThreadFunc);
  Thread.detach();
  ASSERT_TRUE(ThreadStarted.wait());
  ThreadAdvanced.notify();
  ASSERT_TRUE(ThreadFinished.wait());
}

TEST(Threading, RunOnThreadSync) {
  std::atomic_bool Executed(false);
  llvm::thread Thread(
      [](void *Arg) { *static_cast<std::atomic_bool *>(Arg) = true; },
      &Executed);
  Thread.join();
  ASSERT_EQ(Executed, true);
}

#if defined(__APPLE__)
TEST(Threading, AppleStackSize) {
  llvm::thread Thread([] {
    volatile unsigned char Var[8 * 1024 * 1024 - 10240];
    Var[0] = 0xff;
    ASSERT_EQ(Var[0], 0xff);
  });
  Thread.join();
}
#endif
#endif

} // end anon namespace
