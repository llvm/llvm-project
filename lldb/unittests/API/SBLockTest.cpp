//===-- SBLockTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "lldb/API/SBLock.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBTarget.h"
#include "gtest/gtest.h"
#include <atomic>
#include <chrono>
#include <thread>

TEST(SBLockTest, LockTest) {
  lldb::SBDebugger debugger = lldb::SBDebugger::Create();
  lldb::SBTarget target = debugger.GetDummyTarget();

  std::mutex m;
  std::condition_variable cv;
  bool wakeup = false;
  std::atomic<bool> locked = false;

  std::thread test_thread([&]() {
    {
      {
        std::unique_lock lk(m);
        cv.wait(lk, [&] { return wakeup; });
      }

      ASSERT_TRUE(locked);
      target.BreakpointCreateByName("foo", "bar");
      ASSERT_FALSE(locked);

      cv.notify_one();
    }
  });

  // Take the API lock.
  {
    lldb::SBLock lock = target.GetAPILock();
    ASSERT_FALSE(locked.exchange(true));

    // Wake up the test thread.
    {
      std::lock_guard lk(m);
      wakeup = true;
    }
    cv.notify_one();

    // Wait 500ms to confirm the thread is blocked.
    {
      std::unique_lock<std::mutex> lk(m);
      auto result = cv.wait_for(lk, std::chrono::milliseconds(500));
      ASSERT_EQ(result, std::cv_status::timeout);
    }

    ASSERT_TRUE(locked.exchange(false));
  }

  test_thread.join();
}
