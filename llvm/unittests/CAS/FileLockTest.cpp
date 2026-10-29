//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OnDiskCommonUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace llvm;
using namespace llvm::cas::ondisk;

/// When an exclusive lock is released, all of the threads blocked waiting for
/// a shared lock on the same file should be woken.
TEST(CASFileLockTest, SharedWaitersWokenTogether) {
  constexpr unsigned NumWaiters = 4;
  unittest::TempDir Temp("cas-file-lock", /*Unique=*/true);
  SmallString<128> LockPath(Temp.path("lock"));

  int FD = -1;
  ASSERT_FALSE(sys::fs::openFileForReadWrite(
      LockPath, FD, sys::fs::CD_OpenAlways, sys::fs::OF_None));
  auto CloseFD = llvm::scope_exit([&] {
    sys::fs::file_t File = sys::fs::convertFDToNativeFile(FD);
    sys::fs::closeFile(File);
  });
  ASSERT_FALSE(lockFileThreadSafe(FD, sys::fs::LockKind::Exclusive));

  std::mutex M;
  std::condition_variable CV;
  unsigned NumStarted = 0, NumAcquired = 0;
  bool Released = false;

  std::vector<std::thread> Waiters;
  for (unsigned I = 0; I != NumWaiters; ++I) {
    Waiters.emplace_back([&] {
      int WaiterFD = -1;
      ASSERT_FALSE(sys::fs::openFileForReadWrite(
          LockPath, WaiterFD, sys::fs::CD_OpenAlways, sys::fs::OF_None));
      auto CloseWaiterFD = llvm::scope_exit([&] {
        sys::fs::file_t WaiterFile = sys::fs::convertFDToNativeFile(WaiterFD);
        sys::fs::closeFile(WaiterFile);
      });
      {
        std::lock_guard<std::mutex> Lock(M);
        ++NumStarted;
        CV.notify_all();
      }
      ASSERT_FALSE(lockFileThreadSafe(WaiterFD, sys::fs::LockKind::Shared));
      {
        // Hold the shared lock until all the waiters have been counted.
        std::unique_lock<std::mutex> Lock(M);
        ++NumAcquired;
        CV.notify_all();
        CV.wait(Lock, [&] { return Released; });
      }
      ASSERT_FALSE(unlockFileThreadSafe(WaiterFD));
    });
  }

  {
    std::unique_lock<std::mutex> Lock(M);
    CV.wait(Lock, [&] { return NumStarted == NumWaiters; });
  }
  // Give waiters some time to queue up. If this is too short it can cause the
  // test to pass spuriously since waiters will be trying to get the shared
  // lock for the first time.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  {
    std::lock_guard<std::mutex> Lock(M);
    EXPECT_EQ(NumAcquired, 0u)
        << "shared lock acquired while exclusive lock was held";
  }

  EXPECT_FALSE(unlockFileThreadSafe(FD));

  // Readers hold their shared lock until Released is set, so count how many
  // acquire access in 10 seconds. We expect all of them.
  unsigned NumConcurrentReads;
  {
    std::unique_lock<std::mutex> Lock(M);
    CV.wait_for(Lock, std::chrono::seconds(10),
                [&] { return NumAcquired == NumWaiters; });
    NumConcurrentReads = NumAcquired;
    Released = true;
    CV.notify_all();
  }
  for (std::thread &T : Waiters)
    T.join();

  EXPECT_EQ(NumConcurrentReads, NumWaiters)
      << "after releasing the exclusive lock only " << NumConcurrentReads
      << " of " << NumWaiters << " shared lock waiters were woken";
}
