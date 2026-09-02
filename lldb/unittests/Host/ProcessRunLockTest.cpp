//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/ProcessRunLock.h"

#include "gtest/gtest.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

using namespace lldb_private;

TEST(ProcessRunLockTest, BasicLockUnlock) {
  ProcessRunLock lock;
  ProcessRunLock::ProcessRunLocker locker;
  EXPECT_FALSE(locker.IsLocked());
  EXPECT_TRUE(locker.TryLock(&lock));
  EXPECT_TRUE(locker.IsLocked());
  // Destructor releases.
}

TEST(ProcessRunLockTest, SameThreadRecursion) {
  ProcessRunLock lock;
  ProcessRunLock::ProcessRunLocker outer;
  EXPECT_TRUE(outer.TryLock(&lock));

  // Re-entrant acquisition on the same thread must succeed without
  // deadlocking on the underlying rwlock.
  ProcessRunLock::ProcessRunLocker inner;
  EXPECT_TRUE(inner.TryLock(&lock));
  EXPECT_TRUE(inner.IsLocked());
}

TEST(ProcessRunLockTest, ConcurrentReaders) {
  ProcessRunLock lock;
  ProcessRunLock::ProcessRunLocker outer;
  EXPECT_TRUE(outer.TryLock(&lock));

  // Another thread concurrently acquires the same lock and is also
  // able to recurse on its own. Both threads must get the same-thread
  // recursion fast-path.
  bool other_outer_ok = false, other_inner_ok = false;
  std::thread t([&] {
    ProcessRunLock::ProcessRunLocker thread_outer;
    other_outer_ok = thread_outer.TryLock(&lock);
    ProcessRunLock::ProcessRunLocker thread_inner;
    other_inner_ok = thread_inner.TryLock(&lock);
  });
  t.join();
  EXPECT_TRUE(other_outer_ok);
  EXPECT_TRUE(other_inner_ok);
}

TEST(ProcessRunLockTest, MoveUnlocked) {
  // Moving an unlocked locker is always allowed.
  ProcessRunLock::ProcessRunLocker a;
  ProcessRunLock::ProcessRunLocker b(std::move(a));
  EXPECT_FALSE(b.IsLocked());

  ProcessRunLock::ProcessRunLocker c;
  c = std::move(b);
  EXPECT_FALSE(c.IsLocked());
}

TEST(ProcessRunLockTest, MoveLockedSameThread) {
  // Moving a held locker on the owning thread is allowed.
  ProcessRunLock lock;
  ProcessRunLock::ProcessRunLocker a;
  EXPECT_TRUE(a.TryLock(&lock));
  ProcessRunLock::ProcessRunLocker b(std::move(a));
  EXPECT_TRUE(b.IsLocked());
  EXPECT_FALSE(a.IsLocked());
}

TEST(ProcessRunLockTest, MoveAssignReleasesWaitingWriter) {
  // Move-assigning a fresh locker into a held one calls Unlock() on
  // the destination. If the rwlock isn't actually released, the writer
  // hangs in pthread_rwlock_wrlock and never reaches the flag.
  ProcessRunLock lock;
  ProcessRunLock::ProcessRunLocker reader;
  ASSERT_TRUE(reader.TryLock(&lock));

  bool writer_acquired = false;
  std::thread writer([&] {
    lock.SetRunning();
    writer_acquired = true;
  });

  ProcessRunLock::ProcessRunLocker replacement;
  reader = std::move(replacement);
  EXPECT_FALSE(reader.IsLocked());

  writer.join();
  EXPECT_TRUE(writer_acquired);
}

TEST(ProcessRunLockTest, MoveAssignFromHeldKeepsWriterWaiting) {
  // Opposite of MoveAssignReleasesWaitingWriter: move-assigning a held
  // locker into a fresh one transfers ownership without releasing the
  // rwlock. The writer must only succeed *after* new_owner is destroyed.
  ProcessRunLock lock;
  ProcessRunLock::ProcessRunLocker source;
  ASSERT_TRUE(source.TryLock(&lock));

  std::mutex sync_mutex;
  std::condition_variable sync_cv;
  bool writer_started = false;
  bool release_signaled = false;
  bool acquired_after_release = false;

  std::thread writer([&] {
    {
      std::lock_guard<std::mutex> g(sync_mutex);
      writer_started = true;
    }
    sync_cv.notify_one();
    lock.SetRunning();
    // If move-assign held the lock correctly, the writer only gets here
    // after the listener signaled release. If move-assign released
    // early, the writer can race ahead and observe the flag unset.
    std::lock_guard<std::mutex> g(sync_mutex);
    acquired_after_release = release_signaled;
  });

  std::thread listener([&] {
    std::unique_lock<std::mutex> g(sync_mutex);
    sync_cv.wait(g, [&] { return writer_started; });
    release_signaled = true;
  });

  {
    ProcessRunLock::ProcessRunLocker new_owner;
    new_owner = std::move(source);
    EXPECT_TRUE(new_owner.IsLocked());
    EXPECT_FALSE(source.IsLocked());

    listener.join();
    // new_owner destructs here, releasing the rwlock.
  }

  writer.join();
  EXPECT_TRUE(acquired_after_release);
}

TEST(ProcessRunLockTest, MoveUnlockedAcrossThreads) {
  // Moving an unlocked locker across threads is allowed, and the
  // locker can then be used (TryLock + Unlock) entirely on the worker
  // thread. Note that `lock` is owned by the outer scope so it
  // outlives the worker thread (the captured locker holds a pointer
  // into it).
  ProcessRunLock lock;
  ProcessRunLock::ProcessRunLocker a;
  std::thread t([&lock, locker = std::move(a)]() mutable {
    EXPECT_FALSE(locker.IsLocked());
    EXPECT_TRUE(locker.TryLock(&lock));
    EXPECT_TRUE(locker.IsLocked());
    // ~locker runs on this worker thread, the same thread that
    // acquired -- so the destructor's thread-affinity check passes
    // and the rwlock is released here.
  });
  t.join();
}

#if GTEST_HAS_DEATH_TEST
TEST(ProcessRunLockDeathTest, MoveLockedAcrossThreads) {
  ProcessRunLock lock;
  ProcessRunLock::ProcessRunLocker a;
  ASSERT_TRUE(a.TryLock(&lock));
  // The expected message will vary based on the call site so match the
  // "ProcessRunLocker" common prefix only.
  EXPECT_DEATH(
      {
        std::thread t([locker = std::move(a)]() mutable { (void)locker; });
        t.join();
      },
      "ProcessRunLocker");
}
#endif
