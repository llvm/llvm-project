//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TargetAPIMutex.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/Policy.h"
#include "gtest/gtest.h"

#include <thread>

using namespace lldb_private;
using namespace lldb;

TEST(TargetAPIMutexTest, DefaultConstructedIsANoOp) {
  // No synchronization primitive is touched at all in this state, so
  // there is no pairing requirement: try_lock() always succeeds, and
  // lock()/unlock() are callable with no invariant to violate.
  TargetAPIMutex lock;
  EXPECT_TRUE(lock.try_lock());
  lock.lock();
  lock.unlock();
  lock.lock();
  lock.unlock();
}

namespace {
class TargetAPIMutexTargetTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    platform_linux::PlatformLinux::Initialize();
  }
  void TearDown() override {
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};

TargetSP CreateTarget() {
  ArchSpec arch("x86_64-pc-linux");
  Platform::SetHostPlatform(
      platform_linux::PlatformLinux::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  TargetSP target_sp;
  PlatformSP platform_sp;
  Status error = debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
  return target_sp;
}
} // namespace

TEST_F(TargetAPIMutexTargetTest, WrapsTheTargetMutex) {
  TargetSP target_sp = CreateTarget();
  ASSERT_TRUE(target_sp);

  TargetAPIMutex lock(target_sp);
  lock.lock();

  // Recursive reentrancy is delegated straight to the underlying
  // std::recursive_mutex: a second handle over the same target, locked
  // from the same thread, must not block.
  TargetAPIMutex second_lock(target_sp);
  EXPECT_TRUE(second_lock.try_lock());
  second_lock.unlock();

  lock.unlock();

  std::thread t([target_sp]() {
    TargetAPIMutex background_lock(target_sp);
    EXPECT_TRUE(background_lock.try_lock());
    background_lock.unlock();
  });
  t.join();
}

TEST_F(TargetAPIMutexTargetTest, RealMutexBlocksOtherThreads) {
  TargetSP target_sp = CreateTarget();
  ASSERT_TRUE(target_sp);

  TargetAPIMutex lock(target_sp);
  lock.lock();

  std::thread t([target_sp]() {
    TargetAPIMutex background_lock(target_sp);
    EXPECT_FALSE(background_lock.try_lock());
  });
  t.join();

  lock.unlock();
}

TEST_F(TargetAPIMutexTargetTest, BareHandleDoesNotAutoUnlockOnDestruction) {
  // TargetAPIMutex carries no RAII of its own -- exactly like
  // std::recursive_mutex, a bare handle going out of scope without an
  // explicit unlock() leaves the real mutex held. Callers that want
  // scope-based release must wrap it in std::lock_guard/std::unique_lock.
  TargetSP target_sp = CreateTarget();
  ASSERT_TRUE(target_sp);

  {
    TargetAPIMutex lock(target_sp);
    lock.lock();
  }

  std::thread t([target_sp]() {
    TargetAPIMutex background_lock(target_sp);
    EXPECT_FALSE(background_lock.try_lock());
  });
  t.join();

  // Unlock the original locked mutex.
  // Calling try_lock() resolves the underlying mutex and re-enters it on this
  // thread (incrementing the recursive count), so we unlock twice to fully
  // release both acquisitions.
  TargetAPIMutex cleanup_lock(target_sp);
  ASSERT_TRUE(cleanup_lock.try_lock());
  cleanup_lock.unlock();
  cleanup_lock.unlock();
}

TEST_F(TargetAPIMutexTargetTest, LockGuardReleasesOnScopeExit) {
  TargetSP target_sp = CreateTarget();
  ASSERT_TRUE(target_sp);

  {
    TargetAPIMutex lock(target_sp);
    std::lock_guard<TargetAPIMutex> guard(lock);
  }

  std::thread t([target_sp]() {
    TargetAPIMutex background_lock(target_sp);
    EXPECT_TRUE(background_lock.try_lock());
    background_lock.unlock();
  });
  t.join();
}

TEST_F(TargetAPIMutexTargetTest, MoveTransfersOwnership) {
  TargetSP target_sp = CreateTarget();
  ASSERT_TRUE(target_sp);

  TargetAPIMutex lock(target_sp);
  lock.lock();

  TargetAPIMutex moved(std::move(lock));

  // The moved-from handle no longer references the real mutex: unlocking
  // it is a no-op, so the mutex stays held until `moved` releases it.
  lock.unlock();
  std::thread contended([target_sp]() {
    TargetAPIMutex background_lock(target_sp);
    EXPECT_FALSE(background_lock.try_lock());
  });
  contended.join();

  moved.unlock();
  std::thread released([target_sp]() {
    TargetAPIMutex background_lock(target_sp);
    EXPECT_TRUE(background_lock.try_lock());
    background_lock.unlock();
  });
  released.join();
}

TEST_F(TargetAPIMutexTargetTest, ResolvesFreshOnEachLockCall) {
  // lock()/try_lock() re-resolve the real mutex on every call rather
  // than caching a single resolution for the handle's lifetime: a
  // handle can be locked, unlocked, and locked again, each time
  // correctly contending with other threads for the same target mutex.
  TargetSP target_sp = CreateTarget();
  ASSERT_TRUE(target_sp);

  TargetAPIMutex lock(target_sp);
  lock.lock();
  lock.unlock();

  std::thread t([target_sp]() {
    TargetAPIMutex background_lock(target_sp);
    background_lock.lock();
    background_lock.unlock();
  });
  t.join();

  lock.lock();
  std::thread contended([target_sp]() {
    TargetAPIMutex background_lock(target_sp);
    EXPECT_FALSE(background_lock.try_lock());
  });
  contended.join();
  lock.unlock();
}

TEST_F(TargetAPIMutexTargetTest,
       UnlockReplaysLockResolutionAcrossPolicyChange) {
  // lock() and unlock() must agree on which mutex they touch even if the
  // calling thread's policy changes in between: unlock() replays lock()'s
  // resolution rather than re-resolving from the current policy.
  TargetSP target_sp = CreateTarget();
  ASSERT_TRUE(target_sp);

  TargetAPIMutex lock(target_sp);
  lock.lock();

  // If unlock() re-resolved here it would see the bypass and skip releasing
  // the mutex it actually locked.
  {
    PolicyStack::Guard guard = PolicyStack::Get().PushScriptedExtensionCall();
    lock.unlock();
  }

  // The real mutex must have actually been released: a fresh acquisition
  // from a different thread (outside the bypass policy) must succeed
  // immediately. A same-thread try_lock() would pass even if unlock() had
  // incorrectly no-op'd, since std::recursive_mutex lets the same thread
  // reenter a lock it still holds.
  std::thread t([target_sp]() {
    TargetAPIMutex background_lock(target_sp);
    EXPECT_TRUE(background_lock.try_lock());
    background_lock.unlock();
  });
  t.join();
}

TEST_F(TargetAPIMutexTargetTest, BypassPolicyMakesTryLockANoOp) {
  TargetSP target_sp = CreateTarget();
  ASSERT_TRUE(target_sp);

  TargetAPIMutex holder(target_sp);
  holder.lock();

  // The contention has to come from another thread: std::recursive_mutex lets
  // the owning thread reenter a lock it already holds, so a same-thread
  // try_lock() would succeed whether or not the bypass is in effect.
  std::thread contended([target_sp]() {
    TargetAPIMutex background_lock(target_sp);
    EXPECT_FALSE(background_lock.try_lock());
  });
  contended.join();

  // The bypass touches no primitive, so the same acquisition succeeds while
  // the real mutex is held elsewhere.
  std::thread bypassed([target_sp]() {
    PolicyStack::Guard guard = PolicyStack::Get().PushScriptedExtensionCall();
    TargetAPIMutex background_lock(target_sp);
    EXPECT_TRUE(background_lock.try_lock());
    background_lock.unlock();
  });
  bypassed.join();

  holder.unlock();
}
