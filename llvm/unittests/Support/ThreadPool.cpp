//========- unittests/Support/ThreadPools.cpp - ThreadPools.h tests --========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ThreadPool.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Threading.h"

#include "gtest/gtest.h"

using namespace llvm;

// Fixture for the unittests, allowing to *temporarily* disable the unittests
// on a particular platform
class ThreadPoolTest : public testing::Test {
  Triple Host;
  SmallVector<Triple::ArchType, 4> UnsupportedArchs;
  SmallVector<Triple::OSType, 4> UnsupportedOSs;
  SmallVector<Triple::EnvironmentType, 1> UnsupportedEnvironments;
protected:
  // This is intended for platform as a temporary "XFAIL"
  bool isUnsupportedOSOrEnvironment() {
    Triple Host(Triple::normalize(sys::getProcessTriple()));

    if (find(UnsupportedEnvironments, Host.getEnvironment()) !=
        UnsupportedEnvironments.end())
      return true;

    if (is_contained(UnsupportedOSs, Host.getOS()))
      return true;

    if (is_contained(UnsupportedArchs, Host.getArch()))
      return true;

    return false;
  }

  ThreadPoolTest() {
    // Add unsupported configuration here, example:
    //   UnsupportedArchs.push_back(Triple::x86_64);

    // See https://llvm.org/bugs/show_bug.cgi?id=25829
    UnsupportedArchs.push_back(Triple::ppc64le);
    UnsupportedArchs.push_back(Triple::ppc64);
  }

  /// Make sure this thread not progress faster than the main thread.
  void waitForMainThread() {
    std::unique_lock<std::mutex> LockGuard(WaitMainThreadMutex);
    WaitMainThread.wait(LockGuard, [&] { return MainThreadReady; });
  }

  /// Set the readiness of the main thread.
  void setMainThreadReady() {
    {
      std::unique_lock<std::mutex> LockGuard(WaitMainThreadMutex);
      MainThreadReady = true;
    }
    WaitMainThread.notify_all();
  }

  void SetUp() override { MainThreadReady = false; }

  void RunOnAllSockets(ThreadPoolStrategy S);

  std::condition_variable WaitMainThread;
  std::mutex WaitMainThreadMutex;
  bool MainThreadReady = false;
};

#define CHECK_UNSUPPORTED() \
  do { \
    if (isUnsupportedOSOrEnvironment()) \
      return; \
  } while (0); \

TEST_F(ThreadPoolTest, AsyncBarrier) {
  CHECK_UNSUPPORTED();
  // test that async & barrier work together properly.

  std::atomic_int checked_in{0};

  ThreadPool Pool;
  for (size_t i = 0; i < 5; ++i) {
    Pool.async([this, &checked_in] {
      waitForMainThread();
      ++checked_in;
    });
  }
  ASSERT_EQ(0, checked_in);
  setMainThreadReady();
  Pool.wait();
  ASSERT_EQ(5, checked_in);
}

static void TestFunc(std::atomic_int &checked_in, int i) { checked_in += i; }

TEST_F(ThreadPoolTest, AsyncBarrierArgs) {
  CHECK_UNSUPPORTED();
  // Test that async works with a function requiring multiple parameters.
  std::atomic_int checked_in{0};

  ThreadPool Pool;
  for (size_t i = 0; i < 5; ++i) {
    Pool.async(TestFunc, std::ref(checked_in), i);
  }
  Pool.wait();
  ASSERT_EQ(10, checked_in);
}

TEST_F(ThreadPoolTest, Async) {
  CHECK_UNSUPPORTED();
  ThreadPool Pool;
  std::atomic_int i{0};
  Pool.async([this, &i] {
    waitForMainThread();
    ++i;
  });
  Pool.async([&i] { ++i; });
  ASSERT_NE(2, i.load());
  setMainThreadReady();
  Pool.wait();
  ASSERT_EQ(2, i.load());
}

TEST_F(ThreadPoolTest, GetFuture) {
  CHECK_UNSUPPORTED();
  ThreadPool Pool(hardware_concurrency(2));
  std::atomic_int i{0};
  Pool.async([this, &i] {
    waitForMainThread();
    ++i;
  });
  // Force the future using get()
  Pool.async([&i] { ++i; }).get();
  ASSERT_NE(2, i.load());
  setMainThreadReady();
  Pool.wait();
  ASSERT_EQ(2, i.load());
}

TEST_F(ThreadPoolTest, PoolDestruction) {
  CHECK_UNSUPPORTED();
  // Test that we are waiting on destruction
  std::atomic_int checked_in{0};
  {
    ThreadPool Pool;
    for (size_t i = 0; i < 5; ++i) {
      Pool.async([this, &checked_in] {
        waitForMainThread();
        ++checked_in;
      });
    }
    ASSERT_EQ(0, checked_in);
    setMainThreadReady();
  }
  ASSERT_EQ(5, checked_in);
}

#if LLVM_ENABLE_THREADS == 1

void ThreadPoolTest::RunOnAllSockets(ThreadPoolStrategy S) {
  // FIXME: Skip these tests on non-Windows because multi-socket system were not
  // tested on Unix yet, and llvm::get_thread_affinity_mask() isn't implemented
  // for Unix.
  Triple Host(Triple::normalize(sys::getProcessTriple()));
  if (!Host.isOSWindows())
    return;

  llvm::DenseSet<llvm::BitVector> ThreadsUsed;
  std::mutex Lock;
  {
    std::condition_variable AllThreads;
    std::mutex AllThreadsLock;
    unsigned Active = 0;

    ThreadPool Pool(S);
    for (size_t I = 0; I < S.compute_thread_count(); ++I) {
      Pool.async([&] {
        {
          std::lock_guard<std::mutex> Guard(AllThreadsLock);
          ++Active;
          AllThreads.notify_one();
        }
        waitForMainThread();
        std::lock_guard<std::mutex> Guard(Lock);
        auto Mask = llvm::get_thread_affinity_mask();
        ThreadsUsed.insert(Mask);
      });
    }
    ASSERT_EQ(true, ThreadsUsed.empty());
    {
      std::unique_lock<std::mutex> Guard(AllThreadsLock);
      AllThreads.wait(Guard,
                      [&]() { return Active == S.compute_thread_count(); });
    }
    setMainThreadReady();
  }
  ASSERT_EQ(llvm::get_cpus(), ThreadsUsed.size());
}

TEST_F(ThreadPoolTest, AllThreads_UseAllRessources) {
  CHECK_UNSUPPORTED();
  RunOnAllSockets({});
}

TEST_F(ThreadPoolTest, AllThreads_OneThreadPerCore) {
  CHECK_UNSUPPORTED();
  RunOnAllSockets(llvm::heavyweight_hardware_concurrency());
}

#endif
