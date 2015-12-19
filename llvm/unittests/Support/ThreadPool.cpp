//========- unittests/Support/ThreadPools.cpp - ThreadPools.h tests --========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ThreadPool.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"

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

    if (std::find(UnsupportedEnvironments.begin(), UnsupportedEnvironments.end(),
                  Host.getEnvironment()) != UnsupportedEnvironments.end())
      return true;

    if (std::find(UnsupportedOSs.begin(), UnsupportedOSs.end(), Host.getOS())
        != UnsupportedOSs.end())
      return true;

    if (std::find(UnsupportedArchs.begin(), UnsupportedArchs.end(), Host.getArch())
        != UnsupportedArchs.end())
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
  void setMainThreadReadyState(bool Ready) {
    std::unique_lock<std::mutex> LockGuard(WaitMainThreadMutex);
    MainThreadReady = Ready;
    WaitMainThread.notify_all();
  }

  std::condition_variable WaitMainThread;
  std::mutex WaitMainThreadMutex;
  bool MainThreadReady;

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

  setMainThreadReadyState(false);
  ThreadPool Pool;
  for (size_t i = 0; i < 5; ++i) {
    Pool.async([this, &checked_in, i] {
      waitForMainThread();
      ++checked_in;
    });
  }
  ASSERT_EQ(0, checked_in);
  setMainThreadReadyState(true);
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
  setMainThreadReadyState(false);
  Pool.async([this, &i] {
    waitForMainThread();
    ++i;
  });
  Pool.async([&i] { ++i; });
  ASSERT_NE(2, i.load());
  setMainThreadReadyState(true);
  Pool.wait();
  ASSERT_EQ(2, i.load());
}

TEST_F(ThreadPoolTest, GetFuture) {
  CHECK_UNSUPPORTED();
  ThreadPool Pool;
  std::atomic_int i{0};
  setMainThreadReadyState(false);
  Pool.async([this, &i] {
    waitForMainThread();
    ++i;
  });
  // Force the future using get()
  Pool.async([&i] { ++i; }).get();
  ASSERT_NE(2, i.load());
  setMainThreadReadyState(true);
  Pool.wait();
  ASSERT_EQ(2, i.load());
}

TEST_F(ThreadPoolTest, PoolDestruction) {
  CHECK_UNSUPPORTED();
  // Test that we are waiting on destruction
  std::atomic_int checked_in{0};
  {
    setMainThreadReadyState(false);
    ThreadPool Pool;
    for (size_t i = 0; i < 5; ++i) {
      Pool.async([this, &checked_in, i] {
        waitForMainThread();
        ++checked_in;
      });
    }
    ASSERT_EQ(0, checked_in);
    setMainThreadReadyState(true);
  }
  ASSERT_EQ(5, checked_in);
}
