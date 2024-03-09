//========- unittests/Support/ThreadPools.cpp - ThreadPools.h tests --========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ThreadPool.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Threading.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#ifdef _WIN32
#include "llvm/Support/Windows/WindowsSupport.h"
#endif

#include <chrono>
#include <thread>

#include "gtest/gtest.h"

namespace testing {
namespace internal {
// Specialize gtest construct to provide friendlier name in the output.
#if LLVM_ENABLE_THREADS
template <> std::string GetTypeName<llvm::StdThreadPool>() {
  return "llvm::StdThreadPool";
}
#endif
template <> std::string GetTypeName<llvm::SingleThreadExecutor>() {
  return "llvm::SingleThreadExecutor";
}
} // namespace internal
} // namespace testing

using namespace llvm;

// Fixture for the unittests, allowing to *temporarily* disable the unittests
// on a particular platform
template <typename ThreadPoolImpl> class ThreadPoolTest : public testing::Test {
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
  void waitForMainThread() { waitForPhase(1); }

  /// Set the readiness of the main thread.
  void setMainThreadReady() { setPhase(1); }

  /// Wait until given phase is set using setPhase(); first "main" phase is 1.
  /// See also PhaseResetHelper below.
  void waitForPhase(int Phase) {
    std::unique_lock<std::mutex> LockGuard(CurrentPhaseMutex);
    CurrentPhaseCondition.wait(
        LockGuard, [&] { return CurrentPhase == Phase || CurrentPhase < 0; });
  }
  /// If a thread waits on another phase, the test could bail out on a failed
  /// assertion and ThreadPool destructor would wait() on all threads, which
  /// would deadlock on the task waiting. Create this helper to automatically
  /// reset the phase and unblock such threads.
  struct PhaseResetHelper {
    PhaseResetHelper(ThreadPoolTest *test) : test(test) {}
    ~PhaseResetHelper() { test->setPhase(-1); }
    ThreadPoolTest *test;
  };

  /// Advance to the given phase.
  void setPhase(int Phase) {
    {
      std::unique_lock<std::mutex> LockGuard(CurrentPhaseMutex);
      assert(Phase == CurrentPhase + 1 || Phase < 0);
      CurrentPhase = Phase;
    }
    CurrentPhaseCondition.notify_all();
  }

  void SetUp() override { CurrentPhase = 0; }

  SmallVector<llvm::BitVector, 0> RunOnAllSockets(ThreadPoolStrategy S);

  std::condition_variable CurrentPhaseCondition;
  std::mutex CurrentPhaseMutex;
  int CurrentPhase; // -1 = error, 0 = setup, 1 = ready, 2+ = custom
};

using ThreadPoolImpls = ::testing::Types<
#if LLVM_ENABLE_THREADS
    StdThreadPool,
#endif
    SingleThreadExecutor>;

TYPED_TEST_SUITE(ThreadPoolTest, ThreadPoolImpls);

#define CHECK_UNSUPPORTED()                                                    \
  do {                                                                         \
    if (this->isUnsupportedOSOrEnvironment())                                  \
      GTEST_SKIP();                                                            \
  } while (0);

TYPED_TEST(ThreadPoolTest, AsyncBarrier) {
  CHECK_UNSUPPORTED();
  // test that async & barrier work together properly.

  std::atomic_int checked_in{0};

  DefaultThreadPool Pool;
  for (size_t i = 0; i < 5; ++i) {
    Pool.async([this, &checked_in] {
      this->waitForMainThread();
      ++checked_in;
    });
  }
  ASSERT_EQ(0, checked_in);
  this->setMainThreadReady();
  Pool.wait();
  ASSERT_EQ(5, checked_in);
}

static void TestFunc(std::atomic_int &checked_in, int i) { checked_in += i; }

TYPED_TEST(ThreadPoolTest, AsyncBarrierArgs) {
  CHECK_UNSUPPORTED();
  // Test that async works with a function requiring multiple parameters.
  std::atomic_int checked_in{0};

  DefaultThreadPool Pool;
  for (size_t i = 0; i < 5; ++i) {
    Pool.async(TestFunc, std::ref(checked_in), i);
  }
  Pool.wait();
  ASSERT_EQ(10, checked_in);
}

TYPED_TEST(ThreadPoolTest, Async) {
  CHECK_UNSUPPORTED();
  DefaultThreadPool Pool;
  std::atomic_int i{0};
  Pool.async([this, &i] {
    this->waitForMainThread();
    ++i;
  });
  Pool.async([&i] { ++i; });
  ASSERT_NE(2, i.load());
  this->setMainThreadReady();
  Pool.wait();
  ASSERT_EQ(2, i.load());
}

TYPED_TEST(ThreadPoolTest, GetFuture) {
  CHECK_UNSUPPORTED();
  DefaultThreadPool Pool(hardware_concurrency(2));
  std::atomic_int i{0};
  Pool.async([this, &i] {
    this->waitForMainThread();
    ++i;
  });
  // Force the future using get()
  Pool.async([&i] { ++i; }).get();
  ASSERT_NE(2, i.load());
  this->setMainThreadReady();
  Pool.wait();
  ASSERT_EQ(2, i.load());
}

TYPED_TEST(ThreadPoolTest, GetFutureWithResult) {
  CHECK_UNSUPPORTED();
  DefaultThreadPool Pool(hardware_concurrency(2));
  auto F1 = Pool.async([] { return 1; });
  auto F2 = Pool.async([] { return 2; });

  this->setMainThreadReady();
  Pool.wait();
  ASSERT_EQ(1, F1.get());
  ASSERT_EQ(2, F2.get());
}

TYPED_TEST(ThreadPoolTest, GetFutureWithResultAndArgs) {
  CHECK_UNSUPPORTED();
  DefaultThreadPool Pool(hardware_concurrency(2));
  auto Fn = [](int x) { return x; };
  auto F1 = Pool.async(Fn, 1);
  auto F2 = Pool.async(Fn, 2);

  this->setMainThreadReady();
  Pool.wait();
  ASSERT_EQ(1, F1.get());
  ASSERT_EQ(2, F2.get());
}

TYPED_TEST(ThreadPoolTest, PoolDestruction) {
  CHECK_UNSUPPORTED();
  // Test that we are waiting on destruction
  std::atomic_int checked_in{0};
  {
    DefaultThreadPool Pool;
    for (size_t i = 0; i < 5; ++i) {
      Pool.async([this, &checked_in] {
        this->waitForMainThread();
        ++checked_in;
      });
    }
    ASSERT_EQ(0, checked_in);
    this->setMainThreadReady();
  }
  ASSERT_EQ(5, checked_in);
}

// Check running tasks in different groups.
TYPED_TEST(ThreadPoolTest, Groups) {
  CHECK_UNSUPPORTED();
  // Need at least two threads, as the task in group2
  // might block a thread until all tasks in group1 finish.
  ThreadPoolStrategy S = hardware_concurrency(2);
  if (S.compute_thread_count() < 2)
    GTEST_SKIP();
  DefaultThreadPool Pool(S);
  typename TestFixture::PhaseResetHelper Helper(this);
  ThreadPoolTaskGroup Group1(Pool);
  ThreadPoolTaskGroup Group2(Pool);

  // Check that waiting for an empty group is a no-op.
  Group1.wait();

  std::atomic_int checked_in1{0};
  std::atomic_int checked_in2{0};

  for (size_t i = 0; i < 5; ++i) {
    Group1.async([this, &checked_in1] {
      this->waitForMainThread();
      ++checked_in1;
    });
  }
  Group2.async([this, &checked_in2] {
    this->waitForPhase(2);
    ++checked_in2;
  });
  ASSERT_EQ(0, checked_in1);
  ASSERT_EQ(0, checked_in2);
  // Start first group and wait for it.
  this->setMainThreadReady();
  Group1.wait();
  ASSERT_EQ(5, checked_in1);
  // Second group has not yet finished, start it and wait for it.
  ASSERT_EQ(0, checked_in2);
  this->setPhase(2);
  Group2.wait();
  ASSERT_EQ(5, checked_in1);
  ASSERT_EQ(1, checked_in2);
}

// Check recursive tasks.
TYPED_TEST(ThreadPoolTest, RecursiveGroups) {
  CHECK_UNSUPPORTED();
  DefaultThreadPool Pool;
  ThreadPoolTaskGroup Group(Pool);

  std::atomic_int checked_in1{0};

  for (size_t i = 0; i < 5; ++i) {
    Group.async([this, &Pool, &checked_in1] {
      this->waitForMainThread();

      ThreadPoolTaskGroup LocalGroup(Pool);

      // Check that waiting for an empty group is a no-op.
      LocalGroup.wait();

      std::atomic_int checked_in2{0};
      for (size_t i = 0; i < 5; ++i) {
        LocalGroup.async([&checked_in2] { ++checked_in2; });
      }
      LocalGroup.wait();
      ASSERT_EQ(5, checked_in2);

      ++checked_in1;
    });
  }
  ASSERT_EQ(0, checked_in1);
  this->setMainThreadReady();
  Group.wait();
  ASSERT_EQ(5, checked_in1);
}

TYPED_TEST(ThreadPoolTest, RecursiveWaitDeadlock) {
  CHECK_UNSUPPORTED();
  ThreadPoolStrategy S = hardware_concurrency(2);
  if (S.compute_thread_count() < 2)
    GTEST_SKIP();
  DefaultThreadPool Pool(S);
  typename TestFixture::PhaseResetHelper Helper(this);
  ThreadPoolTaskGroup Group(Pool);

  // Test that a thread calling wait() for a group and is waiting for more tasks
  // returns when the last task finishes in a different thread while the waiting
  // thread was waiting for more tasks to process while waiting.

  // Task A runs in the first thread. It finishes and leaves
  // the background thread waiting for more tasks.
  Group.async([this] {
    this->waitForMainThread();
    this->setPhase(2);
  });
  // Task B is run in a second thread, it launches yet another
  // task C in a different group, which will be handled by the waiting
  // thread started above.
  Group.async([this, &Pool] {
    this->waitForPhase(2);
    ThreadPoolTaskGroup LocalGroup(Pool);
    LocalGroup.async([this] {
      this->waitForPhase(3);
      // Give the other thread enough time to check that there's no task
      // to process and suspend waiting for a notification. This is indeed racy,
      // but probably the best that can be done.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });
    // And task B only now will wait for the tasks in the group (=task C)
    // to finish. This test checks that it does not deadlock. If the
    // `NotifyGroup` handling in ThreadPool::processTasks() didn't take place,
    // this task B would be stuck waiting for tasks to arrive.
    this->setPhase(3);
    LocalGroup.wait();
  });
  this->setMainThreadReady();
  Group.wait();
}

#if LLVM_ENABLE_THREADS == 1

// FIXME: Skip some tests below on non-Windows because multi-socket systems
// were not fully tested on Unix yet, and llvm::get_thread_affinity_mask()
// isn't implemented for Unix (need AffinityMask in Support/Unix/Program.inc).
#ifdef _WIN32

template <typename ThreadPoolImpl>
SmallVector<llvm::BitVector, 0>
ThreadPoolTest<ThreadPoolImpl>::RunOnAllSockets(ThreadPoolStrategy S) {
  llvm::SetVector<llvm::BitVector> ThreadsUsed;
  std::mutex Lock;
  {
    std::condition_variable AllThreads;
    std::mutex AllThreadsLock;
    unsigned Active = 0;

    DefaultThreadPool Pool(S);
    for (size_t I = 0; I < S.compute_thread_count(); ++I) {
      Pool.async([&] {
        {
          std::lock_guard<std::mutex> Guard(AllThreadsLock);
          ++Active;
          AllThreads.notify_one();
        }
        this->waitForMainThread();
        std::lock_guard<std::mutex> Guard(Lock);
        auto Mask = llvm::get_thread_affinity_mask();
        ThreadsUsed.insert(Mask);
      });
    }
    EXPECT_EQ(true, ThreadsUsed.empty());
    {
      std::unique_lock<std::mutex> Guard(AllThreadsLock);
      AllThreads.wait(Guard,
                      [&]() { return Active == S.compute_thread_count(); });
    }
    this->setMainThreadReady();
  }
  return ThreadsUsed.takeVector();
}

TYPED_TEST(ThreadPoolTest, AllThreads_UseAllRessources) {
  CHECK_UNSUPPORTED();
  // After Windows 11, the OS is free to deploy the threads on any CPU socket.
  // We cannot relibly ensure that all thread affinity mask are covered,
  // therefore this test should not run.
  if (llvm::RunningWindows11OrGreater())
    GTEST_SKIP();
  auto ThreadsUsed = this->RunOnAllSockets({});
  ASSERT_EQ(llvm::get_cpus(), ThreadsUsed.size());
}

TYPED_TEST(ThreadPoolTest, AllThreads_OneThreadPerCore) {
  CHECK_UNSUPPORTED();
  // After Windows 11, the OS is free to deploy the threads on any CPU socket.
  // We cannot relibly ensure that all thread affinity mask are covered,
  // therefore this test should not run.
  if (llvm::RunningWindows11OrGreater())
    GTEST_SKIP();
  auto ThreadsUsed =
      this->RunOnAllSockets(llvm::heavyweight_hardware_concurrency());
  ASSERT_EQ(llvm::get_cpus(), ThreadsUsed.size());
}

// From TestMain.cpp.
extern const char *TestMainArgv0;

// Just a reachable symbol to ease resolving of the executable's path.
static cl::opt<std::string> ThreadPoolTestStringArg1("thread-pool-string-arg1");

#ifdef _WIN32
#define setenv(name, var, ignore) _putenv_s(name, var)
#endif

TYPED_TEST(ThreadPoolTest, AffinityMask) {
  CHECK_UNSUPPORTED();

  // Skip this test if less than 4 threads are available.
  if (llvm::hardware_concurrency().compute_thread_count() < 4)
    GTEST_SKIP();

  using namespace llvm::sys;
  if (getenv("LLVM_THREADPOOL_AFFINITYMASK")) {
    auto ThreadsUsed = this->RunOnAllSockets({});
    // Ensure the threads only ran on CPUs 0-3.
    // NOTE: Don't use ASSERT* here because this runs in a subprocess,
    // and will show up as un-executed in the parent.
    assert(llvm::all_of(ThreadsUsed,
                        [](auto &T) { return T.getData().front() < 16UL; }) &&
           "Threads ran on more CPUs than expected! The affinity mask does not "
           "seem to work.");
    GTEST_SKIP();
  }
  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &ThreadPoolTestStringArg1);
  StringRef argv[] = {Executable, "--gtest_filter=ThreadPoolTest.AffinityMask"};

  // Add environment variable to the environment of the child process.
  int Res = setenv("LLVM_THREADPOOL_AFFINITYMASK", "1", false);
  ASSERT_EQ(Res, 0);

  std::string Error;
  bool ExecutionFailed;
  BitVector Affinity;
  Affinity.resize(4);
  Affinity.set(0, 4); // Use CPUs 0,1,2,3.
  int Ret = sys::ExecuteAndWait(Executable, argv, {}, {}, 0, 0, &Error,
                                &ExecutionFailed, nullptr, &Affinity);
  ASSERT_EQ(0, Ret);
}

#endif // #ifdef _WIN32
#endif // #if LLVM_ENABLE_THREADS == 1
