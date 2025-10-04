//===- llvm/unittest/Support/JobserverTest.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Jobserver.h unit tests.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/Jobserver.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <future>
#include <random>
#include <stdlib.h>

#if defined(LLVM_ON_UNIX)
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include <atomic>
#include <condition_variable>
#include <fcntl.h>
#include <mutex>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#define DEBUG_TYPE "jobserver-test"

using namespace llvm;

namespace {

// RAII helper to set an environment variable for the duration of a test.
class ScopedEnvironment {
  std::string Name;
  std::string OldValue;
  bool HadOldValue;

public:
  ScopedEnvironment(const char *Name, const char *Value) : Name(Name) {
#if defined(_WIN32)
    char *Old = nullptr;
    size_t OldLen;
    errno_t err = _dupenv_s(&Old, &OldLen, Name);
    if (err == 0 && Old != nullptr) {
      HadOldValue = true;
      OldValue = Old;
      free(Old);
    } else {
      HadOldValue = false;
    }
    _putenv_s(Name, Value);
#else
    const char *Old = getenv(Name);
    if (Old) {
      HadOldValue = true;
      OldValue = Old;
    } else {
      HadOldValue = false;
    }
    setenv(Name, Value, 1);
#endif
  }

  ~ScopedEnvironment() {
#if defined(_WIN32)
    if (HadOldValue)
      _putenv_s(Name.c_str(), OldValue.c_str());
    else
      // On Windows, setting an environment variable to an empty string
      // unsets it, making getenv() return NULL.
      _putenv_s(Name.c_str(), "");
#else
    if (HadOldValue)
      setenv(Name.c_str(), OldValue.c_str(), 1);
    else
      unsetenv(Name.c_str());
#endif
  }
};

TEST(Jobserver, Slot) {
  // Default constructor creates an invalid slot.
  JobSlot S1;
  EXPECT_FALSE(S1.isValid());
  EXPECT_FALSE(S1.isImplicit());

  // Create an implicit slot.
  JobSlot S2 = JobSlot::createImplicit();
  EXPECT_TRUE(S2.isValid());
  EXPECT_TRUE(S2.isImplicit());

  // Create an explicit slot.
  JobSlot S3 = JobSlot::createExplicit(42);
  EXPECT_TRUE(S3.isValid());
  EXPECT_FALSE(S3.isImplicit());

  // Test move construction.
  JobSlot S4 = std::move(S2);
  EXPECT_TRUE(S4.isValid());
  EXPECT_TRUE(S4.isImplicit());
  EXPECT_FALSE(S2.isValid()); // S2 is now invalid.

  // Test move assignment.
  S1 = std::move(S3);
  EXPECT_TRUE(S1.isValid());
  EXPECT_FALSE(S1.isImplicit());
  EXPECT_FALSE(S3.isValid()); // S3 is now invalid.
}

// Test fixture for parsing tests to ensure the singleton state is
// reset between each test case.
class JobserverParsingTest : public ::testing::Test {
protected:
  void TearDown() override { JobserverClient::resetForTesting(); }
};

TEST_F(JobserverParsingTest, NoMakeflags) {
  // No MAKEFLAGS, should be null.
  ScopedEnvironment Env("MAKEFLAGS", "");
  // On Unix, setting an env var to "" makes getenv() return an empty
  // string, not NULL. We must call unsetenv() to test the case where
  // the variable is truly not present.
#if !defined(_WIN32)
  unsetenv("MAKEFLAGS");
#endif
  EXPECT_EQ(JobserverClient::getInstance(), nullptr);
}

TEST_F(JobserverParsingTest, EmptyMakeflags) {
  // Empty MAKEFLAGS, should be null.
  ScopedEnvironment Env("MAKEFLAGS", "");
  EXPECT_EQ(JobserverClient::getInstance(), nullptr);
}

TEST_F(JobserverParsingTest, DryRunFlag) {
  // Dry-run flag 'n', should be null.
  ScopedEnvironment Env("MAKEFLAGS", "n -j --jobserver-auth=fifo:/tmp/foo");
  EXPECT_EQ(JobserverClient::getInstance(), nullptr);
}

// Separate fixture for non-threaded client tests.
class JobserverClientTest : public JobserverParsingTest {};

#if defined(LLVM_ON_UNIX)
// RAII helper to create and clean up a temporary FIFO file.
class ScopedFifo {
  SmallString<128> Path;
  bool IsValid = false;

public:
  ScopedFifo() {
    // To get a unique, non-colliding name for a FIFO, we use the
    // createTemporaryFile function to reserve a name in the filesystem.
    std::error_code EC =
        sys::fs::createTemporaryFile("jobserver-test", "fifo", Path);
    if (EC)
      return;
    // Then we immediately remove the regular file it created, but keep the
    // unique path.
    sys::fs::remove(Path);
    // Finally, we create the FIFO at that safe, unique path.
    if (mkfifo(Path.c_str(), 0600) != 0)
      return;
    IsValid = true;
  }

  ~ScopedFifo() {
    if (IsValid)
      sys::fs::remove(Path);
  }

  const char *c_str() const { return Path.data(); }
  bool isValid() const { return IsValid; }
};

TEST_F(JobserverClientTest, UnixClientFifo) {
  // This test covers basic FIFO client creation and behavior with an empty
  // FIFO. No job tokens are available.
  ScopedFifo F;
  ASSERT_TRUE(F.isValid());

  // Intentionally inserted \t in environment string.
  std::string Makeflags = " \t -j4\t \t--jobserver-auth=fifo:";
  Makeflags += F.c_str();
  ScopedEnvironment Env("MAKEFLAGS", Makeflags.c_str());

  JobserverClient *Client = JobserverClient::getInstance();
  ASSERT_NE(Client, nullptr);

  // Get the implicit token.
  JobSlot S1 = Client->tryAcquire();
  EXPECT_TRUE(S1.isValid());
  EXPECT_TRUE(S1.isImplicit());

  // FIFO is empty, next acquire fails.
  JobSlot S2 = Client->tryAcquire();
  EXPECT_FALSE(S2.isValid());

  // Release does not write to the pipe for the implicit token.
  Client->release(std::move(S1));

  // Re-acquire the implicit token.
  S1 = Client->tryAcquire();
  EXPECT_TRUE(S1.isValid());
}

#if LLVM_ENABLE_THREADS
// Test fixture for tests that use the jobserver strategy. It creates a
// temporary FIFO, sets MAKEFLAGS, and provides a helper to pre-load the FIFO
// with job tokens, simulating `make -jN`.
class JobserverStrategyTest : public JobserverParsingTest {
protected:
  std::unique_ptr<ScopedFifo> TheFifo;
  std::thread MakeThread;
  std::atomic<bool> StopMakeThread{false};
  // Save and restore the global parallel strategy to avoid interfering with
  // other tests in the same process.
  ThreadPoolStrategy SavedStrategy;

  void SetUp() override {
    SavedStrategy = parallel::strategy;
    TheFifo = std::make_unique<ScopedFifo>();
    ASSERT_TRUE(TheFifo->isValid());

    std::string MakeFlags = "--jobserver-auth=fifo:";
    MakeFlags += TheFifo->c_str();
    setenv("MAKEFLAGS", MakeFlags.c_str(), 1);
  }

  void TearDown() override {
    if (MakeThread.joinable()) {
      StopMakeThread = true;
      MakeThread.join();
    }
    unsetenv("MAKEFLAGS");
    TheFifo.reset();
    // Restore the original strategy to ensure subsequent tests are unaffected.
    parallel::strategy = SavedStrategy;
  }

  // Starts a background thread that emulates `make`. It populates the FIFO
  // with initial tokens and then recycles tokens released by clients.
  void startMakeProxy(int NumInitialJobs) {
    MakeThread = std::thread([this, NumInitialJobs]() {
      LLVM_DEBUG(dbgs() << "[MakeProxy] Thread started.\n");
      // Open the FIFO for reading and writing. This call does not block.
      int RWFd = open(TheFifo->c_str(), O_RDWR);
      LLVM_DEBUG(dbgs() << "[MakeProxy] Opened FIFO " << TheFifo->c_str()
                        << " with O_RDWR, FD=" << RWFd << "\n");
      if (RWFd == -1) {
        LLVM_DEBUG(
            dbgs()
            << "[MakeProxy] ERROR: Failed to open FIFO with O_RDWR. Errno: "
            << errno << "\n");
        return;
      }

      // Populate with initial jobs.
      LLVM_DEBUG(dbgs() << "[MakeProxy] Writing " << NumInitialJobs
                        << " initial tokens.\n");
      for (int i = 0; i < NumInitialJobs; ++i) {
        if (write(RWFd, "+", 1) != 1) {
          LLVM_DEBUG(dbgs()
                     << "[MakeProxy] ERROR: Failed to write initial token " << i
                     << ".\n");
          close(RWFd);
          return;
        }
      }
      LLVM_DEBUG(dbgs() << "[MakeProxy] Finished writing initial tokens.\n");

      // Make the read non-blocking so we can periodically check StopMakeThread.
      int flags = fcntl(RWFd, F_GETFL, 0);
      fcntl(RWFd, F_SETFL, flags | O_NONBLOCK);

      while (!StopMakeThread) {
        char Token;
        ssize_t Ret = read(RWFd, &Token, 1);
        if (Ret == 1) {
          LLVM_DEBUG(dbgs() << "[MakeProxy] Read token '" << Token
                            << "' to recycle.\n");
          // A client released a token, 'make' makes it available again.
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          ssize_t WRet;
          do {
            WRet = write(RWFd, &Token, 1);
          } while (WRet < 0 && errno == EINTR);
          if (WRet <= 0) {
            LLVM_DEBUG(
                dbgs()
                << "[MakeProxy] ERROR: Failed to write recycled token.\n");
            break; // Error, stop the proxy.
          }
          LLVM_DEBUG(dbgs()
                     << "[MakeProxy] Wrote token '" << Token << "' back.\n");
        } else if (Ret < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
          LLVM_DEBUG(dbgs() << "[MakeProxy] ERROR: Read failed with errno "
                            << errno << ".\n");
          break; // Error, stop the proxy.
        }
        // Yield to prevent this thread from busy-waiting.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      LLVM_DEBUG(dbgs() << "[MakeProxy] Thread stopping.\n");
      close(RWFd);
    });

    // Give the proxy thread a moment to start and populate the FIFO.
    // This is a simple way to avoid a race condition where the client starts
    // before the initial tokens are in the pipe.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
};

TEST_F(JobserverStrategyTest, ThreadPoolConcurrencyIsLimited) {
  // This test simulates `make -j3`. We will have 1 implicit job slot and
  // we will add 2 explicit job tokens to the FIFO, for a total of 3.
  const int NumExplicitJobs = 2;
  const int ConcurrencyLimit = NumExplicitJobs + 1; // +1 for the implicit slot
  const int NumTasks = 8; // More tasks than available slots.

  LLVM_DEBUG(dbgs() << "Calling startMakeProxy with " << NumExplicitJobs
                    << " jobs.\n");
  startMakeProxy(NumExplicitJobs);
  LLVM_DEBUG(dbgs() << "MakeProxy is running.\n");

  // Create the thread pool. Its constructor will call jobserver_concurrency()
  // and create a client that reads from our pre-loaded FIFO.
  StdThreadPool Pool(jobserver_concurrency());

  std::atomic<int> ActiveTasks{0};
  std::atomic<int> MaxActiveTasks{0};
  std::atomic<int> CompletedTasks{0};
  std::mutex M;
  std::condition_variable CV;

  // Dispatch more tasks than there are job slots. The pool should block
  // and only run up to `ConcurrencyLimit` tasks at once.
  for (int i = 0; i < NumTasks; ++i) {
    Pool.async([&, i] {
      // Track the number of concurrently running tasks.
      int CurrentActive = ++ActiveTasks;
      LLVM_DEBUG(dbgs() << "Task " << i << ": Active tasks: " << CurrentActive
                        << "\n");
      int OldMax = MaxActiveTasks.load();
      while (CurrentActive > OldMax)
        MaxActiveTasks.compare_exchange_weak(OldMax, CurrentActive);

      std::this_thread::sleep_for(std::chrono::milliseconds(25));

      --ActiveTasks;
      if (++CompletedTasks == NumTasks) {
        std::lock_guard<std::mutex> Lock(M);
        CV.notify_one();
      }
    });
  }

  // Wait for all tasks to complete.
  std::unique_lock<std::mutex> Lock(M);
  CV.wait(Lock, [&] { return CompletedTasks == NumTasks; });

  LLVM_DEBUG(dbgs() << "Test finished. Max active tasks was " << MaxActiveTasks
                    << ".\n");
  // The key assertion: the maximum number of concurrent tasks should
  // not have exceeded the limit imposed by the jobserver.
  EXPECT_LE(MaxActiveTasks, ConcurrencyLimit);
  EXPECT_EQ(CompletedTasks, NumTasks);
}

TEST_F(JobserverStrategyTest, ParallelForIsLimited) {
  // This test verifies that llvm::parallelFor respects the jobserver limit.
  const int NumExplicitJobs = 3;
  const int ConcurrencyLimit = NumExplicitJobs + 1; // +1 implicit
  const int NumTasks = 20;

  LLVM_DEBUG(dbgs() << "Calling startMakeProxy with " << NumExplicitJobs
                    << " jobs.\n");
  startMakeProxy(NumExplicitJobs);
  LLVM_DEBUG(dbgs() << "MakeProxy is running.\n");

  // Set the global strategy. parallelFor will use this.
  parallel::strategy = jobserver_concurrency();

  std::atomic<int> ActiveTasks{0};
  std::atomic<int> MaxActiveTasks{0};

  parallelFor(0, NumTasks, [&](int i) {
    int CurrentActive = ++ActiveTasks;
    LLVM_DEBUG(dbgs() << "Task " << i << ": Active tasks: " << CurrentActive
                      << "\n");
    int OldMax = MaxActiveTasks.load();
    while (CurrentActive > OldMax)
      MaxActiveTasks.compare_exchange_weak(OldMax, CurrentActive);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    --ActiveTasks;
  });

  LLVM_DEBUG(dbgs() << "ParallelFor finished. Max active tasks was "
                    << MaxActiveTasks << ".\n");
  EXPECT_LE(MaxActiveTasks, ConcurrencyLimit);
}

TEST_F(JobserverStrategyTest, ParallelSortIsLimited) {
  // This test serves as an integration test to ensure parallelSort completes
  // correctly when running under the jobserver strategy. It doesn't directly
  // measure concurrency but verifies correctness.
  const int NumExplicitJobs = 3;
  startMakeProxy(NumExplicitJobs);

  parallel::strategy = jobserver_concurrency();

  std::vector<int> V(1024);
  // Fill with random data
  std::mt19937 randEngine;
  std::uniform_int_distribution<int> dist;
  for (int &i : V)
    i = dist(randEngine);

  parallelSort(V.begin(), V.end());
  ASSERT_TRUE(llvm::is_sorted(V));
}

#endif // LLVM_ENABLE_THREADS

#endif // defined(LLVM_ON_UNIX)

} // end anonymous namespace
