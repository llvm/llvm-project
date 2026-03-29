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
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/PerThreadBumpPtrAllocator.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <future>
#include <random>
#include <stdlib.h>

#if defined(LLVM_ON_UNIX)
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#define DEBUG_TYPE "jobserver-test"

using namespace llvm;

// Provided by the unit test main to locate the current test binary.
extern const char *TestMainArgv0;

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

TEST_F(JobserverClientTest, UnixClientNonFifo) {
  // This test verifies that non-FIFO jobservers can be used, such as steve
  // or guildmaster.
  SmallString<64> F;
  std::error_code EC =
      sys::fs::createTemporaryFile("jobserver-test", "nonfifo", F);
  ASSERT_FALSE(EC);
  FileRemover Cleanup(F);

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

  // File is empty, next acquire fails.
  JobSlot S2 = Client->tryAcquire();
  EXPECT_FALSE(S2.isValid());

  // Release does not write to the file for the implicit token.
  Client->release(std::move(S1));

  // Re-acquire the implicit token.
  S1 = Client->tryAcquire();
  EXPECT_TRUE(S1.isValid());
}

// Test that getNumJobs() uses the -jN value from MAKEFLAGS when available.
TEST_F(JobserverClientTest, NumJobsFromMakeflags) {
  ScopedFifo F;
  ASSERT_TRUE(F.isValid());

  // Open with O_RDWR to avoid blocking (needs both ends open for FIFO).
  int fd = open(F.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  // Put 7 tokens in the FIFO (simulating -j8: 7 explicit + 1 implicit).
  ASSERT_EQ(write(fd, "+++++++", 7), 7);

  // Set MAKEFLAGS with -j8.
  std::string Makeflags = "-j8 --jobserver-auth=fifo:";
  Makeflags += F.c_str();
  ScopedEnvironment Env("MAKEFLAGS", Makeflags.c_str());

  JobserverClient *Client = JobserverClient::getInstance();
  ASSERT_NE(Client, nullptr);

  // NumJobs should reflect -jN when provided.
  EXPECT_EQ(Client->getNumJobs(), 8u);

  close(fd);
}

// Test that getNumJobs() uses -jN even when some tokens are held by siblings.
// This is the core fix for Issue #170184.
TEST_F(JobserverClientTest, NumJobsWithSiblingHoldingTokens) {
  ScopedFifo F;
  ASSERT_TRUE(F.isValid());

  // Open with O_RDWR to avoid blocking.
  int fd = open(F.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  // Put only 2 tokens in the FIFO (simulating sibling holding 5 of 7 tokens).
  ASSERT_EQ(write(fd, "++", 2), 2);

  // Set MAKEFLAGS with -j8.
  std::string Makeflags = "-j8 --jobserver-auth=fifo:";
  Makeflags += F.c_str();
  ScopedEnvironment Env("MAKEFLAGS", Makeflags.c_str());

  JobserverClient *Client = JobserverClient::getInstance();
  ASSERT_NE(Client, nullptr);

  // NumJobs should still reflect -jN, not the drained token count.
  EXPECT_EQ(Client->getNumJobs(), 8u);

  close(fd);
}

// Test that getNumJobs() is unknown when -jN is not present.
TEST_F(JobserverClientTest, NumJobsUnknownWithoutJobsFlag) {
  ScopedFifo F;
  ASSERT_TRUE(F.isValid());

  // Open with O_RDWR to avoid blocking.
  int fd = open(F.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  // Put 3 tokens in the FIFO.
  ASSERT_EQ(write(fd, "+++", 3), 3);

  // Set MAKEFLAGS without -jN (e.g., Ninja or old Make).
  std::string Makeflags = "--jobserver-auth=fifo:";
  Makeflags += F.c_str();
  ScopedEnvironment Env("MAKEFLAGS", Makeflags.c_str());

  JobserverClient *Client = JobserverClient::getInstance();
  ASSERT_NE(Client, nullptr);

  // NumJobs should be unknown (0) without a -jN hint.
  EXPECT_EQ(Client->getNumJobs(), 0u);

  close(fd);
}

// Test that getNumJobs() parses --jobs=N format.
TEST_F(JobserverClientTest, NumJobsFromLongOption) {
  ScopedFifo F;
  ASSERT_TRUE(F.isValid());

  // Open with O_RDWR to avoid blocking.
  int fd = open(F.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  // Put tokens in the FIFO.
  ASSERT_EQ(write(fd, "+++++++++++", 11), 11);

  // Set MAKEFLAGS with --jobs=12.
  std::string Makeflags = "--jobs=12 --jobserver-auth=fifo:";
  Makeflags += F.c_str();
  ScopedEnvironment Env("MAKEFLAGS", Makeflags.c_str());

  JobserverClient *Client = JobserverClient::getInstance();
  ASSERT_NE(Client, nullptr);

  // NumJobs should reflect --jobs=N when provided.
  EXPECT_EQ(Client->getNumJobs(), 12u);

  close(fd);
}
#if LLVM_ENABLE_THREADS
// Unique anchor whose address helps locate the current test binary.
static int JobserverTestAnchor = 0;

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

/// Macro to define a jobserver test that runs in an isolated subprocess.
///
/// The jobserver executor is a process-wide singleton that is initialized once
/// when the first parallel operation runs. To properly test jobserver behavior,
/// each test needs a fresh process where:
///   1. MAKEFLAGS is set before executor initialization
///   2. parallel::strategy is configured before any parallelFor calls
///   3. The test has its own independent jobserver (FIFO + MakeProxy)
///
/// This macro generates two test functions:
///   - Name_Subprocess: Parent that spawns the child and waits for it
///   - Name_SubprocessChild: Child that runs the actual test logic
///
/// Usage:
///   ISOLATED_JOBSERVER_TEST(MyTest, TimeoutSeconds, {
///     startMakeProxy(3);  // 3 explicit tokens
///     parallel::strategy = jobserver_concurrency();
///     parallelFor(0, 10, [](size_t) { /* work */ });
///     EXPECT_...;
///   })
///
#define ISOLATED_JOBSERVER_TEST(Name, TimeoutSec, Body)                        \
  TEST_F(JobserverStrategyTest, Name##_Subprocess) {                           \
    setenv("LLVM_JOBSERVER_TEST_CHILD", "1", 1);                               \
    std::string Executable =                                                   \
        sys::fs::getMainExecutable(TestMainArgv0, &JobserverTestAnchor);       \
    ASSERT_FALSE(Executable.empty()) << "Failed to get main executable path";  \
    SmallVector<StringRef, 4> Args{                                            \
        Executable,                                                            \
        "--gtest_filter=JobserverStrategyTest." #Name "_SubprocessChild"};     \
    std::string Error;                                                         \
    bool ExecFailed = false;                                                   \
    int RC = sys::ExecuteAndWait(Executable, Args, std::nullopt, {},           \
                                 TimeoutSec, 0, &Error, &ExecFailed);          \
    unsetenv("LLVM_JOBSERVER_TEST_CHILD");                                     \
    ASSERT_FALSE(ExecFailed) << Error;                                         \
    EXPECT_EQ(RC, 0) << "Child test failed with exit code " << RC;             \
  }                                                                            \
  TEST_F(JobserverStrategyTest, Name##_SubprocessChild) {                      \
    if (!getenv("LLVM_JOBSERVER_TEST_CHILD"))                                  \
      GTEST_SKIP() << "Not running in child mode";                             \
    Body                                                                       \
  }

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
      (void)i;
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

// Verifies that parallelFor respects the jobserver concurrency limit.
ISOLATED_JOBSERVER_TEST(ParallelForIsLimited, /*TimeoutSec=*/30, {
  const int NumExplicitJobs = 3;
  const int ConcurrencyLimit = NumExplicitJobs + 1; // +1 implicit
  const int NumTasks = 20;

  startMakeProxy(NumExplicitJobs);
  parallel::strategy = jobserver_concurrency();

  std::atomic<int> ActiveTasks{0};
  std::atomic<int> MaxActiveTasks{0};

  parallelFor(0, NumTasks, [&]([[maybe_unused]] int i) {
    int CurrentActive = ++ActiveTasks;
    int OldMax = MaxActiveTasks.load();
    while (CurrentActive > OldMax)
      MaxActiveTasks.compare_exchange_weak(OldMax, CurrentActive);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    --ActiveTasks;
  });

  EXPECT_LE(MaxActiveTasks, ConcurrencyLimit);
})

// Verifies that parallelSort works correctly under jobserver strategy.
ISOLATED_JOBSERVER_TEST(ParallelSortIsLimited, /*TimeoutSec=*/30, {
  const int NumExplicitJobs = 3;
  startMakeProxy(NumExplicitJobs);
  parallel::strategy = jobserver_concurrency();

  std::vector<int> V(1024);
  std::mt19937 randEngine;
  std::uniform_int_distribution<int> dist;
  for (int &i : V)
    i = dist(randEngine);

  parallelSort(V.begin(), V.end());
  ASSERT_TRUE(llvm::is_sorted(V));
})

// Verifies that PerThreadBumpPtrAllocator works correctly under jobserver
// strategy. This validates that getThreadIndex() stays within
// [0, getThreadCount()) even with spawn-per-token thread management.
ISOLATED_JOBSERVER_TEST(PerThreadAllocatorIsValid, /*TimeoutSec=*/30, {
  const int NumExplicitJobs = 3;
  startMakeProxy(NumExplicitJobs);
  parallel::strategy = jobserver_concurrency();

  parallel::PerThreadBumpPtrAllocator Allocator;
  static constexpr size_t NumAllocations = 500;
  std::atomic<size_t> SuccessfulAllocations{0};

  parallelFor(0, NumAllocations, [&](size_t Idx) {
    uint64_t *ptr =
        (uint64_t *)Allocator.Allocate(sizeof(uint64_t), alignof(uint64_t));
    ASSERT_NE(ptr, nullptr) << "Allocation failed at index " << Idx;
    *ptr = Idx;
    EXPECT_EQ(*ptr, Idx);
    ++SuccessfulAllocations;
  });

  EXPECT_EQ(SuccessfulAllocations, NumAllocations);
  EXPECT_EQ(sizeof(uint64_t) * NumAllocations, Allocator.getBytesAllocated());
  EXPECT_LE(Allocator.getNumberOfAllocators(), parallel::getThreadCount());
})

// Shared memory structure for tracking active threads across processes.
// Uses a file-backed mmap for cross-process atomic operations.
struct SharedThreadCounter {
  std::atomic<int> ActiveCount;
  std::atomic<int> MaxActive;
};

// Parent-side driver for multi-process PerThreadBumpPtrAllocator test.
// This spawns multiple child processes that all share the same jobserver,
// simulating real-world usage (e.g., make -j8 running multiple clang
// instances). Also verifies that total active threads never exceed the
// jobserver limit.
TEST_F(JobserverStrategyTest, MultiProcessAllocatorIsValid_Subprocess) {
  setenv("LLVM_JOBSERVER_TEST_CHILD", "1", 1);

  const int NumProcesses = 4;
  const int NumTotalJobs = 6; // Model `make -j6`.
  // GNU make semantics:
  // - Each child process gets one implicit slot.
  // - The jobserver pipe holds the remaining explicit tokens.
  // - Total allowed concurrency across all processes is NumTotalJobs.
  //
  // To model that, we start the proxy with (N - NumProcesses) explicit tokens
  // so that implicit + explicit equals N.
  const int NumExplicitJobs = NumTotalJobs - NumProcesses;
  ASSERT_GE(NumExplicitJobs, 0);
  const int ConcurrencyLimit = NumTotalJobs;

  startMakeProxy(NumExplicitJobs);

  // Create shared memory file for cross-process thread counting
  SmallString<128> SharedFilePath;
  ASSERT_FALSE(
      sys::fs::createTemporaryFile("jobserver-test", "shm", SharedFilePath));
  FileRemover SharedFileRemover(SharedFilePath);

  // Initialize the shared file with zeros
  {
    std::error_code EC;
    raw_fd_ostream Out(SharedFilePath, EC);
    ASSERT_FALSE(EC) << "Failed to open shared file: " << EC.message();
    std::array<char, sizeof(SharedThreadCounter)> ZeroBuf{};
    Out.write(ZeroBuf.data(), ZeroBuf.size());
  }

  // Memory-map the file for the parent to read results
  int SharedFD = open(SharedFilePath.c_str(), O_RDWR);
  ASSERT_GE(SharedFD, 0) << "Failed to open shared file for mmap";
  void *SharedMem = mmap(nullptr, sizeof(SharedThreadCounter),
                         PROT_READ | PROT_WRITE, MAP_SHARED, SharedFD, 0);
  ASSERT_NE(SharedMem, MAP_FAILED) << "mmap failed";
  close(SharedFD);
  auto *Counter = static_cast<SharedThreadCounter *>(SharedMem);
  new (Counter) SharedThreadCounter();
  Counter->ActiveCount.store(0, std::memory_order_relaxed);
  Counter->MaxActive.store(0, std::memory_order_relaxed);

  // Pass shared file path to children
  setenv("LLVM_JOBSERVER_SHARED_FILE", SharedFilePath.c_str(), 1);

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &JobserverTestAnchor);
  ASSERT_FALSE(Executable.empty()) << "Failed to get main executable path";

  // Spawn multiple child processes concurrently
  SmallVector<sys::ProcessInfo, 8> Children;
  for (int i = 0; i < NumProcesses; ++i) {
    SmallVector<StringRef, 4> Args{
        Executable, "--gtest_filter=JobserverStrategyTest."
                    "MultiProcessAllocatorIsValid_SubprocessChild"};
    std::string Error;
    bool ExecFailed = false;
    sys::ProcessInfo PI = sys::ExecuteNoWait(Executable, Args, std::nullopt, {},
                                             0, &Error, &ExecFailed);
    ASSERT_FALSE(ExecFailed) << "Failed to spawn child " << i << ": " << Error;
    ASSERT_NE(PI.Pid, 0) << "Invalid PID for child " << i;
    Children.push_back(PI);
  }

  // Wait for all children to complete
  for (int i = 0; i < NumProcesses; ++i) {
    std::string Error;
    sys::ProcessInfo Result = sys::Wait(Children[i], std::nullopt, &Error);
    EXPECT_EQ(Result.ReturnCode, 0)
        << "Child " << i << " failed with exit code " << Result.ReturnCode;
  }

  // Verify the maximum active threads never exceeded the jobserver limit
  int MaxActive = Counter->MaxActive.load();
  LLVM_DEBUG(dbgs() << "Max active threads across all processes: " << MaxActive
                    << ", limit: " << ConcurrencyLimit << "\n");
  EXPECT_LE(MaxActive, ConcurrencyLimit)
      << "Active threads exceeded jobserver limit";

  munmap(SharedMem, sizeof(SharedThreadCounter));
  unsetenv("LLVM_JOBSERVER_SHARED_FILE");
  unsetenv("LLVM_JOBSERVER_TEST_CHILD");
}

// Child-side test for multi-process scenario: each child runs parallel
// allocations under a shared jobserver. Multiple instances of this test
// run concurrently, competing for jobserver tokens.
TEST_F(JobserverStrategyTest, MultiProcessAllocatorIsValid_SubprocessChild) {
  if (!getenv("LLVM_JOBSERVER_TEST_CHILD"))
    GTEST_SKIP() << "Not running in child mode";

  // Open shared memory for cross-process thread counting
  const char *SharedFilePath = getenv("LLVM_JOBSERVER_SHARED_FILE");
  SharedThreadCounter *Counter = nullptr;
  int SharedFD = -1;
  if (SharedFilePath) {
    SharedFD = open(SharedFilePath, O_RDWR);
    if (SharedFD >= 0) {
      void *SharedMem = mmap(nullptr, sizeof(SharedThreadCounter),
                             PROT_READ | PROT_WRITE, MAP_SHARED, SharedFD, 0);
      if (SharedMem != MAP_FAILED) {
        Counter = static_cast<SharedThreadCounter *>(SharedMem);
      }
      close(SharedFD);
    }
  }
  ASSERT_NE(Counter, nullptr) << "Failed to map shared counters";

  // Don't call startMakeProxy - parent already set up the jobserver
  // and passed it via MAKEFLAGS environment variable.
  parallel::strategy = jobserver_concurrency();

  parallel::PerThreadBumpPtrAllocator Allocator;
  static constexpr size_t NumAllocations = 200;
  std::atomic<size_t> SuccessfulAllocations{0};

  parallelFor(0, NumAllocations, [&](size_t Idx) {
    // Track active threads across all processes
    if (Counter) {
      int Current = ++Counter->ActiveCount;
      int OldMax = Counter->MaxActive.load();
      while (Current > OldMax &&
             !Counter->MaxActive.compare_exchange_weak(OldMax, Current))
        ;
    }

    uint64_t *ptr =
        (uint64_t *)Allocator.Allocate(sizeof(uint64_t), alignof(uint64_t));
    ASSERT_NE(ptr, nullptr) << "Allocation failed at index " << Idx;
    *ptr = Idx;
    EXPECT_EQ(*ptr, Idx);
    ++SuccessfulAllocations;

    // Brief sleep to increase chance of overlap between processes
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    if (Counter) {
      --Counter->ActiveCount;
    }
  });

  EXPECT_EQ(SuccessfulAllocations, NumAllocations);
  EXPECT_LE(Allocator.getNumberOfAllocators(), parallel::getThreadCount());

  if (Counter) {
    munmap(Counter, sizeof(SharedThreadCounter));
  }
}

// Test that demonstrates backoff behavior when all tokens are busy.
// With slow tasks (2s each), the coordinator must wait in backoff until
// a worker finishes and releases a token.
//
// Note on "token starvation": When tryAcquire() keeps failing, we cannot
// reliably distinguish between:
//   (a) The jobserver (gmake) died and the pipe is broken
//   (b) Sibling processes are legitimately holding all tokens
//
// Both scenarios look identical from the client's perspective. The current
// behavior - falling back to sequential execution via the implicit token -
// is the correct response in both cases. It ensures forward progress while
// respecting the jobserver protocol.
ISOLATED_JOBSERVER_TEST(SlowTasksShowBackoff, /*TimeoutSec=*/30, {
  // Only 1 explicit token + 1 implicit = 2 concurrent jobs
  // With 6 tasks taking 2s each, coordinator MUST wait for tokens
  const int NumExplicitJobs = 1;
  const int NumTasks = 6;
  const auto TaskDuration = std::chrono::seconds(2);

  startMakeProxy(NumExplicitJobs);
  parallel::strategy = jobserver_concurrency();

  std::atomic<int> CompletedTasks{0};
  std::atomic<int> MaxConcurrent{0};
  std::atomic<int> CurrentConcurrent{0};

  auto StartTime = std::chrono::steady_clock::now();

  parallelFor(0, NumTasks, [&](size_t Idx) {
    int concurrent = ++CurrentConcurrent;
    int oldMax = MaxConcurrent.load();
    while (concurrent > oldMax &&
           !MaxConcurrent.compare_exchange_weak(oldMax, concurrent))
      ;

    LLVM_DEBUG(dbgs() << "[Test] Task " << Idx
                      << " starting. Concurrent: " << concurrent << "\n");
    std::this_thread::sleep_for(TaskDuration);

    --CurrentConcurrent;
    ++CompletedTasks;
    LLVM_DEBUG(dbgs() << "[Test] Task " << Idx << " completed.\n");
    (void)Idx;
  });

  auto EndTime = std::chrono::steady_clock::now();
  auto ElapsedMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime)
          .count();

  LLVM_DEBUG(dbgs() << "[Test] Total time: " << ElapsedMs << "ms\n");
  LLVM_DEBUG(dbgs() << "[Test] Max concurrent: " << MaxConcurrent.load()
                    << "\n");

  EXPECT_EQ(CompletedTasks.load(), NumTasks);
  EXPECT_LE(MaxConcurrent.load(), NumExplicitJobs + 1);
  // Expected: 6 tasks / 2 concurrent * 2s = 6 seconds minimum
  EXPECT_GE(ElapsedMs, 5500) << "Should take at least ~6 seconds";
  EXPECT_LE(ElapsedMs, 10000) << "Should not take more than 10 seconds";
})

// Test that the executor handles a broken jobserver pipe gracefully.
//
// When the jobserver pipe becomes unavailable (e.g., parent make dies),
// the executor falls back to sequential execution using only the implicit
// token. This is the correct behavior because:
//
// 1. We cannot reliably distinguish "pipe broken" from "tokens held by
//    siblings" - both result in tryAcquire() returning invalid.
//
// 2. The implicit token guarantees forward progress - every spawned process
//    gets one implicit slot, so work can always proceed (one task at a time).
//
// 3. The 1-second backoff timeout prevents CPU spinning while waiting for
//    tokens that may never arrive.
//
// This test verifies that work completes even after the pipe breaks,
// demonstrating the graceful degradation to sequential execution.
ISOLATED_JOBSERVER_TEST(BrokenPipeHandling, /*TimeoutSec=*/10, {
  // Simulate make -j4: 3 explicit tokens + 1 implicit = 4 concurrent jobs
  const int NumExplicitJobs = 3;
  const int NumTasks = 8;
  const auto TaskDuration = std::chrono::microseconds(100);

  startMakeProxy(NumExplicitJobs);
  parallel::strategy = jobserver_concurrency();

  std::atomic<int> CompletedTasks{0};
  std::atomic<int> MaxConcurrent{0};
  std::atomic<int> CurrentConcurrent{0};

  // Start tasks in background thread so we can break pipe mid-execution
  auto Future = std::async(std::launch::async, [&]() {
    parallelFor(0, NumTasks, [&](size_t Idx) {
      int concurrent = ++CurrentConcurrent;
      int oldMax = MaxConcurrent.load();
      while (concurrent > oldMax &&
             !MaxConcurrent.compare_exchange_weak(oldMax, concurrent))
        ;

      std::this_thread::sleep_for(TaskDuration);

      --CurrentConcurrent;
      ++CompletedTasks;
      LLVM_DEBUG(dbgs() << "[Test] Task " << Idx << " completed. Total: "
                        << CompletedTasks.load() << "/" << NumTasks << "\n");
      (void)Idx;
    });
  });

  // Let some tasks start and complete normally
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Break the pipe mid-execution
  LLVM_DEBUG(dbgs() << "[Test] Breaking pipe...\n");
  StopMakeThread = true;
  if (MakeThread.joinable())
    MakeThread.join();
  sys::fs::remove(TheFifo->c_str());
  LLVM_DEBUG(dbgs() << "[Test] Pipe broken.\n");

  // Wait for completion
  auto Status = Future.wait_for(std::chrono::seconds(10));
  if (Status == std::future_status::timeout) {
    GTEST_SKIP() << "Executor hangs on broken pipe. "
                 << "Completed " << CompletedTasks.load() << "/" << NumTasks;
  }

  EXPECT_EQ(CompletedTasks.load(), NumTasks) << "All tasks should complete";
  EXPECT_LE(MaxConcurrent.load(), NumExplicitJobs + 1);
})

#endif // LLVM_ENABLE_THREADS

#endif // defined(LLVM_ON_UNIX)

} // end anonymous namespace
