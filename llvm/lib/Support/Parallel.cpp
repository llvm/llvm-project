//===- llvm/Support/Parallel.cpp - Parallel algorithms --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Parallel.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/ExponentialBackoff.h"
#include "llvm/Support/Jobserver.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Threading.h"

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

llvm::ThreadPoolStrategy llvm::parallel::strategy;

namespace llvm {
namespace parallel {
#if LLVM_ENABLE_THREADS

#ifdef _WIN32
static thread_local unsigned threadIndex = UINT_MAX;

unsigned getThreadIndex() { GET_THREAD_INDEX_IMPL; }
#else
thread_local unsigned threadIndex = UINT_MAX;
#endif

namespace detail {

namespace {

/// An abstract class that takes closures and runs them asynchronously.
class Executor {
public:
  virtual ~Executor() = default;
  virtual void add(std::function<void()> func) = 0;
  virtual void stop() = 0;
  virtual size_t getThreadCount() const = 0;

  static Executor *getDefaultExecutor();
};

/// An implementation of an Executor that runs closures on a thread pool
/// in filo order. This is the standard executor for non-jobserver mode.
class ThreadPoolExecutor : public Executor {
public:
  explicit ThreadPoolExecutor(ThreadPoolStrategy S) {
    ThreadCount = S.compute_thread_count();
    // Spawn all but one of the threads in another thread as spawning threads
    // can take a while.
    Threads.reserve(ThreadCount);
    Threads.resize(1);
    std::lock_guard<std::mutex> Lock(Mutex);
    // Use operator[] before creating the thread to avoid data race in .size()
    // in 'safe libc++' mode.
    auto &Thread0 = Threads[0];
    Thread0 = std::thread([this, S] {
      for (unsigned I = 1; I < ThreadCount; ++I) {
        Threads.emplace_back([this, S, I] { work(S, I); });
        if (Stop)
          break;
      }
      ThreadsCreated.set_value();
      work(S, 0);
    });
  }

  // To make sure the thread pool executor can only be created with a parallel
  // strategy.
  ThreadPoolExecutor() = delete;

  void stop() override {
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      if (Stop)
        return;
      Stop = true;
    }
    Cond.notify_all();
    ThreadsCreated.get_future().wait();

    std::thread::id CurrentThreadId = std::this_thread::get_id();
    for (std::thread &T : Threads)
      if (T.get_id() == CurrentThreadId)
        T.detach();
      else
        T.join();
  }

  ~ThreadPoolExecutor() override { stop(); }

  struct Creator {
    static void *call();
  };
  struct Deleter {
    static void call(void *Ptr) { ((Executor *)Ptr)->stop(); }
  };

  void add(std::function<void()> F) override {
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      WorkStack.push_back(std::move(F));
    }
    Cond.notify_one();
  }

  size_t getThreadCount() const override { return ThreadCount; }

private:
  void work(ThreadPoolStrategy S, unsigned ThreadID) {
    threadIndex = ThreadID;
    S.apply_thread_strategy(ThreadID);

    while (true) {
      std::unique_lock<std::mutex> Lock(Mutex);
      Cond.wait(Lock, [&] { return Stop || !WorkStack.empty(); });
      if (Stop)
        break;
      auto Task = std::move(WorkStack.back());
      WorkStack.pop_back();
      Lock.unlock();
      Task();
    }
  }

  std::atomic<bool> Stop{false};
  std::vector<std::function<void()>> WorkStack;
  std::mutex Mutex;
  std::condition_variable Cond;
  std::promise<void> ThreadsCreated;
  std::vector<std::thread> Threads;
  unsigned ThreadCount;
};

/// Jobserver-aware executor that uses spawn-per-token thread management.
///
/// Problem (GitHub Issue #170184):
/// The original JobserverClient::getNumJobs() tried to determine the total
/// job count by draining all available tokens at startup. This is unreliable
/// because sibling processes may already hold tokens, causing getNumJobs() to
/// return a value lower than the actual -jN limit. If we then spawn that many
/// threads (like the standard ThreadPoolExecutor does), we underutilize the
/// available parallelism.
///
/// A naive fix (parsing -jN from MAKEFLAGS) causes a different problem: if
/// make -j16 spawns 16 LLVM processes, and each creates 16 threads, we get
/// 256 threads competing for 16 tokens - massive oversubscription.
///
/// Solution (spawn-per-token):
/// Instead of pre-spawning threads based on an unreliable job count, we:
/// 1. Start a single coordinator thread that waits for work
/// 2. When work arrives, the coordinator acquires a jobserver token
/// 3. Only after acquiring a token do we spawn an ephemeral worker thread
/// 4. The worker executes one task, then releases the token and exits
///
/// Benefits:
/// - Thread count automatically matches tokens actually held (not -jN guess)
/// - No idle threads blocking on token acquisition
/// - Multiple LLVM processes naturally share tokens without oversubscription
/// - Memory overhead (VmSize) scales with actual parallelism, not -jN
///
/// The thread index pool ensures getThreadIndex() returns values in
/// [0, getThreadCount()) for compatibility with PerThreadBumpPtrAllocator.
class JobserverThreadPoolExecutor : public Executor {
public:
  explicit JobserverThreadPoolExecutor(ThreadPoolStrategy S) : Strategy(S) {
    TheJobserver = JobserverClient::getInstance();
    assert(TheJobserver && "JobserverThreadPoolExecutor requires jobserver");

    // Size the thread-index pool independently of jobserver token count.
    ThreadPoolStrategy NonJobserver = Strategy;
    NonJobserver.UseJobserver = false;
    ThreadCount = NonJobserver.compute_thread_count();
    initThreadIndexPool();

    // Start a single coordinator thread that manages token-based spawning.
    Threads.reserve(1);
    Threads.resize(1);
    std::lock_guard<std::mutex> Lock(Mutex);
    auto &Thread0 = Threads[0];
    Thread0 = std::thread([this] { jobserverCoordinator(); });
    ThreadsCreated.set_value();
  }

  JobserverThreadPoolExecutor() = delete;

  void stop() override {
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      if (Stop)
        return;
      Stop = true;
    }
    Cond.notify_all();
    IndexCond.notify_all();
    ThreadsCreated.get_future().wait();

    std::thread::id CurrentThreadId = std::this_thread::get_id();
    for (std::thread &T : Threads)
      if (T.get_id() == CurrentThreadId)
        T.detach();
      else
        T.join();

    // Join any remaining ephemeral workers.
    joinFinishedWorkers(/*JoinAll=*/true);
  }

  ~JobserverThreadPoolExecutor() override { stop(); }

  void add(std::function<void()> F) override {
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      WorkStack.push_back(std::move(F));
    }
    Cond.notify_one();
  }

  size_t getThreadCount() const override { return ThreadCount; }

private:
  struct EphemeralWorkerRecord {
    std::thread Thread;
    std::shared_ptr<std::atomic<bool>> Done;
  };

  /// Coordinator thread: waits for work, acquires tokens, spawns workers.
  void jobserverCoordinator() {
    threadIndex = 0;
    Strategy.apply_thread_strategy(0);

    while (true) {
      // Wait until there's work or we're stopping.
      {
        std::unique_lock<std::mutex> Lock(Mutex);
        Cond.wait(Lock, [&] { return Stop || !WorkStack.empty(); });
        if (Stop)
          break;
      }

      // Periodically join finished workers to avoid unbounded growth.
      joinFinishedWorkers(/*JoinAll=*/false);

      // Acquire a thread index before attempting to acquire a token.
      std::optional<unsigned> ThreadIndex = acquireThreadIndex();
      if (!ThreadIndex)
        return;
      unsigned Index = *ThreadIndex;
      auto ReleaseIndex = llvm::scope_exit([&] { releaseThreadIndex(Index); });

      // Try to acquire a token. If successful, spawn an ephemeral worker.
      JobSlot Slot = TheJobserver->tryAcquire();
      if (!Slot.isValid()) {
        // No token available right now. Use backoff and retry.
        ExponentialBackoff Backoff(std::chrono::seconds(1));
        do {
          if (Stop)
            return;
          Slot = TheJobserver->tryAcquire();
          if (Slot.isValid())
            break;
        } while (Backoff.waitForNextAttempt());
      }

      if (!Slot.isValid())
        continue; // ReleaseIndex will run, loop back and check Stop/WorkStack.

      ReleaseIndex.release();

      // Spawn an ephemeral worker that owns this token.
      auto Done = std::make_shared<std::atomic<bool>>(false);
      std::thread Worker([this, Slot = std::move(Slot), Index, Done]() mutable {
        ephemeralWorker(std::move(Slot), Index, Done);
      });

      {
        std::lock_guard<std::mutex> Lock(EphemeralMutex);
        EphemeralWorkers.push_back({std::move(Worker), Done});
      }
    }
  }

  /// Ephemeral worker: executes one task while holding a token, then exits.
  /// This ensures tokens are released promptly for load balancing across
  /// processes. The coordinator will spawn another worker if more work exists.
  void ephemeralWorker(JobSlot Slot, unsigned ThreadIdx,
                       std::shared_ptr<std::atomic<bool>> Done) {
    threadIndex = ThreadIdx;
    Strategy.apply_thread_strategy(ThreadIdx);

    auto ReleaseIndex = llvm::scope_exit([&] {
      releaseThreadIndex(ThreadIdx);
      Done->store(true, std::memory_order_release);
    });

    llvm::scope_exit SlotReleaser(
        [&] { TheJobserver->release(std::move(Slot)); });

    // Execute exactly one task, then release the token.
    // This follows jobserver semantics: acquire token -> do one unit of work
    // -> release token. Holding tokens longer would prevent other processes
    // from obtaining them and break load balancing.
    std::function<void()> Task;
    {
      std::unique_lock<std::mutex> Lock(Mutex);
      if (WorkStack.empty() || Stop)
        return;
      Task = std::move(WorkStack.back());
      WorkStack.pop_back();
    }
    Task();
  }

  /// Join finished ephemeral workers to reclaim resources.
  void joinFinishedWorkers(bool JoinAll) {
    std::vector<std::thread> ToJoin;
    std::thread::id Current = std::this_thread::get_id();

    {
      std::lock_guard<std::mutex> Lock(EphemeralMutex);
      for (auto It = EphemeralWorkers.begin(); It != EphemeralWorkers.end();) {
        bool Finished = JoinAll || It->Done->load(std::memory_order_acquire);
        if (!Finished) {
          ++It;
          continue;
        }

        if (It->Thread.get_id() == Current) {
          // Avoid self-join in shutdown paths.
          It->Thread.detach();
          It = EphemeralWorkers.erase(It);
          continue;
        }

        ToJoin.push_back(std::move(It->Thread));
        It = EphemeralWorkers.erase(It);
      }
    }

    for (auto &T : ToJoin)
      if (T.joinable())
        T.join();
  }

  void initThreadIndexPool() {
    if (ThreadCount == 0)
      ThreadCount = 1;
    std::lock_guard<std::mutex> Lock(IndexMutex);
    FreeThreadIndices.clear();
    FreeThreadIndices.reserve(ThreadCount);
    for (unsigned I = 0; I < ThreadCount; ++I)
      FreeThreadIndices.push_back(I);
  }

  std::optional<unsigned> acquireThreadIndex() {
    std::unique_lock<std::mutex> Lock(IndexMutex);
    IndexCond.wait(Lock,
                   [&] { return Stop.load() || !FreeThreadIndices.empty(); });
    if (Stop)
      return std::nullopt;
    unsigned Index = FreeThreadIndices.back();
    FreeThreadIndices.pop_back();
    return Index;
  }

  void releaseThreadIndex(unsigned Index) {
    {
      std::lock_guard<std::mutex> Lock(IndexMutex);
      FreeThreadIndices.push_back(Index);
    }
    IndexCond.notify_one();
  }

  ThreadPoolStrategy Strategy;
  std::atomic<bool> Stop{false};
  std::vector<std::function<void()>> WorkStack;
  std::mutex Mutex;
  std::condition_variable Cond;
  std::promise<void> ThreadsCreated;
  std::vector<std::thread> Threads;
  unsigned ThreadCount;

  JobserverClient *TheJobserver = nullptr;

  // Ephemeral workers and their mutex.
  std::mutex EphemeralMutex;
  std::vector<EphemeralWorkerRecord> EphemeralWorkers;

  // Thread index pool for getThreadIndex() semantics.
  std::mutex IndexMutex;
  std::condition_variable IndexCond;
  std::vector<unsigned> FreeThreadIndices;
};

// Factory function to create the appropriate executor.
void *ThreadPoolExecutor::Creator::call() {
  if (strategy.UseJobserver && JobserverClient::getInstance())
    return new JobserverThreadPoolExecutor(strategy);
  return new ThreadPoolExecutor(strategy);
}

Executor *Executor::getDefaultExecutor() {
#ifdef _WIN32
  // The ManagedStatic enables the executor to be stopped via llvm_shutdown()
  // on Windows. This is important to avoid various race conditions at process
  // exit that can cause crashes or deadlocks.
  static ManagedStatic<Executor, ThreadPoolExecutor::Creator,
                       ThreadPoolExecutor::Deleter>
      ManagedExec;
  static std::unique_ptr<Executor> Exec(&(*ManagedExec));
  return Exec.get();
#else
  // ManagedStatic is not desired on other platforms. When the executor is
  // destroyed by llvm_shutdown(), worker threads will clean up and invoke TLS
  // destructors. This can lead to race conditions if other threads attempt to
  // access TLS objects that have already been destroyed.
  static std::unique_ptr<Executor> Exec(
      static_cast<Executor *>(ThreadPoolExecutor::Creator::call()));
  return Exec.get();
#endif
}
} // namespace
} // namespace detail

size_t getThreadCount() {
  return detail::Executor::getDefaultExecutor()->getThreadCount();
}
#endif

// Latch::sync() called by the dtor may cause one thread to block. If is a dead
// lock if all threads in the default executor are blocked. To prevent the dead
// lock, only allow the root TaskGroup to run tasks parallelly. In the scenario
// of nested parallel_for_each(), only the outermost one runs parallelly.
TaskGroup::TaskGroup()
#if LLVM_ENABLE_THREADS
    : Parallel((parallel::strategy.ThreadsRequested != 1) &&
               (threadIndex == UINT_MAX)) {}
#else
    : Parallel(false) {}
#endif
TaskGroup::~TaskGroup() {
  // We must ensure that all the workloads have finished before decrementing the
  // instances count.
  L.sync();
}

void TaskGroup::spawn(std::function<void()> F) {
#if LLVM_ENABLE_THREADS
  if (Parallel) {
    L.inc();
    detail::Executor::getDefaultExecutor()->add([&, F = std::move(F)] {
      F();
      L.dec();
    });
    return;
  }
#endif
  F();
}

} // namespace parallel
} // namespace llvm

void llvm::parallelFor(size_t Begin, size_t End,
                       llvm::function_ref<void(size_t)> Fn) {
#if LLVM_ENABLE_THREADS
  if (parallel::strategy.ThreadsRequested != 1) {
    auto NumItems = End - Begin;
    // Limit the number of tasks to MaxTasksPerGroup to limit job scheduling
    // overhead on large inputs.
    auto TaskSize = NumItems / parallel::detail::MaxTasksPerGroup;
    if (TaskSize == 0)
      TaskSize = 1;

    parallel::TaskGroup TG;
    for (; Begin + TaskSize < End; Begin += TaskSize) {
      TG.spawn([=, &Fn] {
        for (size_t I = Begin, E = Begin + TaskSize; I != E; ++I)
          Fn(I);
      });
    }
    if (Begin != End) {
      TG.spawn([=, &Fn] {
        for (size_t I = Begin; I != End; ++I)
          Fn(I);
      });
    }
    return;
  }
#endif

  for (; Begin != End; ++Begin)
    Fn(Begin);
}
