//===-- EJitSharedTaskPool.h - Cross-core shared single-worker facade -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  EJitSharedTaskPool drives a single EJitSharedTaskPoolState shared across
//  cores. It does NOT own the blob (the blob lives in shared memory / a test
//  fixture); it binds to it and provides:
//
//   * owner election: the first core to CAS Uninitialized->Initializing becomes
//     the worker owner, builds the shared state, optionally starts the ONE
//     worker, and publishes Ready (or Failed). Every other core observes the
//     outcome with an acquire load and binds — it never creates a second
//     worker.
//   * the producer path compileOrGet() operating purely on shared state.
//   * the consumer path pollOne()/pollBudget() (the worker, or a test, drives
//     it) with the two version checkpoints and the commit-gated cache publish.
//   * read-only diagnostics.
//
//  The compile callback, the physical-code release callback, and the worker
//  start/stop hooks are all OWNER-CORE-PRIVATE function pointers injected
//  before init(): they reach the owner's private EJit/ORC objects, which must
//  never be placed in shared memory.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITSHAREDTASKPOOL_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITSHAREDTASKPOOL_H

#include "llvm/ExecutionEngine/EJIT/EJitSharedTaskPoolState.h"
#include "llvm/ExecutionEngine/EJIT/EJitTaskPool.h" // EJitCompileMode, status enum
#include <cstdint>

namespace llvm {
namespace ejit {

//===----------------------------------------------------------------------===//
// Read-only diagnostics snapshot (spec §11 observability). Every field is a
// plain copy of an atomic load; no field exposes a raw shared pointer.
//===----------------------------------------------------------------------===//
struct EJitSharedDiagnostics {
  uint32_t initState;     ///< EJitSharedInitState
  uint32_t ownerCoreId;   ///< kEJitInvalidCoreId until elected
  uint32_t generation;    ///< bumps each (re)init
  uint32_t lastInitError; ///< error code recorded on Failed
  uint32_t initAttempts;  ///< total election attempts
  uint32_t codeSharingEnabled;
  uint64_t workerTaskId;
  uint64_t registrationFingerprint;
  uint32_t queueDepth;      ///< approximate in-ring requests
  uint32_t pendingCount;    ///< in-flight dedup slots
  uint32_t cacheReadyCount; ///< Ready cache slots
  uint64_t cacheHits;
  uint64_t asyncEnqueues;
  uint64_t asyncCompiles;
  uint64_t alreadyPending;
  uint64_t queueFull;
  uint64_t compileFailed;
  uint64_t publishFailed;
  uint64_t instanceDisabled;
  uint64_t executePrepareFailed;
};

//===----------------------------------------------------------------------===//
// One step of the worker state machine (spec §11 worker startup timing). The
// worker MUST NOT exit when it observes a not-yet-Ready owner (the SRE task may
// be scheduled before the owner publishes Ready); it waits instead.
//===----------------------------------------------------------------------===//
enum class EJitWorkerStep : uint32_t {
  WaitForReady =
      0,    ///< Owner still Initializing: wait, do NOT read queue/cache.
  Consumed, ///< Ready: dequeued and ran one compile.
  Idle,     ///< Ready: queue empty this iteration.
  Exit,     ///< Failed/Stopping/Uninitialized: leave the loop.
};

class EJitSharedTaskPool {
public:
  /// Owner-private compile callback (reaches the owner's EJit/ORC). Returns
  /// true and *outFn on success.
  using CompileCallback = bool (*)(void *ctx, const EJitCompileRequest &req,
                                   void **outFn);
  /// Owner-private physical-code release callback for an overwritten/retired
  /// pointer. Optional; a purely logical drop happens when unset.
  using ReleaseCallback = void (*)(void *ctx, void *oldFn);
  /// Install execute permission for \p fnPtr in the calling core's translation
  /// context. Required before a non-owner core may consume a shared fnPtr.
  using PrepareCodeCallback = bool (*)(void *ctx, const void *fnPtr);

  /// Worker loop entry (provided by this class, run on the injected task).
  using WorkerEntryFn = void (*)(void *ctx);
  /// Owner-injected worker starter: create ONE task running \p entry(\p
  /// entryCtx); store its id in *outTaskId; return false on failure.
  using WorkerStartFn = bool (*)(void *startCtx, WorkerEntryFn entry,
                                 void *entryCtx, uint64_t *outTaskId);
  /// Owner-injected worker stopper: soft-stop and JOIN the task. Must not
  /// return until the worker has exited (no use-after-free of owner-private
  /// state).
  using WorkerStopFn = void (*)(void *startCtx);
  /// Idle/yield hook the worker calls whenever it has no work to do (waiting on
  /// the owner to publish Ready, or Ready with an empty queue). The production
  /// build injects a platform yield (EJitSreTask::yield: SRE_TaskDelay on
  /// freestanding, std::this_thread::yield on host) so a high-priority worker
  /// never busy-spins and starves the core trying to publish Ready. MUST NOT be
  /// called while holding a bucket lock / queue slot / dedup critical state.
  using WorkerIdleFn = void (*)(void *ctx);

  enum class InitResult : uint32_t {
    BecameOwner =
        0,         ///< Won election; built state; worker started (if injected).
    AttachedReady, ///< Bound to an already-Ready shared state.
    OwnerFailed,   ///< State is Failed/Stopping: clean fallback, no wait.
    InitInProgress, ///< Another core still Initializing; bounded retry hit.
    AbiMismatch,    ///< magic/version/size mismatch — refuse to use the blob.
    FingerprintMismatch, ///< owner/peer registration mapping differs — clean
                         ///< fail.
    NoState,             ///< bind() not called.
  };

  struct CompileOrGetResult {
    EJitCompileOrGetStatus status = EJitCompileOrGetStatus::CompileFailed;
    void *fnPtr = nullptr;
    uint32_t bucketIndex = 0;
    bool hasReadToken = false;
    /// True when a Ready result exists but this core may not read the
    /// cross-core pointer (code sharing not platform-validated): a clean
    /// fallback that did NOT re-enqueue.
    bool readyButNotShareable = false;
  };

  EJitSharedTaskPool() = default;
  EJitSharedTaskPool(const EJitSharedTaskPool &) = delete;
  EJitSharedTaskPool &operator=(const EJitSharedTaskPool &) = delete;

  /// Bind to the shared blob (not owned). Call before init().
  void bind(EJitSharedTaskPoolState *state) { state_ = state; }
  EJitSharedTaskPoolState *state() const { return state_; }

  //--- owner-only configuration (applied if this core wins election) ----------
  void setCompiler(CompileCallback fn, void *ctx) {
    compileFn_ = fn;
    compileCtx_ = ctx;
  }
  void setReleaser(ReleaseCallback fn, void *ctx) {
    releaseFn_ = fn;
    releaseCtx_ = ctx;
  }
  void setPrepareCodeCallback(PrepareCodeCallback fn, void *ctx) {
    prepareCodeFn_ = fn;
    prepareCodeCtx_ = ctx;
  }
  void setWorkerHooks(WorkerStartFn start, WorkerStopFn stop, void *ctx) {
    workerStart_ = start;
    workerStop_ = stop;
    workerCtx_ = ctx;
  }
  /// Inject the worker idle/yield hook (see WorkerIdleFn). When unset the loop
  /// falls back to a compiler reordering barrier only (used by step tests).
  void setWorkerIdleHook(WorkerIdleFn fn, void *ctx) {
    workerIdle_ = fn;
    workerIdleCtx_ = ctx;
  }
  /// Owner publishes this digest of its funcIndex/dimType registration mapping
  /// into the shared state; a peer attaching to a Ready blob compares its own
  /// digest and cleanly fails (FingerprintMismatch) on any divergence, so a
  /// core with a different mapping never submits requests against the wrong
  /// indices (spec §11). 0 means "unknown / not checked".
  void setRegistrationFingerprint(uint64_t fp) { regFingerprint_ = fp; }
  /// Platform capability: may a NON-owner core read a cache fnPtr? Only true
  /// when the code pool is mapped at the same VA on every core, sealed, and
  /// I/D-cache coherent for cross-core execution (spec §11 fnPtr
  /// prerequisites).
  void setCodeSharingEnabled(bool enabled) { codeSharingEnabled_ = enabled; }
  void setMode(EJitCompileMode mode) { configuredMode_ = mode; }

  /// Run owner election + bind. Idempotent: re-observes the same outcome.
  InitResult init();
  bool isOwner() const { return isOwner_; }

  /// Owner-only orderly shutdown: stop+join the worker, then return the state
  /// to Uninitialized so a later init() can re-elect. No-op for a non-owner.
  void ownerShutdown();

  //--- producer path ----------------------------------------------------------
  CompileOrGetResult compileOrGet(uint32_t funcIndex, const EJitDimPair *dims,
                                  uint32_t numDims, void *fallback);
  void releaseRead(uint32_t bucketIndex);
  bool setInstanceEnabled(uint32_t dimType, uint32_t instanceId, bool enabled);

  //--- consumer path (worker / test) -----------------------------------------
  bool pollOne();
  unsigned pollBudget(unsigned maxItems);

  //--- diagnostics ------------------------------------------------------------
  void getDiagnostics(EJitSharedDiagnostics &out) const;
  uint32_t sharedInitState() const;
  /// In-flight dedup count (used by the taskpool C ABI pending_count).
  uint32_t pendingCount() const;

  /// The worker loop body: poll until the shared state leaves Ready. Public so
  /// an injected task entry can forward to it; normally reached via
  /// WorkerEntry.
  void runWorkerLoop();

  /// One step of the worker state machine. Public so a deterministic test can
  /// drive the REAL machine across controlled state transitions (no thread).
  /// Waits (never exits) while the owner is still Initializing.
  EJitWorkerStep workerPollOnce();

  //--- worker observability (owner-local; the worker runs on this instance) ---
  /// Ready iterations the worker loop executed (proves it reached the consume
  /// phase, not merely that start was called).
  uint64_t workerConsumeLoops() const {
    return workerConsumeLoops_.loadRelaxed();
  }
  /// True if the worker loop ever had to wait on a not-yet-Ready owner.
  bool workerWaitedForReady() const {
    return workerWaitedForReady_.loadRelaxed() != 0;
  }
  /// Number of times the worker yielded (idle hook calls): proves it does not
  /// busy-spin while waiting or idle.
  uint64_t workerIdleYields() const { return workerIdleYields_.loadRelaxed(); }

private:
  static void workerEntryThunk(void *ctx);
  /// Yield the CPU between work items (injected hook, or a reordering barrier).
  void workerIdle();

  /// Result of a shared-cache lookup, including the cross-core fnPtr gate.
  struct SharedLookup {
    void *fnPtr = nullptr;
    uint32_t bucketIndex = 0;
    bool hasReadToken = false;
    bool readyButNotShareable = false;
  };

  // shared cache helpers (POD table in the shared blob)
  uint64_t hashIdentity(uint32_t funcIndex, const EJitDimPair *dims,
                        uint32_t numDims) const;
  SharedLookup cacheLookup(uint32_t funcIndex, const EJitDimPair *dims,
                           uint32_t numDims);
  EJitPublishStatus cachePublish(const EJitCompileRequest &req, void *fnPtr);

  // switch/version helpers
  bool isInstanceEnabled(uint32_t dimType, uint32_t instanceId) const;
  uint32_t instanceVersion(uint32_t dimType, uint32_t instanceId) const;
  bool versionsCurrent(const EJitCompileRequest &req) const;

  // queue/dedup helpers
  bool queuePush(const EJitCompileRequest &req);
  bool queuePop(EJitCompileRequest &out);
  /// Claim the in-flight slot for \p funcIndex at generation \p gen: CAS
  /// 0->gen.
  EJitDedupResult dedupMark(uint32_t funcIndex, uint32_t gen);
  /// Release the in-flight slot ONLY if it still holds \p gen: CAS gen->0. A
  /// stale worker (older gen) therefore cannot clear a newer generation's bit.
  void dedupClear(uint32_t funcIndex, uint32_t gen);

  void runCompile(const EJitCompileRequest &req);

  EJitSharedTaskPoolState *state_ = nullptr;
  CompileCallback compileFn_ = nullptr;
  void *compileCtx_ = nullptr;
  ReleaseCallback releaseFn_ = nullptr;
  void *releaseCtx_ = nullptr;
  PrepareCodeCallback prepareCodeFn_ = nullptr;
  void *prepareCodeCtx_ = nullptr;
  WorkerStartFn workerStart_ = nullptr;
  WorkerStopFn workerStop_ = nullptr;
  void *workerCtx_ = nullptr;
  WorkerIdleFn workerIdle_ = nullptr;
  void *workerIdleCtx_ = nullptr;
  uint64_t regFingerprint_ = 0;
  EJitCompileMode configuredMode_ = EJitCompileMode::Async;
  bool codeSharingEnabled_ = false;
  bool isOwner_ = false;

  // Worker observability + startup-wait bound (owner-local).
  EJitAtomicU64 workerConsumeLoops_{0};
  EJitAtomicU32 workerWaitedForReady_{0};
  EJitAtomicU64 workerIdleYields_{0};
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITSHAREDTASKPOOL_H
