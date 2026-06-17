//===-- EJitTaskPool.h - SRE taskpool compile scheduler -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  EmbeddedJIT SRE taskpool: a unified compile-scheduling front end for
//  ejit_compile_or_get. Cache hit, synchronous miss, and asynchronous miss all
//  flow through one dedup/cache pipeline; only the consumption differs:
//
//    * cache hit  -> return the JIT pointer immediately;
//    * sync miss  -> reserve dedup, compile on the calling stack, publish;
//    * async miss -> reserve dedup, enqueue, return fallback/pending. An
//                    external SRE task (or an explicit poll worker) later
//                    dequeues, compiles, and publishes. EJIT never starts a
//                    thread; "async" means producer + explicit single worker.
//
//  Target platform is aarch64_be with no C++ thread library, so this file uses
//  no std::thread/async/future/promise/mutex/shared_mutex/condition_variable.
//  All shared state lives in EJitAtomic cells (EJitAtomic.h); the work queue is
//  EJitQueue (EJitSreQueue.h). The fixed-layout request record,
//  EJitCompileRequest, is defined in EJitSreQueue.h (the queue element) and is
//  used unchanged here.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITTASKPOOL_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITTASKPOOL_H

#include "llvm/ExecutionEngine/EJIT/EJitAtomic.h"
#include "llvm/ExecutionEngine/EJIT/EJitIpcLock.h"
#include "llvm/ExecutionEngine/EJIT/EJitSreQueue.h"
#include <cstdint>

//===----------------------------------------------------------------------===//
// Compile-time sizing (overridable by the build via -D). The dedup table and
// cache are bucketed (bucket = funcIndex % kBuckets) with a small fixed number
// of slots per bucket — this spreads contention across 32 buckets instead of
// degenerating to a single funcIndex%32 flag, and avoids std::unordered_map.
//===----------------------------------------------------------------------===//
#ifndef EJIT_SRE_TASKPOOL_BUCKETS
#define EJIT_SRE_TASKPOOL_BUCKETS 32u
#endif
#ifndef EJIT_SRE_TASKPOOL_BUCKET_SLOTS
#define EJIT_SRE_TASKPOOL_BUCKET_SLOTS 8u
#endif

namespace llvm {
namespace ejit {

//===----------------------------------------------------------------------===//
// Modes and result statuses
//===----------------------------------------------------------------------===//

/// Scheduling mode held by the SwitchController.
enum class EJitCompileMode : uint32_t {
  Off = 0,   ///< Disabled: compile_or_get never enqueues/compiles → fallback.
  Sync = 1,  ///< Miss compiles on the calling stack.
  Async = 2, ///< Miss enqueues; an external worker compiles later.
};

/// Outcome of EJitTaskPool::compileOrGet().
enum class EJitCompileOrGetStatus : uint32_t {
  CacheHit = 0,        ///< Found in the taskpool cache; fnPtr is the JIT code.
  DisabledFallback,    ///< Off/disabled; fnPtr is the caller's fallback.
  EnqueuedPending,     ///< Async miss enqueued; fnPtr is fallback (pending).
  AlreadyPending,      ///< Equal request already in flight; fnPtr is fallback.
  QueueFullFallback,   ///< Queue full; dedup rolled back; fnPtr is fallback.
  DedupFullFallback,   ///< Dedup bucket full; fnPtr is fallback.
  SyncCompiled,        ///< Compiled+published on this stack; fnPtr is JIT code.
  CacheFullFallback,   ///< Compiled, but the fixed cache bucket was full so the
                       ///< result was NOT published; fnPtr is fallback and a
                       ///< later lookup will not falsely hit.
  CompileFailed,       ///< Compile failed/cancelled; fnPtr is fallback.
};

//===----------------------------------------------------------------------===//
// EJitSwitchController
//
// Enable flag + scheduling mode + a monotonically increasing version. Bumping
// the version logically invalidates older in-flight requests: a worker drops
// any request whose version no longer matches and does not publish stale code.
//===----------------------------------------------------------------------===//
class EJitSwitchController {
public:
  EJitSwitchController()
      : enabled_(1u), mode_(static_cast<uint32_t>(EJitCompileMode::Sync)),
        version_(1u) {}

  bool isEnabled() const { return enabled_.loadAcquire() != 0u; }
  void setEnabled(bool e) { enabled_.storeRelease(e ? 1u : 0u); }

  EJitCompileMode getMode() const {
    return static_cast<EJitCompileMode>(mode_.loadAcquire());
  }
  void setMode(EJitCompileMode m) {
    mode_.storeRelease(static_cast<uint32_t>(m));
  }

  uint32_t getVersion() const { return version_.loadAcquire(); }

  /// Increment and return the new version.
  uint32_t bumpVersion() { return version_.fetchAdd(1u) + 1u; }

  /// Turn on scheduling and publish a new generation.
  uint32_t activate(EJitCompileMode mode) {
    mode_.storeRelease(static_cast<uint32_t>(mode));
    enabled_.storeRelease(1u);
    return bumpVersion();
  }

  /// Turn off scheduling and publish a new generation.
  uint32_t deactivate() {
    enabled_.storeRelease(0u);
    return bumpVersion();
  }

private:
  EJitAtomicU32 enabled_;
  EJitAtomicU32 mode_;
  EJitAtomicU32 version_;
};

//===----------------------------------------------------------------------===//
// EJitDedupTable
//===----------------------------------------------------------------------===//

/// Per-slot dedup state. The slot lifecycle is:
///
///   Empty -> Claiming -> Pending -> Compiling -> Publishing -> Empty
///
/// Claiming is a transient, producer-owned state used WHILE the slot identity
/// (funcIndex/cacheKey/version) is being written. The identity is published to
/// the reader-visible "committed" states (Pending/Compiling/Publishing) via a
/// release store of `state`; only those committed states participate in
/// duplicate matching. A reader therefore never observes a half-written
/// identity (Task 2 race fix), and a Claiming slot is never matched and never
/// causes a permanent stall (the claiming producer always resolves it).
///
/// Publishing sits between "compile finished" and "cache write committed". It
/// lets freeCode reliably cancel a result that is mid-publish: freeCode forces
/// any committed state back to Empty, and the worker's final
/// Publishing->Empty CAS then fails, so the worker rolls back its cache write
/// (Task 3 publish gate).
enum EJitDedupState : uint32_t {
  EJitDedupEmpty = 0,
  EJitDedupClaiming = 1,
  EJitDedupPending = 2,
  EJitDedupCompiling = 3,
  EJitDedupPublishing = 4,
};

/// Result of tryMarkPending().
enum class EJitDedupResult : uint32_t {
  AcquiredPending = 0, ///< Caller claimed the slot; proceed to enqueue/compile.
  AlreadyPending = 1,  ///< An equal (funcIndex,cacheKey,version) is in flight.
  DedupFull = 2,       ///< Bucket has no free slot.
};

/// One dedup slot: a single CAS-able state plus the identity it guards. Every
/// field is a standalone naturally-aligned atomic scalar (no bitfields), so
/// the layout and access semantics are identical on big- and little-endian.
struct EJitDedupSlot {
  EJitAtomicU32 state;
  EJitAtomicU32 funcIndex;
  EJitAtomicU32 version;
  EJitAtomicU64 cacheKey;
};

/// Fixed-size, bucketed dedup table. Slot identity is matched on the full
/// triple (funcIndex, cacheKey, version), so distinct functions that hash to
/// the same bucket never alias.
class EJitDedupTable {
public:
  static constexpr uint32_t kBuckets = EJIT_SRE_TASKPOOL_BUCKETS;
  static constexpr uint32_t kSlots = EJIT_SRE_TASKPOOL_BUCKET_SLOTS;

  EJitDedupTable() = default;

  /// Try to reserve a slot for the request. On success the slot ends in the
  /// committed Pending state with a fully-published identity.
  EJitDedupResult tryMarkPending(uint32_t funcIndex, uint64_t cacheKey,
                                 uint32_t version);

  /// Transition a matching Pending slot to Compiling. Returns false if the slot
  /// is not in Pending state for this triple (e.g. it was cancelled/freed).
  bool markCompiling(uint32_t funcIndex, uint64_t cacheKey, uint32_t version);

  /// Commit gate part 1: CAS a matching Compiling slot to Publishing. Returns
  /// true if the worker still owns the slot (freeCode has not cancelled it) and
  /// may proceed to write the cache. Returns false if it was cancelled.
  bool beginPublish(uint32_t funcIndex, uint64_t cacheKey, uint32_t version);

  /// Commit gate part 2: CAS a matching Publishing slot back to Empty. Returns
  /// true if the worker won (its cache write stands). Returns false if freeCode
  /// forced the slot Empty during the publish window → the worker must roll the
  /// just-written cache entry back.
  bool finishPublish(uint32_t funcIndex, uint64_t cacheKey, uint32_t version);

  /// Cancel any committed in-flight slot (Pending/Compiling/Publishing) for
  /// (funcIndex,cacheKey), regardless of version, forcing it back to Empty.
  /// Returns true if a slot was cancelled. Used by freeCode.
  bool cancel(uint32_t funcIndex, uint64_t cacheKey);

  /// Release a matching committed slot back to Empty (rollback/cleanup).
  void clear(uint32_t funcIndex, uint64_t cacheKey, uint32_t version);

  /// Best-effort count of committed in-flight slots (Pending/Compiling/Publishing).
  uint32_t pendingCount() const;

#ifdef EJIT_SRE_TASKPOOL_TESTING
  /// Test-only: claim a free slot and LEAVE it in Claiming (identity written
  /// but not yet released to Pending), to exercise the duplicate-detection
  /// protocol's handling of in-initialization slots. Returns false if full.
  bool beginClaimForTest(uint32_t funcIndex, uint64_t cacheKey,
                         uint32_t version) {
    const uint32_t base = bucketBase(funcIndex);
    for (uint32_t i = 0; i < kSlots; ++i) {
      EJitDedupSlot &s = slots_[base + i];
      uint32_t expected = EJitDedupEmpty;
      if (s.state.compareExchange(expected, EJitDedupClaiming)) {
        s.funcIndex.storeRelaxed(funcIndex);
        s.cacheKey.storeRelaxed(cacheKey);
        s.version.storeRelaxed(version);
        return true;
      }
    }
    return false;
  }

  /// Test-only: release a Claiming slot to the committed Pending state.
  void releaseClaimForTest(uint32_t funcIndex, uint64_t cacheKey,
                           uint32_t version) {
    const uint32_t base = bucketBase(funcIndex);
    for (uint32_t i = 0; i < kSlots; ++i) {
      EJitDedupSlot &s = slots_[base + i];
      if (s.state.loadRelaxed() == EJitDedupClaiming &&
          s.funcIndex.loadRelaxed() == funcIndex &&
          s.cacheKey.loadRelaxed() == cacheKey &&
          s.version.loadRelaxed() == version) {
        s.state.storeRelease(EJitDedupPending);
        return;
      }
    }
  }

  /// Test-only: read the raw state of the first non-Empty slot matching
  /// (funcIndex,cacheKey), or EJitDedupEmpty if none.
  uint32_t peekStateForTest(uint32_t funcIndex, uint64_t cacheKey) const {
    const uint32_t base = bucketBase(funcIndex);
    for (uint32_t i = 0; i < kSlots; ++i) {
      const EJitDedupSlot &s = slots_[base + i];
      if (s.state.loadRelaxed() != EJitDedupEmpty &&
          s.funcIndex.loadRelaxed() == funcIndex &&
          s.cacheKey.loadRelaxed() == cacheKey)
        return s.state.loadRelaxed();
    }
    return EJitDedupEmpty;
  }
#endif // EJIT_SRE_TASKPOOL_TESTING

private:
  uint32_t bucketBase(uint32_t funcIndex) const {
    return (funcIndex % kBuckets) * kSlots;
  }

  uint32_t bucketIndex(uint32_t funcIndex) const { return funcIndex % kBuckets; }

  EJitDedupSlot slots_[kBuckets * kSlots];
  EJitIpcBucketLock bucketLocks_{kBuckets};
};

//===----------------------------------------------------------------------===//
// EJitTaskPoolCache
//===----------------------------------------------------------------------===//

enum class EJitCacheEntryState : uint32_t {
  Empty = 0,
  Ready = 1,
  Failed = 2,
  Cancelled = 3,
};

/// One cache entry. fnPtr is published with release and read with acquire so a
/// reader that observes Ready also observes a fully-written pointer.
struct EJitCacheEntry {
  EJitAtomicU32 state;
  EJitAtomicU32 funcIndex;
  EJitAtomicU32 version;
  EJitAtomicU64 cacheKey;
  EJitAtomicUPtr fnPtr;
};

/// Fixed-size, bucketed cache. Writers are the single consumer (sync caller or
/// poll worker); readers are the many producers. No std::unordered_map.
class EJitTaskPoolCache {
public:
  static constexpr uint32_t kBuckets = EJIT_SRE_TASKPOOL_BUCKETS;
  static constexpr uint32_t kSlots = EJIT_SRE_TASKPOOL_BUCKET_SLOTS;

  EJitTaskPoolCache() = default;

  /// Return the published JIT pointer for a Ready entry matching the triple, or
  /// nullptr on miss.
  void *lookup(uint32_t funcIndex, uint64_t cacheKey, uint32_t version);

  /// Publish a Ready entry. Returns false if no slot is available (the cache
  /// never silently overwrites a different key).
  bool publish(uint32_t funcIndex, uint64_t cacheKey, uint32_t version,
               void *ptr);

  /// Logical free: mark a matching Ready entry Empty so future lookups miss.
  /// Returns true if an entry was freed. Does NOT release physical code memory.
  bool freeCode(uint32_t funcIndex, uint64_t cacheKey);

  /// Best-effort count of Ready entries.
  uint32_t readyCount() const;

private:
  uint32_t bucketBase(uint32_t funcIndex) const {
    return (funcIndex % kBuckets) * kSlots;
  }

  uint32_t bucketIndex(uint32_t funcIndex) const { return funcIndex % kBuckets; }

  EJitCacheEntry entries_[kBuckets * kSlots];
  EJitIpcBucketLock bucketLocks_{kBuckets};
};

//===----------------------------------------------------------------------===//
// EJitTaskPool stats
//===----------------------------------------------------------------------===//

/// Live atomic counters for taskpool activity. EJitAtomic only (no std::atomic).
/// These are monotonically increasing event counts; live gauges (ready/pending/
/// queue size) are sampled separately at snapshot time.
struct EJitTaskPoolCounters {
  EJitAtomicU64 cacheHits;
  EJitAtomicU64 syncCompiles;   ///< Successful compiles on the caller's stack.
  EJitAtomicU64 asyncCompiles;  ///< Successful compiles via a poll worker.
  EJitAtomicU64 asyncEnqueues;  ///< Requests pushed onto the async queue.
  EJitAtomicU64 alreadyPending; ///< Duplicate submissions coalesced.
  EJitAtomicU64 queueFull;      ///< Async enqueues rejected (dedup rolled back).
  EJitAtomicU64 dedupFull;      ///< Reservations rejected (bucket full).
  EJitAtomicU64 compileFailed;  ///< Compiles that failed/were cancelled/dropped.
  EJitAtomicU64 publishFailed;  ///< Compiles whose publish hit a full cache bucket.
  EJitAtomicU64 freeCodeCalls;  ///< freeCode() invocations.
};

/// Plain-old-data snapshot of taskpool stats (no atomics; trivially copyable).
struct EJitTaskPoolStatsSnapshot {
  uint64_t cacheHits;
  uint64_t syncCompiles;
  uint64_t asyncCompiles;
  uint64_t asyncEnqueues;
  uint64_t alreadyPending;
  uint64_t queueFull;
  uint64_t dedupFull;
  uint64_t compileFailed;
  uint64_t publishFailed;
  uint64_t freeCodeCalls;
  uint32_t readyEntries;     ///< Live count of Ready cache entries.
  uint32_t pendingEntries;   ///< Live count of in-flight dedup slots.
  uint32_t queueApproxSize;  ///< Approximate async queue depth.
};

//===----------------------------------------------------------------------===//
// EJitTaskPool
//===----------------------------------------------------------------------===//

/// The scheduler that ties the SwitchController, dedup table, cache, and queue
/// together. The actual code generation is supplied via a plain function
/// pointer (never std::function) so nothing heavyweight sits on the hot path
/// and host tests can inject a mock compiler.
class EJitTaskPool {
public:
  /// Compile callback: produce JIT code for \p req. Return true and set
  /// *outFn on success; return false on failure. \p ctx is the cookie passed
  /// to setCompiler().
  using CompileCallback = bool (*)(void *ctx, const EJitCompileRequest &req,
                                   void **outFn);

  struct CompileOrGetResult {
    EJitCompileOrGetStatus status;
    void *fnPtr; ///< JIT code for hits/sync-compiled; fallback otherwise.
  };

  explicit EJitTaskPool(
      uint32_t queueCapacity = EJIT_SRE_TASKPOOL_QUEUE_CAPACITY)
      : queue_(queueCapacity) {}

  void setCompiler(CompileCallback fn, void *ctx) {
    compileFn_ = fn;
    compileCtx_ = ctx;
  }

#ifdef EJIT_SRE_TASKPOOL_TESTING
  /// Test-only hook: invoked inside runCompile after the Compiling->Publishing
  /// gate succeeds and BEFORE the cache write, so a test can deterministically
  /// inject a freeCode() into the publish window (Task 3 race test).
  using PrePublishHook = void (*)(void *ctx, const EJitCompileRequest &req);
  void setPrePublishHookForTest(PrePublishHook fn, void *ctx) {
    prePublishHook_ = fn;
    prePublishCtx_ = ctx;
  }
#endif

  EJitSwitchController &switchController() { return switch_; }
  EJitTaskPoolCache &cache() { return cache_; }
  EJitDedupTable &dedup() { return dedup_; }
  EJitQueue &queue() { return queue_; }

  /// Unified entry point used by ejit_compile_or_get.
  CompileOrGetResult compileOrGet(uint32_t funcIndex, uint64_t cacheKey,
                                  void *fallback, uintptr_t userData = 0);

  /// Explicit synchronous compile of a prepared request (ignores Async/Off
  /// mode; still honors cache + dedup). Compiles on the calling stack.
  CompileOrGetResult syncCompile(const EJitCompileRequest &req);

  /// Low-level enqueue of a prepared request (no dedup bookkeeping).
  bool enqueue(const EJitCompileRequest &req) { return queue_.push(req); }

  /// Consume and process at most one queued request. Returns true if an item
  /// was dequeued (work attempted), false if the queue was empty.
  bool pollOne();

  /// Drain up to \p maxItems queued requests; returns the number processed.
  unsigned pollBudget(unsigned maxItems);

  /// Worker-friendly alias of pollOne().
  bool workerStep() { return pollOne(); }

  /// Activate scheduling in \\p mode and bump generation.
  uint32_t activate(EJitCompileMode mode) { return switch_.activate(mode); }

  /// Deactivate scheduling and bump generation.
  uint32_t deactivate() { return switch_.deactivate(); }

  /// Logical free for (funcIndex, cacheKey): drop the cache entry and cancel
  /// any in-flight request. Returns true if anything was freed/cancelled. Does
  /// NOT release SRE code-pool physical memory (deferred to a later version).
  bool freeCode(uint32_t funcIndex, uint64_t cacheKey);

  /// Best-effort count of in-flight (pending/compiling) requests.
  uint32_t pendingCount() const { return dedup_.pendingCount(); }

  /// Fill \p out with a consistent-enough snapshot of the taskpool counters and
  /// live gauges. Reads are lock-free; gauges may be slightly stale.
  void getStats(EJitTaskPoolStatsSnapshot &out) const;

private:
  /// Run compileFn_ for \p req honoring version cancellation and the dedup
  /// commit gate; publish on success. Returns the published pointer or nullptr,
  /// setting \p outStatus. \p fromWorker selects the success counter
  /// (asyncCompiles vs syncCompiles).
  void *runCompile(const EJitCompileRequest &req,
                   EJitCompileOrGetStatus &outStatus, bool fromWorker);

  EJitSwitchController switch_;
  EJitDedupTable dedup_;
  EJitTaskPoolCache cache_;
  EJitQueue queue_;
  CompileCallback compileFn_ = nullptr;
  void *compileCtx_ = nullptr;
  EJitTaskPoolCounters counters_;
#ifdef EJIT_SRE_TASKPOOL_TESTING
  PrePublishHook prePublishHook_ = nullptr;
  void *prePublishCtx_ = nullptr;
#endif
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITTASKPOOL_H
