//===-- EJitTaskPool.cpp - SRE taskpool compile scheduler -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implementation of the dedup table, taskpool cache, and the EJitTaskPool
//  scheduler. Uses only EJitAtomic + EJitQueue — no std::thread/async/future/
//  promise/mutex/shared_mutex/condition_variable, no <atomic>, and no
//  std::string/std::function/std::vector on any path.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitTaskPool.h"

using namespace llvm;
using namespace llvm::ejit;

namespace {
constexpr uint32_t kCacheEmpty =
    static_cast<uint32_t>(EJitCacheEntryState::Empty);
constexpr uint32_t kCacheReady =
    static_cast<uint32_t>(EJitCacheEntryState::Ready);
constexpr uint32_t kCacheFailed =
    static_cast<uint32_t>(EJitCacheEntryState::Failed);
constexpr uint32_t kCacheCancelled =
    static_cast<uint32_t>(EJitCacheEntryState::Cancelled);
} // namespace

//===----------------------------------------------------------------------===//
// EJitDedupTable
//
// Slot publication protocol (see EJitDedupState in the header):
//   Empty -> Claiming -> Pending -> Compiling -> Publishing -> Empty
//
// Identity (funcIndex/cacheKey/version) is written while the slot is in the
// transient Claiming state and then published with a single release store of
// `state` (Empty/Claiming are skipped by every matcher). Readers match only the
// committed states via an acquire load of `state`, so they never observe a
// half-written identity. On aarch64_be this is safe because each field is a
// single naturally-aligned atomic scalar accessed by type (never reinterpreted
// as bytes): big-endian only reorders bytes WITHIN a scalar, which is invisible
// when the same typed load/store is used on both ends. The release/acquire pair
// maps to stlr/ldar (or a dmb-guarded sequence) and orders the identity writes
// before the state publication independently of endianness.
//===----------------------------------------------------------------------===//

namespace {
/// True if \p st is a committed (reader-visible) in-flight state.
inline bool isCommitted(uint32_t st) {
  return st == EJitDedupPending || st == EJitDedupCompiling ||
         st == EJitDedupPublishing;
}
} // namespace

EJitDedupResult EJitDedupTable::tryMarkPending(uint32_t funcIndex,
                                               uint64_t cacheKey,
                                               uint32_t version) {
  const uint32_t base = bucketBase(funcIndex);

  // Pass 1: duplicate detection. Only committed slots match — a slot still in
  // Claiming has no published identity yet and is deliberately skipped (it is
  // never treated as a valid duplicate and never blocks: the claiming producer
  // always resolves it to Pending).
  for (uint32_t i = 0; i < kSlots; ++i) {
    EJitDedupSlot &s = slots_[base + i];
    if (isCommitted(s.state.loadAcquire()) &&
        s.funcIndex.loadRelaxed() == funcIndex &&
        s.cacheKey.loadRelaxed() == cacheKey &&
        s.version.loadRelaxed() == version)
      return EJitDedupResult::AlreadyPending;
  }

  // Pass 2: claim a free slot. CAS Empty->Claiming reserves it; the identity is
  // written while Claiming, then a release store of Pending publishes both the
  // identity and the slot to acquiring readers.
  for (uint32_t i = 0; i < kSlots; ++i) {
    EJitDedupSlot &s = slots_[base + i];
    uint32_t expected = EJitDedupEmpty;
    if (s.state.compareExchange(expected, EJitDedupClaiming)) {
      s.funcIndex.storeRelaxed(funcIndex);
      s.cacheKey.storeRelaxed(cacheKey);
      s.version.storeRelaxed(version);
      s.state.storeRelease(EJitDedupPending); // publishes identity
      return EJitDedupResult::AcquiredPending;
    }
  }
  return EJitDedupResult::DedupFull;
}

bool EJitDedupTable::markCompiling(uint32_t funcIndex, uint64_t cacheKey,
                                   uint32_t version) {
  const uint32_t base = bucketBase(funcIndex);
  for (uint32_t i = 0; i < kSlots; ++i) {
    EJitDedupSlot &s = slots_[base + i];
    if (isCommitted(s.state.loadAcquire()) &&
        s.funcIndex.loadRelaxed() == funcIndex &&
        s.cacheKey.loadRelaxed() == cacheKey &&
        s.version.loadRelaxed() == version) {
      uint32_t expected = EJitDedupPending;
      if (s.state.compareExchange(expected, EJitDedupCompiling))
        return true;
    }
  }
  return false;
}

bool EJitDedupTable::beginPublish(uint32_t funcIndex, uint64_t cacheKey,
                                  uint32_t version) {
  const uint32_t base = bucketBase(funcIndex);
  for (uint32_t i = 0; i < kSlots; ++i) {
    EJitDedupSlot &s = slots_[base + i];
    if (isCommitted(s.state.loadAcquire()) &&
        s.funcIndex.loadRelaxed() == funcIndex &&
        s.cacheKey.loadRelaxed() == cacheKey &&
        s.version.loadRelaxed() == version) {
      uint32_t expected = EJitDedupCompiling;
      if (s.state.compareExchange(expected, EJitDedupPublishing))
        return true;
    }
  }
  return false;
}

bool EJitDedupTable::finishPublish(uint32_t funcIndex, uint64_t cacheKey,
                                   uint32_t version) {
  const uint32_t base = bucketBase(funcIndex);
  for (uint32_t i = 0; i < kSlots; ++i) {
    EJitDedupSlot &s = slots_[base + i];
    if (isCommitted(s.state.loadAcquire()) &&
        s.funcIndex.loadRelaxed() == funcIndex &&
        s.cacheKey.loadRelaxed() == cacheKey &&
        s.version.loadRelaxed() == version) {
      uint32_t expected = EJitDedupPublishing;
      if (s.state.compareExchange(expected, EJitDedupEmpty))
        return true;
    }
  }
  return false;
}

bool EJitDedupTable::cancel(uint32_t funcIndex, uint64_t cacheKey) {
  const uint32_t base = bucketBase(funcIndex);
  bool cancelled = false;
  for (uint32_t i = 0; i < kSlots; ++i) {
    EJitDedupSlot &s = slots_[base + i];
    uint32_t st = s.state.loadAcquire();
    while (isCommitted(st) && s.funcIndex.loadRelaxed() == funcIndex &&
           s.cacheKey.loadRelaxed() == cacheKey) {
      uint32_t expected = st;
      if (s.state.compareExchange(expected, EJitDedupEmpty)) {
        cancelled = true;
        break;
      }
      st = expected; // CAS failed: re-observe the state and retry.
    }
  }
  return cancelled;
}

void EJitDedupTable::clear(uint32_t funcIndex, uint64_t cacheKey,
                           uint32_t version) {
  const uint32_t base = bucketBase(funcIndex);
  for (uint32_t i = 0; i < kSlots; ++i) {
    EJitDedupSlot &s = slots_[base + i];
    if (isCommitted(s.state.loadAcquire()) &&
        s.funcIndex.loadRelaxed() == funcIndex &&
        s.cacheKey.loadRelaxed() == cacheKey &&
        s.version.loadRelaxed() == version) {
      s.state.storeRelease(EJitDedupEmpty);
      return;
    }
  }
}

uint32_t EJitDedupTable::pendingCount() const {
  uint32_t n = 0;
  for (uint32_t i = 0; i < kBuckets * kSlots; ++i)
    if (isCommitted(slots_[i].state.loadRelaxed()))
      ++n;
  return n;
}

//===----------------------------------------------------------------------===//
// EJitTaskPoolCache
//===----------------------------------------------------------------------===//

void *EJitTaskPoolCache::lookup(uint32_t funcIndex, uint64_t cacheKey,
                                uint32_t version) {
  const uint32_t base = bucketBase(funcIndex);
  for (uint32_t i = 0; i < kSlots; ++i) {
    EJitCacheEntry &e = entries_[base + i];
    if (e.state.loadAcquire() == kCacheReady &&
        e.funcIndex.loadRelaxed() == funcIndex &&
        e.cacheKey.loadRelaxed() == cacheKey &&
        e.version.loadRelaxed() == version)
      return reinterpret_cast<void *>(e.fnPtr.loadAcquire());
  }
  return nullptr;
}

bool EJitTaskPoolCache::publish(uint32_t funcIndex, uint64_t cacheKey,
                                uint32_t version, void *ptr) {
  const uint32_t base = bucketBase(funcIndex);

  // Reuse an existing slot for the same (funcIndex, cacheKey) if present.
  for (uint32_t i = 0; i < kSlots; ++i) {
    EJitCacheEntry &e = entries_[base + i];
    if (e.state.loadAcquire() != kCacheEmpty &&
        e.funcIndex.loadRelaxed() == funcIndex &&
        e.cacheKey.loadRelaxed() == cacheKey) {
      e.fnPtr.storeRelease(reinterpret_cast<uintptr_t>(ptr));
      e.version.storeRelaxed(version);
      e.state.storeRelease(kCacheReady);
      return true;
    }
  }

  // Otherwise claim a reusable slot (Empty/Failed/Cancelled). Writers are the
  // single consumer, so a plain store sequence is sufficient; readers acquire
  // on state, which is released last after identity and pointer are written.
  for (uint32_t i = 0; i < kSlots; ++i) {
    EJitCacheEntry &e = entries_[base + i];
    uint32_t st = e.state.loadAcquire();
    if (st == kCacheEmpty || st == kCacheFailed || st == kCacheCancelled) {
      e.funcIndex.storeRelaxed(funcIndex);
      e.cacheKey.storeRelaxed(cacheKey);
      e.version.storeRelaxed(version);
      e.fnPtr.storeRelease(reinterpret_cast<uintptr_t>(ptr));
      e.state.storeRelease(kCacheReady);
      return true;
    }
  }
  return false; // Bucket full: never silently overwrite a different key.
}

bool EJitTaskPoolCache::freeCode(uint32_t funcIndex, uint64_t cacheKey) {
  const uint32_t base = bucketBase(funcIndex);
  bool freed = false;
  for (uint32_t i = 0; i < kSlots; ++i) {
    EJitCacheEntry &e = entries_[base + i];
    if (e.state.loadAcquire() == kCacheReady &&
        e.funcIndex.loadRelaxed() == funcIndex &&
        e.cacheKey.loadRelaxed() == cacheKey) {
      // Logical free: future lookups miss. fnPtr is intentionally left intact —
      // the SRE code-pool physical memory is NOT released here (deferred).
      e.state.storeRelease(kCacheEmpty);
      freed = true;
    }
  }
  return freed;
}

uint32_t EJitTaskPoolCache::readyCount() const {
  uint32_t n = 0;
  for (uint32_t i = 0; i < kBuckets * kSlots; ++i)
    if (entries_[i].state.loadRelaxed() == kCacheReady)
      ++n;
  return n;
}

//===----------------------------------------------------------------------===//
// EJitTaskPool
//===----------------------------------------------------------------------===//

void *EJitTaskPool::runCompile(const EJitCompileRequest &req,
                               EJitCompileOrGetStatus &outStatus,
                               bool fromWorker) {
  // Drop requests whose version no longer matches (logically invalidated).
  if (req.version != switch_.getVersion()) {
    dedup_.clear(req.funcIndex, req.cacheKey, req.version);
    counters_.compileFailed.fetchAdd(1);
    outStatus = EJitCompileOrGetStatus::CompileFailed;
    return nullptr;
  }

  // Take ownership of the compile (Pending -> Compiling). Fails if the slot was
  // cancelled/freed (freeCode) or is no longer Pending.
  if (!dedup_.markCompiling(req.funcIndex, req.cacheKey, req.version)) {
    counters_.compileFailed.fetchAdd(1);
    outStatus = EJitCompileOrGetStatus::CompileFailed;
    return nullptr;
  }

  void *fn = nullptr;
  bool ok = compileFn_ && compileFn_(compileCtx_, req, &fn);

  // A version bump during compilation invalidates this result. Force the slot
  // Empty (cancel) regardless of its current committed state.
  if (req.version != switch_.getVersion()) {
    dedup_.cancel(req.funcIndex, req.cacheKey);
    counters_.compileFailed.fetchAdd(1);
    outStatus = EJitCompileOrGetStatus::CompileFailed;
    return nullptr;
  }

  if (!ok || !fn) {
    // Compile failed: release the dedup slot so the key can be retried.
    dedup_.clear(req.funcIndex, req.cacheKey, req.version);
    counters_.compileFailed.fetchAdd(1);
    outStatus = EJitCompileOrGetStatus::CompileFailed;
    return nullptr;
  }

  // Commit gate part 1 (Compiling -> Publishing). If freeCode cancelled the
  // slot during compilation this fails and we drop the result without writing.
  if (!dedup_.beginPublish(req.funcIndex, req.cacheKey, req.version)) {
    counters_.compileFailed.fetchAdd(1);
    outStatus = EJitCompileOrGetStatus::CompileFailed;
    return nullptr;
  }

#ifdef EJIT_SRE_TASKPOOL_TESTING
  // Deterministically exercise a freeCode() racing inside the publish window.
  if (prePublishHook_)
    prePublishHook_(prePublishCtx_, req);
#endif

  // Write the result into the fixed cache. publish() returns false only when
  // the bucket is full of other keys — it never silently overwrites.
  bool published = cache_.publish(req.funcIndex, req.cacheKey, req.version, fn);
  if (!published) {
    // Cache bucket full: release the dedup slot and report it explicitly so the
    // caller does not mistake this for a successful compile. Nothing is cached,
    // so a later lookup will not falsely hit.
    dedup_.clear(req.funcIndex, req.cacheKey, req.version);
    counters_.publishFailed.fetchAdd(1);
    outStatus = EJitCompileOrGetStatus::CacheFullFallback;
    return nullptr;
  }

  // Commit gate part 2 (Publishing -> Empty). If freeCode forced the slot Empty
  // during the publish window this CAS fails; the entry we just wrote is now
  // logically freed, so roll it back out of the cache and report cancellation.
  if (!dedup_.finishPublish(req.funcIndex, req.cacheKey, req.version)) {
    cache_.freeCode(req.funcIndex, req.cacheKey);
    counters_.compileFailed.fetchAdd(1);
    outStatus = EJitCompileOrGetStatus::CompileFailed;
    return nullptr;
  }

  if (fromWorker)
    counters_.asyncCompiles.fetchAdd(1);
  else
    counters_.syncCompiles.fetchAdd(1);
  outStatus = EJitCompileOrGetStatus::SyncCompiled;
  return fn;
}

EJitTaskPool::CompileOrGetResult
EJitTaskPool::compileOrGet(uint32_t funcIndex, uint64_t cacheKey,
                           void *fallback, uintptr_t userData) {
  const uint32_t version = switch_.getVersion();

  // 1. Cache hit.
  if (void *p = cache_.lookup(funcIndex, cacheKey, version)) {
    counters_.cacheHits.fetchAdd(1);
    return {EJitCompileOrGetStatus::CacheHit, p};
  }

  // 2. Disabled / Off → fallback, never enqueue or compile.
  if (!switch_.isEnabled() || switch_.getMode() == EJitCompileMode::Off)
    return {EJitCompileOrGetStatus::DisabledFallback, fallback};

  // 3. Dedup reservation (shared by sync and async).
  EJitDedupResult d = dedup_.tryMarkPending(funcIndex, cacheKey, version);
  if (d == EJitDedupResult::AlreadyPending) {
    counters_.alreadyPending.fetchAdd(1);
    return {EJitCompileOrGetStatus::AlreadyPending, fallback};
  }
  if (d == EJitDedupResult::DedupFull) {
    counters_.dedupFull.fetchAdd(1);
    return {EJitCompileOrGetStatus::DedupFullFallback, fallback};
  }

  EJitCompileRequest req;
  req.funcIndex = funcIndex;
  req.version = version;
  req.cacheKey = cacheKey;
  req.fallbackPtr = reinterpret_cast<uintptr_t>(fallback);
  req.userData = userData;

  // 4a. Sync miss: compile on the calling stack and publish.
  if (switch_.getMode() == EJitCompileMode::Sync) {
    EJitCompileOrGetStatus st;
    void *p = runCompile(req, st, /*fromWorker=*/false);
    return {st, p ? p : fallback};
  }

  // 4b. Async miss: enqueue and return pending. On queue-full, roll back the
  // dedup reservation so the request is not stuck permanently pending.
  if (!queue_.push(req)) {
    dedup_.clear(funcIndex, cacheKey, version);
    counters_.queueFull.fetchAdd(1);
    return {EJitCompileOrGetStatus::QueueFullFallback, fallback};
  }
  counters_.asyncEnqueues.fetchAdd(1);
  return {EJitCompileOrGetStatus::EnqueuedPending, fallback};
}

EJitTaskPool::CompileOrGetResult
EJitTaskPool::syncCompile(const EJitCompileRequest &reqIn) {
  const uint32_t version = switch_.getVersion();
  void *fallback = reinterpret_cast<void *>(reqIn.fallbackPtr);

  if (void *p = cache_.lookup(reqIn.funcIndex, reqIn.cacheKey, version)) {
    counters_.cacheHits.fetchAdd(1);
    return {EJitCompileOrGetStatus::CacheHit, p};
  }

  EJitDedupResult d =
      dedup_.tryMarkPending(reqIn.funcIndex, reqIn.cacheKey, version);
  if (d == EJitDedupResult::AlreadyPending) {
    counters_.alreadyPending.fetchAdd(1);
    return {EJitCompileOrGetStatus::AlreadyPending, fallback};
  }
  if (d == EJitDedupResult::DedupFull) {
    counters_.dedupFull.fetchAdd(1);
    return {EJitCompileOrGetStatus::DedupFullFallback, fallback};
  }

  EJitCompileRequest req = reqIn;
  req.version = version;
  EJitCompileOrGetStatus st;
  void *p = runCompile(req, st, /*fromWorker=*/false);
  return {st, p ? p : fallback};
}

bool EJitTaskPool::pollOne() {
  EJitCompileRequest req;
  if (!queue_.pop(req))
    return false;
  EJitCompileOrGetStatus st;
  runCompile(req, st, /*fromWorker=*/true);
  return true;
}

unsigned EJitTaskPool::pollBudget(unsigned maxItems) {
  unsigned n = 0;
  while (n < maxItems && pollOne())
    ++n;
  return n;
}

bool EJitTaskPool::freeCode(uint32_t funcIndex, uint64_t cacheKey) {
  counters_.freeCodeCalls.fetchAdd(1);
  // Cancel any in-flight request first so a worker mid-compile cannot pass its
  // publish gate, then drop any Ready cache entry. A worker that already wrote
  // the cache between these two steps will fail its finishPublish CAS and roll
  // its own entry back (see runCompile), so a cancelled key never stays cached.
  // This is a LOGICAL free: the SRE code-pool physical memory is not released.
  bool cancelled = dedup_.cancel(funcIndex, cacheKey);
  bool freed = cache_.freeCode(funcIndex, cacheKey);
  return cancelled || freed;
}

void EJitTaskPool::getStats(EJitTaskPoolStatsSnapshot &out) const {
  out.cacheHits = counters_.cacheHits.loadRelaxed();
  out.syncCompiles = counters_.syncCompiles.loadRelaxed();
  out.asyncCompiles = counters_.asyncCompiles.loadRelaxed();
  out.asyncEnqueues = counters_.asyncEnqueues.loadRelaxed();
  out.alreadyPending = counters_.alreadyPending.loadRelaxed();
  out.queueFull = counters_.queueFull.loadRelaxed();
  out.dedupFull = counters_.dedupFull.loadRelaxed();
  out.compileFailed = counters_.compileFailed.loadRelaxed();
  out.publishFailed = counters_.publishFailed.loadRelaxed();
  out.freeCodeCalls = counters_.freeCodeCalls.loadRelaxed();
  out.readyEntries = cache_.readyCount();
  out.pendingEntries = dedup_.pendingCount();
  out.queueApproxSize = queue_.approximateSize();
}
