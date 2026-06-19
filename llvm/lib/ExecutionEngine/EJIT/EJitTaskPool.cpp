//===-- EJitTaskPool.cpp - SRE taskpool compile scheduler -----------------===//

#include "llvm/ExecutionEngine/EJIT/EJitTaskPool.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
#include <algorithm>
#include <vector>

using namespace llvm;
using namespace llvm::ejit;

//===----------------------------------------------------------------------===//
// EJitSwitchController (§5.1)
//
// Strict 2-D arrays indexed DIRECTLY by (dimType, instanceId). No registry, no
// name hashing, no probing. dimType is an explicit lifecycle index in
// [0, MAX_DIM_TYPES) assigned at AOT time; out-of-range (dimType >=
// MAX_DIM_TYPES or instanceId >= MAX_INSTANCES) is rejected.
//===----------------------------------------------------------------------===//

EJitSwitchController::EJitSwitchController() {
  // Spec §5.1: enabled_ initializes to 1 (enabled), version_ to 0.
  for (uint32_t d = 0; d < MAX_DIM_TYPES; ++d)
    for (uint32_t i = 0; i < MAX_INSTANCES; ++i) {
      enabled_[d][i].storeRelaxed(1);
      version_[d][i].storeRelaxed(0);
    }
}

bool EJitSwitchController::isInstanceEnabled(uint32_t dimType,
                                             uint32_t instanceId) const {
  if (dimType >= MAX_DIM_TYPES || instanceId >= MAX_INSTANCES)
    return false;
  return enabled_[dimType][instanceId].loadRelaxed() != 0;
}

uint32_t EJitSwitchController::getInstanceVersion(uint32_t dimType,
                                                  uint32_t instanceId) const {
  if (dimType >= MAX_DIM_TYPES || instanceId >= MAX_INSTANCES)
    return 0;
  return version_[dimType][instanceId].loadAcquire();
}

bool EJitSwitchController::setEnabled(uint32_t dimType, uint32_t instanceId,
                                      bool wantOn) {
  if (dimType >= MAX_DIM_TYPES || instanceId >= MAX_INSTANCES)
    return false;
  // One CAS; version bumps by exactly one only when the flag actually flips.
  uint8_t expected = wantOn ? 0 : 1;
  uint8_t desired = wantOn ? 1 : 0;
  if (enabled_[dimType][instanceId].compareExchange(expected, desired)) {
    version_[dimType][instanceId].fetchAdd(1);
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// EJitDedupTable: flat O(1) in-flight bit keyed by funcIndex only (§3.5).
//===----------------------------------------------------------------------===//

EJitDedupResult EJitDedupTable::tryMarkPending(uint32_t funcIndex) {
  if (funcIndex >= kCapacity)
    return EJitDedupResult::InvalidFuncIndex; // Reject, never fold.
  uint32_t expected = 0;
  if (inFlight_[funcIndex].compareExchange(expected, 1))
    return EJitDedupResult::Claimed;
  return EJitDedupResult::AlreadyPending;
}

void EJitDedupTable::clear(uint32_t funcIndex) {
  if (funcIndex >= kCapacity)
    return;
  inFlight_[funcIndex].storeRelease(0);
}

uint32_t EJitDedupTable::pendingCount() const {
  uint32_t n = 0;
  for (uint32_t i = 0; i < kCapacity; ++i)
    if (inFlight_[i].loadRelaxed() != 0)
      ++n;
  return n;
}

EJitTaskQueue::EnqueueResult
EJitTaskQueue::tryEnqueue(const EJitCompileRequest &req) {
  switch (dedup_.tryMarkPending(req.funcIndex)) {
  case EJitDedupResult::AlreadyPending:
    return EnqueueResult::AlreadyPending;
  case EJitDedupResult::InvalidFuncIndex:
    return EnqueueResult::InvalidFuncIndex;
  case EJitDedupResult::Claimed:
    break;
  }
  if (!queue_.push(req)) {
    dedup_.clear(req.funcIndex); // Queue full: roll back the in-flight bit.
    return EnqueueResult::QueueFull;
  }
  return EnqueueResult::Enqueued;
}

uint64_t EJitTaskPoolCache::hashKey(uint32_t funcIndex, const EJitDimPair *dims,
                                    uint32_t numDims) const {
  uint64_t key = static_cast<uint64_t>(funcIndex);
  for (uint32_t i = 0; i < numDims; ++i) {
    key ^= (static_cast<uint64_t>(dims[i].dimType) << 32) |
           static_cast<uint64_t>(dims[i].instanceId);
    key *= 0x9e3779b97f4a7c15ULL;
  }
  return key;
}

bool EJitTaskPoolCache::identityMatches(const EJitCacheEntry &e,
                                        uint32_t funcIndex,
                                        const EJitDimPair *dims,
                                        uint32_t numDims) {
  if (e.funcIndex != funcIndex || e.numDims != numDims)
    return false;
  for (uint32_t i = 0; i < numDims; ++i)
    if (e.dims[i].dimType != dims[i].dimType ||
        e.dims[i].instanceId != dims[i].instanceId)
      return false;
  return true;
}

EJitCacheLookupResult EJitTaskPoolCache::lookup(uint32_t funcIndex,
                                                const EJitDimPair *dims,
                                                uint32_t numDims) {
  EJitCacheLookupResult R;
  if (numDims > 4)
    return R;
  uint64_t key = hashKey(funcIndex, dims, numDims);
  uint32_t bucket = static_cast<uint32_t>(key % kBuckets);
  Bucket &B = buckets_[bucket];
  if (!B.lock.tryRead())
    return R;

  auto It = B.entries.find(key);
  if (It == B.entries.end()) {
    B.lock.readRelease();
    return R;
  }

  // A hash key may chain several distinct identities (collision); match the
  // full identity, then verify every dim's version is still current.
  for (const EJitCacheEntry &E : It->second) {
    if (!identityMatches(E, funcIndex, dims, numDims))
      continue;
    bool versionsOk = true;
    for (uint32_t i = 0; i < numDims; ++i) {
      if (E.versions[i] !=
          switch_.getInstanceVersion(dims[i].dimType, dims[i].instanceId)) {
        versionsOk = false;
        break;
      }
    }
    if (!versionsOk)
      break; // Identity matched but stale → miss; keep the read token off.
    void *fn = reinterpret_cast<void *>(E.fnPtr);
    if (!fn)
      break;
    R.fnPtr = fn;
    R.bucketIndex = bucket;
    R.hasReadToken = true;
    return R; // Token intentionally held; caller releases after using fnPtr.
  }

  B.lock.readRelease();
  return R;
}

EJitPublishStatus EJitTaskPoolCache::publish(uint32_t funcIndex,
                                             const EJitDimPair *dims,
                                             uint32_t numDims,
                                             const uint32_t *versions,
                                             void *fnPtr) {
  if (!fnPtr || numDims > 4)
    return EJitPublishStatus::InvalidParam;

  uint64_t key = hashKey(funcIndex, dims, numDims);
  uint32_t bucket = static_cast<uint32_t>(key % kBuckets);
  Bucket &B = buckets_[bucket];

  // write() spins until this bucket's readers drain to 0, so no reader can be
  // holding a pointer we are about to overwrite/release (no use-after-free).
  B.lock.write();

#ifdef EJIT_SRE_TASKPOOL_TESTING
  // Deterministically inject a toggle into the commit window (after the lock is
  // held, before the under-lock recheck) for the race tests.
  if (prePublishHook_)
    prePublishHook_(prePublishCtx_);
#endif

  // Commit gate (§5.3/§5.4 hole fix): re-verify the request's version snapshot
  // still equals the current version for every dim WHILE holding the write
  // lock. If a toggle slipped in between checkpoint 2 and here, the freshly
  // compiled code is stale; reject it rather than stamp it with the new
  // version (which would cause a later lookup to falsely hit stale code).
  for (uint32_t i = 0; i < numDims; ++i) {
    if (versions[i] !=
        switch_.getInstanceVersion(dims[i].dimType, dims[i].instanceId)) {
      B.lock.writeRelease();
      return EJitPublishStatus::VersionMismatch;
    }
  }

  // The entry stores exactly the request's versions (== current, just checked).
  for (EJitCacheEntry &E : B.entries[key]) {
    if (!identityMatches(E, funcIndex, dims, numDims))
      continue;
    void *oldFn = reinterpret_cast<void *>(E.fnPtr);
    for (uint32_t i = 0; i < numDims; ++i)
      E.versions[i] = versions[i];
    E.fnPtr = reinterpret_cast<uintptr_t>(fnPtr);
    B.lock.writeRelease();
    // Release the overwritten code OUTSIDE the bucket write lock: the callback
    // may re-enter the code pool / ORC / allocator / platform and must never
    // run in a short critical section. Readers already drained to 0 before we
    // took the write lock and the entry now points at the new fnPtr, so the old
    // pointer is unreachable and safe to free here.
    if (releaseFn_ && oldFn && oldFn != fnPtr)
      releaseFn_(releaseCtx_, oldFn);
    return EJitPublishStatus::Published;
  }

  EJitCacheEntry Entry;
  Entry.funcIndex = funcIndex;
  Entry.numDims = numDims;
  Entry.fnPtr = reinterpret_cast<uintptr_t>(fnPtr);
  for (uint32_t i = 0; i < numDims; ++i) {
    Entry.dims[i] = dims[i];
    Entry.versions[i] = versions[i];
  }
  B.entries[key].push_back(Entry);
  B.lock.writeRelease();
  return EJitPublishStatus::Published;
}

void EJitTaskPoolCache::retireCode(void *fnPtr) {
  // Retire a never-published pointer (rejected stale compile) through the real
  // release callback only; never fabricate a physical free.
  if (releaseFn_ && fnPtr)
    releaseFn_(releaseCtx_, fnPtr);
}

void EJitTaskPoolCache::releaseRead(uint32_t bucketIndex) {
  if (bucketIndex >= kBuckets)
    return;
  buckets_[bucketIndex].lock.readRelease();
}

uint32_t EJitTaskPoolCache::readyCount() const {
  uint32_t count = 0;
  for (uint32_t i = 0; i < kBuckets; ++i) {
    while (!const_cast<Bucket &>(buckets_[i]).lock.tryRead()) {
    }
    for (const auto &KV : buckets_[i].entries)
      count += static_cast<uint32_t>(KV.second.size());
    const_cast<Bucket &>(buckets_[i]).lock.readRelease();
  }
  return count;
}

void EJitTaskPoolCache::shutdown() {
  // Acquire each bucket's write lock (draining readers to 0) before collecting
  // and clearing, so no reader is mid-use when its entry/container is torn
  // down. Callers must have stopped the worker and returned outstanding read
  // tokens, else write() spins waiting for readers_ to reach 0.
  //
  // The release callbacks are deferred until EVERY bucket lock has been
  // dropped: a callback may re-enter the code pool / ORC / allocator / platform
  // and must never run inside a bucket critical section. Each distinct live
  // fnPtr is released exactly once (dedup below) through the real callback (a
  // logical drop when no callback is installed — never a fabricated free).
  std::vector<void *> toRelease;
  for (uint32_t i = 0; i < kBuckets; ++i) {
    Bucket &B = buckets_[i];
    B.lock.write();
    if (releaseFn_) {
      for (auto &KV : B.entries)
        for (EJitCacheEntry &E : KV.second) {
          void *fn = reinterpret_cast<void *>(E.fnPtr);
          if (fn) {
            toRelease.push_back(fn);
            E.fnPtr = 0; // Guard against re-collecting within this sweep.
          }
        }
    }
    B.entries.clear();
    B.lock.writeRelease();
  }
  // De-duplicate so a pointer published under several identities is released at
  // most once, then release lock-free now that all buckets are unlocked.
  std::sort(toRelease.begin(), toRelease.end());
  toRelease.erase(std::unique(toRelease.begin(), toRelease.end()),
                  toRelease.end());
  for (void *fn : toRelease)
    releaseFn_(releaseCtx_, fn);
}

EJitTaskPool::EJitTaskPool(uint32_t queueCapacity, bool autoStartWorker)
    : queue_(queueCapacity), cache_(switch_), worker_(*this),
      autoStartWorker_(autoStartWorker) {
  if (autoStartWorker_)
    startWorker();
}

EJitTaskPool::~EJitTaskPool() {
  // Stop the single consumer first so no worker can publish while we tear the
  // cache down, then drain each bucket (write lock waits readers → 0) and
  // clear.
  stopWorker();
  cache_.shutdown();
}

EJitTaskPool::CompileOrGetResult
EJitTaskPool::compileOrGet(uint32_t funcIndex, const EJitDimPair *dims,
                           uint32_t numDims, void *fallback) {
  CompileOrGetResult R;
  R.fnPtr = fallback;
  EJIT_DIAG("taskpool request func=%u dims=%u fallback=%p", funcIndex, numDims,
            fallback);

  // 1. Parameter check.
  if ((numDims > 0 && !dims) || numDims > 4) {
    EJIT_DIAG("taskpool reject func=%u: invalid dims ptr=%p count=%u",
              funcIndex, dims, numDims);
    R.status = EJitCompileOrGetStatus::InvalidParam;
    return R;
  }

  // 2. Instance-enabled check (§5.2 step 0) — a disabled instance falls back
  //    and never reaches the cache, so it is never served a stale cached JIT.
  for (uint32_t i = 0; i < numDims; ++i) {
    if (!switch_.isInstanceEnabled(dims[i].dimType, dims[i].instanceId)) {
      counters_.instanceDisabled.fetchAdd(1);
      EJIT_DIAG("taskpool disabled func=%u dim[%u]=(%u,%u)", funcIndex, i,
                dims[i].dimType, dims[i].instanceId);
      R.status = EJitCompileOrGetStatus::InstanceDisabled;
      return R;
    }
  }

  // 3. Cache lookup (§5.2 step 1) — runs BEFORE the Off check, so an already
  //    compiled entry is still served while the pool is globally Off (the spec
  //    orders cache lookup ahead of the Off check). A disabled instance was
  //    already rejected in step 2 and never reaches here.
  EJitCacheLookupResult Hit = cache_.lookup(funcIndex, dims, numDims);
  if (Hit.hasReadToken && Hit.fnPtr) {
    counters_.cacheHits.fetchAdd(1);
    R.status = EJitCompileOrGetStatus::CacheHit;
    R.fnPtr = Hit.fnPtr;
    R.bucketIndex = Hit.bucketIndex;
    R.hasReadToken = true;
    EJIT_DIAG("taskpool hit func=%u bucket=%u fn=%p", funcIndex,
              Hit.bucketIndex, Hit.fnPtr);
    return R;
  }

  // 4. Off mode (§5.2 step 2) — fall back, never enqueue/compile.
  if (switch_.getMode() == EJitCompileMode::Off) {
    EJIT_DIAG("taskpool fallback func=%u: mode off", funcIndex);
    R.status = EJitCompileOrGetStatus::OffMode;
    return R;
  }

  // 5. Dedup + enqueue (§5.2 step 3).
  EJitCompileRequest Req{};
  Req.funcIndex = funcIndex;
  Req.numDims = numDims;
  Req.fallbackPtr = reinterpret_cast<uintptr_t>(fallback);
  for (uint32_t i = 0; i < numDims; ++i) {
    Req.dims[i] = dims[i];
    Req.versions[i] =
        switch_.getInstanceVersion(dims[i].dimType, dims[i].instanceId);
  }

  EJitTaskQueue::EnqueueResult EQ = queue_.tryEnqueue(Req);
  if (EQ == EJitTaskQueue::EnqueueResult::Enqueued) {
    counters_.asyncEnqueues.fetchAdd(1);
    EJIT_DIAG("taskpool enqueued func=%u", funcIndex);
    R.status = EJitCompileOrGetStatus::EnqueuedPending;
    return R;
  }
  if (EQ == EJitTaskQueue::EnqueueResult::AlreadyPending) {
    counters_.alreadyPending.fetchAdd(1);
    EJIT_DIAG("taskpool coalesced func=%u: already pending", funcIndex);
    R.status = EJitCompileOrGetStatus::AlreadyPending;
    return R;
  }
  if (EQ == EJitTaskQueue::EnqueueResult::InvalidFuncIndex) {
    // funcIndex out of the flat dedup table's range: reject, never alias.
    EJIT_DIAG("taskpool reject func=%u: out of range", funcIndex);
    R.status = EJitCompileOrGetStatus::InvalidParam;
    return R;
  }

  counters_.queueFull.fetchAdd(1);
  EJIT_DIAG("taskpool fallback func=%u: queue full", funcIndex);
  R.status = EJitCompileOrGetStatus::QueueFullFallback;
  return R;
}

bool EJitTaskPool::versionsMatch(const EJitCompileRequest &req) const {
  for (uint32_t i = 0; i < req.numDims; ++i) {
    if (req.versions[i] !=
        switch_.getInstanceVersion(req.dims[i].dimType, req.dims[i].instanceId))
      return false;
  }
  return true;
}

void EJitTaskPool::runCompile(const EJitCompileRequest &req) {
  EJIT_DIAG("worker compile begin func=%u dims=%u", req.funcIndex, req.numDims);
  // Checkpoint 1 (§5.3): drop a request invalidated before compilation started.
  if (!versionsMatch(req)) {
    queue_.release(req.funcIndex);
    counters_.compileFailed.fetchAdd(1);
    EJIT_DIAG("worker compile drop func=%u: version changed before compile",
              req.funcIndex);
    return;
  }

  void *fn = nullptr;
  bool ok = compileFn_ && compileFn_(compileCtx_, req, &fn);

  if (!ok || !fn) {
    queue_.release(req.funcIndex);
    counters_.compileFailed.fetchAdd(1);
    EJIT_DIAG("worker compile failed func=%u ok=%u fn=%p", req.funcIndex,
              static_cast<unsigned>(ok), fn);
    return;
  }

  // Checkpoint 2 (§5.3): a toggle during compilation invalidates the result.
  if (!versionsMatch(req)) {
    cache_.retireCode(fn); // Retire the now-stale code (real callback only).
    queue_.release(req.funcIndex);
    counters_.compileFailed.fetchAdd(1);
    EJIT_DIAG("worker compile drop func=%u: version changed after compile",
              req.funcIndex);
    return;
  }

  // Commit gate (§5.3/§5.4): publish re-checks the version snapshot under the
  // bucket write lock. A toggle that slips between checkpoint 2 and the write
  // lock is caught there, so stale code is never stamped with a new version.
  EJitPublishStatus PS =
      cache_.publish(req.funcIndex, req.dims, req.numDims, req.versions, fn);
  switch (PS) {
  case EJitPublishStatus::Published:
    counters_.asyncCompiles.fetchAdd(1);
    queue_.release(req.funcIndex);
    EJIT_DIAG("worker publish ok func=%u fn=%p", req.funcIndex, fn);
    return;
  case EJitPublishStatus::VersionMismatch:
    // Rejected at the commit gate: retire the stale code, do not overwrite any
    // existing entry, release the dedup slot, count as a (cancelled) failure.
    cache_.retireCode(fn);
    queue_.release(req.funcIndex);
    counters_.compileFailed.fetchAdd(1);
    EJIT_DIAG("worker publish drop func=%u: version mismatch", req.funcIndex);
    return;
  case EJitPublishStatus::InvalidParam:
  case EJitPublishStatus::Failed:
    cache_.retireCode(fn);
    queue_.release(req.funcIndex);
    counters_.publishFailed.fetchAdd(1);
    EJIT_DIAG("worker publish failed func=%u status=%u", req.funcIndex,
              static_cast<unsigned>(PS));
    return;
  }
}

bool EJitTaskPool::pollOne() {
  EJitCompileRequest Req{};
  if (!queue_.tryDequeue(Req))
    return false;
  runCompile(Req);
  return true;
}

unsigned EJitTaskPool::pollBudget(unsigned maxItems) {
  unsigned N = 0;
  while (N < maxItems && pollOne())
    ++N;
  return N;
}

bool EJitTaskPool::startWorker() { return worker_.start(); }

void EJitTaskPool::stopWorker() { worker_.stop(); }

bool EJitTaskPool::isWorkerRunning() const { return worker_.isRunning(); }

void EJitTaskPool::getStats(EJitTaskPoolStatsSnapshot &out) const {
  out.cacheHits = counters_.cacheHits.loadRelaxed();
  out.asyncCompiles = counters_.asyncCompiles.loadRelaxed();
  out.asyncEnqueues = counters_.asyncEnqueues.loadRelaxed();
  out.alreadyPending = counters_.alreadyPending.loadRelaxed();
  out.queueFull = counters_.queueFull.loadRelaxed();
  out.compileFailed = counters_.compileFailed.loadRelaxed();
  out.publishFailed = counters_.publishFailed.loadRelaxed();
  out.instanceDisabled = counters_.instanceDisabled.loadRelaxed();
  out.readyEntries = cache_.readyCount();
  out.pendingEntries = queue_.inFlightCount();
  out.queueApproxSize = queue_.queueDepth();
}
