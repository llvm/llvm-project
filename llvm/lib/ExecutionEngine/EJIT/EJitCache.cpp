//===-- EJitCache.cpp - EmbeddedJIT Code Cache ----------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitCache.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <mutex>
#include <shared_mutex>

using namespace llvm::ejit;

EJitCache::EJitCache(size_t maxEntries, size_t maxTotalSize,
                     size_t maxSingleFuncSize)
    : maxEntries_(maxEntries), maxTotalSize_(maxTotalSize),
      maxSingleFuncSize_(maxSingleFuncSize) {}

void *EJitCache::getOrNull(uint64_t cacheKey) {
  // unique_lock is required because splice() writes to the list's internal
  // pointers — a write operation that would race under shared_lock.
  // On bare-metal (BareMetalMutex no-op) this has zero overhead vs shared_lock.
  std::unique_lock<decltype(mutex_)> lock(mutex_);
  auto it = cache_.find(cacheKey);       // single hash lookup
  if (it == cache_.end()) {
    misses_++;
    return nullptr;
  }

  hits_++;
  // O(1) LRU bump via embedded iterator — no second hash lookup
  lruList_.splice(lruList_.begin(), lruList_, it->second.lruIt);
  return it->second.funcPtr;
}

bool EJitCache::put(uint64_t cacheKey, void *funcPtr,
                    size_t codeSize,
                    ArrayRef<std::string> periodDeps) {
  std::unique_lock<decltype(mutex_)> lock(mutex_);

  if (codeSize > maxSingleFuncSize_)
    return false;

  // 1. Remove old entry for the same key if present (re-compilation after
  //    deactivate).  Done first so the entry is never in an intermediate
  //    "in cache_ but not in lruList_" state.
  auto oldIt = cache_.find(cacheKey);
  if (oldIt != cache_.end()) {
    currentTotalSize_ -= oldIt->second.codeSize;
    lruList_.erase(oldIt->second.lruIt);
    for (const auto &dep : oldIt->second.periodDeps) {
      auto pit = periodIndex_.find(dep);
      if (pit != periodIndex_.end()) {
        pit->second.erase(cacheKey);
        if (pit->second.empty())
          periodIndex_.erase(pit);
      }
    }
    cache_.erase(oldIt);
  }

  // 2. Evict LRU entries until there is room for the new entry.
  while (!cache_.empty() &&
         (cache_.size() >= maxEntries_ ||
          currentTotalSize_ + codeSize > maxTotalSize_))
    evictLRU();

  // 3. Insert fully-formed Entry — no orphan window, no re-find.
  lruList_.push_front(cacheKey);
  Entry e{funcPtr, codeSize, lruList_.begin(),
          SmallVector<std::string, 4>(periodDeps.begin(), periodDeps.end())};
  cache_.emplace(cacheKey, std::move(e));
  currentTotalSize_ += codeSize;

  for (const auto &dep : periodDeps)
    periodIndex_[dep].insert(cacheKey);

  return true;
}

void EJitCache::invalidateByPeriod(const std::string &periodName,
                                   uint8_t cellIdx) {
  std::unique_lock<decltype(mutex_)> lock(mutex_);
  std::string dep = periodName + "=" + std::to_string(cellIdx);

  auto it = periodIndex_.find(dep);
  if (it == periodIndex_.end())
    return;

  for (uint64_t key : it->second) {
    auto cacheIt = cache_.find(key);
    if (cacheIt != cache_.end()) {
      currentTotalSize_ -= cacheIt->second.codeSize;
      lruList_.erase(cacheIt->second.lruIt);  // O(1) via embedded iterator
      cache_.erase(cacheIt);
    }
  }
  periodIndex_.erase(it);
}

void EJitCache::clear() {
  std::unique_lock<decltype(mutex_)> lock(mutex_);
  cache_.clear();
  lruList_.clear();
  periodIndex_.clear();
  currentTotalSize_ = 0;
}

EJitCache::Stats EJitCache::getStats() const {
  std::shared_lock<decltype(mutex_)> lock(mutex_);
  Stats s;
  s.entryCount = cache_.size();
  s.totalCodeSize = currentTotalSize_;
  s.maxSize = maxTotalSize_;
  s.hits = hits_;
  s.misses = misses_;
  s.evictions = evictions_;
  return s;
}

uint64_t EJitCache::buildCacheKey(
    uint32_t funcIdx,
    const std::pair<std::string, uint8_t> *dims, unsigned count) {
  uint64_t key = static_cast<uint64_t>(funcIdx) << 32;
  for (unsigned i = 0; i < count && i < 4; ++i)
    key |= static_cast<uint64_t>(dims[i].second) << (i * 8);
  return key;
}

void EJitCache::evictLRU() {
  if (lruList_.empty())
    return;

  // Invariant: every key in lruList_ has a corresponding entry in cache_.
  // This is maintained by put() (insert + push_front together) and
  // invalidateByPeriod() (erase from both together).
  uint64_t key = lruList_.back();
  lruList_.pop_back();

  auto it = cache_.find(key);
  assert(it != cache_.end() && "lruList_/cache_ invariant broken");
  if (it == cache_.end())
    return;

  currentTotalSize_ -= it->second.codeSize;

  for (const auto &dep : it->second.periodDeps) {
    auto pit = periodIndex_.find(dep);
    if (pit != periodIndex_.end()) {
      pit->second.erase(key);
      if (pit->second.empty())
        periodIndex_.erase(pit);
    }
  }

  cache_.erase(it);
  evictions_++;
}
