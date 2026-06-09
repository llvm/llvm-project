//===-- EJitCache.cpp - EmbeddedJIT Code Cache ----------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitCache.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <mutex>
#include <shared_mutex>
#ifndef EJIT_FREESTANDING
#include <chrono>
#endif

using namespace llvm::ejit;

namespace {

#ifdef EJIT_FREESTANDING
uint64_t getTimestamp() { return 0; }
#else
uint64_t getTimestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
#endif

} // namespace

EJitCache::EJitCache(size_t maxEntries, size_t maxTotalSize,
                     size_t maxSingleFuncSize)
    : maxEntries_(maxEntries), maxTotalSize_(maxTotalSize),
      maxSingleFuncSize_(maxSingleFuncSize) {}

void *EJitCache::getOrNull(uint32_t cacheKey) {
  std::shared_lock<decltype(mutex_)> lock(mutex_);
  auto it = cache_.find(cacheKey);
  if (it == cache_.end()) {
    misses_++;
    return nullptr;
  }

  hits_++;
  it->second.lastAccessTime = getTimestamp();

  auto lruIt = lruIter_.find(cacheKey);
  if (lruIt != lruIter_.end()) {
    lruList_.splice(lruList_.begin(), lruList_, lruIt->second);
    lruIter_[cacheKey] = lruList_.begin();
  }

  return it->second.funcPtr;
}

bool EJitCache::put(uint32_t cacheKey, void *funcPtr,
                    size_t codeSize,
                    ArrayRef<std::string> periodDeps) {
  std::unique_lock<decltype(mutex_)> lock(mutex_);

  if (codeSize > maxSingleFuncSize_)
    return false;

  // If the same cacheKey already exists, clean up the old entry before
  // inserting the new one. This handles the rare race where two async
  // compilations produce the same key.
  auto [it, inserted] = cache_.try_emplace(cacheKey);
  if (!inserted) {
    currentTotalSize_ -= it->second.codeSize;
    // Remove old LRU node so the key doesn't appear twice in the list.
    auto lruIt = lruIter_.find(cacheKey);
    if (lruIt != lruIter_.end()) {
      lruList_.erase(lruIt->second);
      lruIter_.erase(lruIt);
    }
    // Remove old period dependencies from the index.
    for (const auto &dep : it->second.periodDeps) {
      auto pit = periodIndex_.find(dep);
      if (pit != periodIndex_.end()) {
        pit->second.erase(cacheKey);
        if (pit->second.empty())
          periodIndex_.erase(pit);
      }
    }
  }

  while (!cache_.empty() &&
         (cache_.size() >= maxEntries_ ||
          currentTotalSize_ + codeSize > maxTotalSize_))
    evictLRU();

  it->second = {funcPtr, codeSize, getTimestamp(),
                SmallVector<std::string, 4>(periodDeps.begin(),
                                            periodDeps.end())};
  currentTotalSize_ += codeSize;

  lruList_.push_front(cacheKey);
  lruIter_[cacheKey] = lruList_.begin();

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

  for (uint32_t key : it->second) {
    auto cacheIt = cache_.find(key);
    if (cacheIt != cache_.end()) {
      currentTotalSize_ -= cacheIt->second.codeSize;
      cache_.erase(cacheIt);
    }
    auto lruIt = lruIter_.find(key);
    if (lruIt != lruIter_.end()) {
      lruList_.erase(lruIt->second);
      lruIter_.erase(lruIt);
    }
  }
  periodIndex_.erase(it);
}

void EJitCache::clear() {
  std::unique_lock<decltype(mutex_)> lock(mutex_);
  cache_.clear();
  lruList_.clear();
  lruIter_.clear();
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

uint32_t EJitCache::buildCacheKey(
    uint16_t funcIdx,
    const std::pair<std::string, uint8_t> *dims, unsigned count) {
  uint32_t key = static_cast<uint32_t>(funcIdx) << 16;
  for (unsigned i = 0; i < count && i < 4; ++i)
    key |= (static_cast<uint32_t>(dims[i].second) & 0xF) << (i * 4);
  return key;
}

void EJitCache::evictLRU() {
  if (lruList_.empty())
    return;

  uint32_t key = lruList_.back();
  auto it = cache_.find(key);
  if (it != cache_.end()) {
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
  }

  lruIter_.erase(key);
  lruList_.pop_back();
  evictions_++;
}
