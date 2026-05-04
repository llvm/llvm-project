//===-- EJitCache.cpp - EmbeddedJIT Code Cache ----------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitCache.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <chrono>

using namespace llvm::ejit;

EJitCache::EJitCache(size_t maxEntries, size_t maxTotalSize,
                     size_t maxSingleFuncSize)
    : maxEntries_(maxEntries), maxTotalSize_(maxTotalSize),
      maxSingleFuncSize_(maxSingleFuncSize) {}

void *EJitCache::getOrNull(const std::string &cacheKey) {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = cache_.find(cacheKey);
  if (it == cache_.end()) {
    misses_++;
    return nullptr;
  }

  hits_++;
  it->second.lastAccessTime =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();

  // Move to front of LRU list
  auto lruIt = lruIter_.find(cacheKey);
  if (lruIt != lruIter_.end()) {
    lruList_.splice(lruList_.begin(), lruList_, lruIt->second);
    lruIter_[cacheKey] = lruList_.begin();
  }

  return it->second.funcPtr;
}

bool EJitCache::put(const std::string &cacheKey, void *funcPtr,
                    size_t codeSize,
                    const std::set<std::string> &periodDeps) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (codeSize > maxSingleFuncSize_)
    return false;

  // Evict if needed
  while (!cache_.empty() &&
         (cache_.size() >= maxEntries_ ||
          currentTotalSize_ + codeSize > maxTotalSize_)) {
    evictLRU();
  }

  uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
                     .count();

  Entry entry{cacheKey, funcPtr, codeSize, now, periodDeps};
  cache_[cacheKey] = entry;
  currentTotalSize_ += codeSize;

  // Add to LRU front
  lruList_.push_front(cacheKey);
  lruIter_[cacheKey] = lruList_.begin();

  // Index by period dependencies
  for (const auto &dep : periodDeps)
    periodIndex_[dep].insert(cacheKey);

  return true;
}

void EJitCache::invalidateByPeriod(const std::string &periodName,
                                   unsigned cellIdx) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  std::string dep = periodName + "=" + std::to_string(cellIdx);

  auto it = periodIndex_.find(dep);
  if (it == periodIndex_.end())
    return;

  for (const auto &key : it->second) {
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
  std::unique_lock<std::shared_mutex> lock(mutex_);
  cache_.clear();
  lruList_.clear();
  lruIter_.clear();
  periodIndex_.clear();
  currentTotalSize_ = 0;
}

EJitCache::Stats EJitCache::getStats() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  Stats s;
  s.entryCount = cache_.size();
  s.totalCodeSize = currentTotalSize_;
  s.maxSize = maxTotalSize_;
  s.hits = hits_;
  s.misses = misses_;
  s.evictions = evictions_;
  return s;
}

std::string EJitCache::buildCacheKey(
    const std::string &fnName,
    const std::pair<std::string, unsigned> *dims, unsigned count) {
  if (count == 0)
    return fnName;

  // Sort by periodName for deterministic keys
  llvm::SmallVector<std::pair<std::string, unsigned>, 4> sorted(dims, dims + count);
  std::sort(sorted.begin(), sorted.end());

  std::string key = fnName;
  for (unsigned i = 0; i < count; ++i) {
    key += "|";
    key += sorted[i].first;
    key += "=";
    key += std::to_string(sorted[i].second);
  }
  return key;
}

void EJitCache::evictLRU() {
  if (lruList_.empty())
    return;

  const std::string &key = lruList_.back();
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    currentTotalSize_ -= it->second.codeSize;

    // Clean up period dependencies
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
