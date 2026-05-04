//===-- EJitCache.h - EmbeddedJIT Code Cache ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITCACHE_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITCACHE_H

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace llvm {
namespace ejit {

/// Thread-safe LRU code cache with size and entry limits.
/// Cache key format: "fnName|period1=idx1,period2=idx2,..."
class EJitCache {
public:
  struct Entry {
    std::string cacheKey;
    void *funcPtr;
    size_t codeSize;
    uint64_t lastAccessTime;
    std::set<std::string> periodDeps;
  };

  struct Stats {
    size_t entryCount = 0;
    size_t totalCodeSize = 0;
    size_t maxSize = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
  };

  EJitCache(size_t maxEntries = 256, size_t maxTotalSize = 32 * 1024 * 1024,
            size_t maxSingleFuncSize = 512 * 1024);

  /// Look up a cache entry. Returns nullptr on miss.
  void *getOrNull(const std::string &cacheKey);

  /// Insert a compiled function into the cache.
  bool put(const std::string &cacheKey, void *funcPtr,
           size_t codeSize, const std::set<std::string> &periodDeps = {});

  /// Invalidate all entries that depend on a specific period/cell.
  void invalidateByPeriod(const std::string &periodName, unsigned cellIdx);

  /// Clear all cached entries.
  void clear();

  Stats getStats() const;

  /// Build a deterministic cache key from function name and dimension array.
  static std::string buildCacheKey(const std::string &fnName,
      const std::pair<std::string, unsigned> *dims, unsigned count);

private:
  void evictLRU();

  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, Entry> cache_;
  std::list<std::string> lruList_;              // front = most recent
  std::unordered_map<std::string, std::list<std::string>::iterator> lruIter_;
  std::unordered_map<std::string, std::set<std::string>> periodIndex_;

  size_t maxEntries_;
  size_t maxTotalSize_;
  size_t maxSingleFuncSize_;
  size_t currentTotalSize_ = 0;

  mutable uint64_t hits_ = 0;
  mutable uint64_t misses_ = 0;
  uint64_t evictions_ = 0;
};

} // namespace ejit
} // namespace llvm

#endif
