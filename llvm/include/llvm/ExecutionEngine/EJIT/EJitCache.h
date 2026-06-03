//===-- EJitCache.h - EmbeddedJIT Code Cache ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITCACHE_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITCACHE_H

#include "llvm/ExecutionEngine/EJIT/EJitBareMetal.h"
#include <cstddef>
#include <cstdint>
#include <list>
#include <set>
#include <string>
#include <unordered_map>
#ifndef EJIT_FREESTANDING
#include <mutex>
#include <shared_mutex>
#endif

namespace llvm {
namespace ejit {

/// Thread-safe LRU code cache with size and entry limits.
/// Cache key: uint32_t = funcIdx(16b) | dim3(4b) | dim2(4b) | dim1(4b) | dim0(4b)
class EJitCache {
public:
#ifdef EJIT_FREESTANDING
  using MutexType = BareMetalMutex;
#else
  using MutexType = std::shared_mutex;
#endif
  struct Entry {
    uint32_t cacheKey;
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
  void *getOrNull(uint32_t cacheKey);

  /// Insert a compiled function into the cache.
  bool put(uint32_t cacheKey, void *funcPtr,
           size_t codeSize, const std::set<std::string> &periodDeps = {});

  /// Invalidate all entries that depend on a specific period/cell.
  void invalidateByPeriod(const std::string &periodName, uint8_t cellIdx);

  /// Clear all cached entries.
  void clear();

  Stats getStats() const;

  /// Build cache key: upper 16 bits = funcIdx, lower 16 bits = 4x4-bit dims.
  static uint32_t buildCacheKey(uint16_t funcIdx,
      const std::pair<std::string, uint8_t> *dims, unsigned count);

private:
  void evictLRU();

  mutable MutexType mutex_;
  std::unordered_map<uint32_t, Entry> cache_;
  std::list<uint32_t> lruList_;
  std::unordered_map<uint32_t, std::list<uint32_t>::iterator> lruIter_;
  std::unordered_map<std::string, std::set<uint32_t>> periodIndex_;

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
