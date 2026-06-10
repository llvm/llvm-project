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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
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

/// Thread-safe LRU code cache. Uses iterator embedding in Entry to keep
/// LRU maintenance at a single hash lookup per getOrNull (no reverse map).
/// Cache key: uint64_t = funcIdx(32b) | dim3(8b) | dim2(8b) | dim1(8b) | dim0(8b)
class EJitCache {
public:
#ifdef EJIT_FREESTANDING
  using MutexType = BareMetalMutex;
#else
  using MutexType = std::shared_mutex;
#endif
  using LruList = std::list<uint64_t>;

  struct Entry {
    void *funcPtr;
    size_t codeSize;
    LruList::iterator lruIt;           // embedded iterator → O(1) splice/erase
    SmallVector<std::string, 4> periodDeps;
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

  /// Look up a cache entry. Single hash find + O(1) LRU bump via embedded iterator.
  void *getOrNull(uint64_t cacheKey);

  /// Insert a compiled function into the cache.
  bool put(uint64_t cacheKey, void *funcPtr,
           size_t codeSize,
           ArrayRef<std::string> periodDeps = {});

  /// Invalidate all entries that depend on a specific period/cell.
  void invalidateByPeriod(const std::string &periodName, uint8_t cellIdx);

  /// Clear all cached entries.
  void clear();

  Stats getStats() const;

  /// Build cache key: upper 32 bits = funcIdx, lower 32 bits = 4x8-bit dims.
  static uint64_t buildCacheKey(uint32_t funcIdx,
      const std::pair<std::string, uint8_t> *dims, unsigned count);

private:
  void evictLRU();

  mutable MutexType mutex_;
  std::unordered_map<uint64_t, Entry> cache_;
  LruList lruList_;                                     // uint64_t keys in LRU order
  std::unordered_map<std::string, std::set<uint64_t>> periodIndex_;

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
