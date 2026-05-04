//===-- EJitJITLinkMemoryManager.h - Embedded Memory Manager ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITJITLINKMEMORYMANAGER_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITJITLINKMEMORYMANAGER_H

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/Support/Allocator.h"
#include <cstddef>
#include <mutex>

namespace llvm {
namespace ejit {

/// Fixed-slab memory manager for embedded scenarios.
/// Uses bump allocation - individual deallocations are not supported;
/// the entire slab is reset on cache clear.
class EJitJITLinkMemoryManager : public jitlink::JITLinkMemoryManager {
public:
  EJitJITLinkMemoryManager(size_t maxCodeSize = 384 * 1024,
                           size_t maxDataSize = 128 * 1024);

  ~EJitJITLinkMemoryManager();

  void allocate(const jitlink::JITLinkDylib *JD, jitlink::LinkGraph &G,
                OnAllocatedFunction OnAllocated) override;

  void deallocate(std::vector<FinalizedAlloc> Allocs,
                  OnDeallocatedFunction OnDeallocated) override;

  size_t getCurrentCodeUsage() const;
  size_t getCurrentDataUsage() const;

  /// Reset both slabs (used when clearing the entire cache).
  void reset();

private:
  struct SlabRegion {
    void *baseAddr;
    size_t totalSize;
    size_t usedSize;
    std::mutex mutex;
  };

  SlabRegion codeSlab_;
  SlabRegion dataSlab_;
};

} // namespace ejit
} // namespace llvm

#endif
