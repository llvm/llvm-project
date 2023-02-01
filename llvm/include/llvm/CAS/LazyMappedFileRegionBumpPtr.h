//===- LazyMappedFileRegionBumpPtr.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_LAZYMAPPEDFILEREGIONBUMPPTR_H
#define LLVM_CAS_LAZYMAPPEDFILEREGIONBUMPPTR_H

#include "llvm/CAS/LazyMappedFileRegion.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/FileSystem.h"
#include <atomic>

namespace llvm {

class MemoryBuffer;

namespace cas {

/// Allocator for a lazy mapped file region.
///
/// Provides 8-byte alignment for all allocations.
class LazyMappedFileRegionBumpPtr {
public:
  /// Minimum alignment for allocations, currently hardcoded to 8B.
  static constexpr Align getAlign() {
    // Trick Align into giving us '8' as a constexpr.
    struct alignas(8) T {};
    static_assert(alignof(T) == 8, "Tautology failed?");
    return Align::Of<T>();
  }

  LazyMappedFileRegionBumpPtr() = delete;
  LazyMappedFileRegionBumpPtr(LazyMappedFileRegion &LMFR, int64_t BumpPtrOffset)
      : LMFR(LMFR) {
    initialize(BumpPtrOffset);
  }
  LazyMappedFileRegionBumpPtr(std::shared_ptr<LazyMappedFileRegion> LMFR,
                              int64_t BumpPtrOffset)
      : LMFR(*LMFR), OwnedLMFR(std::move(LMFR)) {
    initialize(BumpPtrOffset);
  }

  /// Allocate at least \p AllocSize. Rounds up to \a getAlign().
  char *allocate(uint64_t AllocSize) {
    return data() + allocateOffset(AllocSize);
  }
  /// Allocate, returning the offset from \a data() instead of a pointer.
  int64_t allocateOffset(uint64_t AllocSize);

  char *data() const { return LMFR.data(); }
  uint64_t size() const { return *BumpPtr; }
  uint64_t capacity() const { return LMFR.capacity(); }

  LazyMappedFileRegion &getRegion() const { return LMFR; }

private:
  void initialize(int64_t BumpPtrOffset);

  LazyMappedFileRegion &LMFR;
  std::shared_ptr<LazyMappedFileRegion> OwnedLMFR;
  std::atomic<int64_t> *BumpPtr = nullptr;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_LAZYMAPPEDFILEREGIONBUMPPTR_H
