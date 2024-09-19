//===------ AutoLoadDylibUtils.h - Auto-Loading Dynamic Library -------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_AUTOLOADDYLIBUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_AUTOLOADDYLIBUTILS_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"

#include <math.h>
#include <type_traits>
#include <vector>

namespace llvm {
namespace orc {

namespace shared {
using SPSBloomFilter =
    SPSTuple<bool, uint32_t, uint32_t, uint32_t, SPSSequence<uint64_t>>;
}

constexpr uint32_t log2u(std::uint32_t n) {
  return (n > 1) ? 1 + log2u(n >> 1) : 0;
}

class BloomFilter {
private:
  static constexpr int Bits = 8 * sizeof(uint64_t);
  static constexpr float P = 0.02f;

  bool Initialized = false;
  uint32_t SymbolsCount = 0;
  uint32_t BloomSize = 0;
  uint32_t BloomShift = 0;
  std::vector<uint64_t> BloomTable;

  // This is a GNU implementation of hash used in bloom filter!
  static uint32_t GNUHash(StringRef S) {
    uint32_t H = 5381;
    for (uint8_t C : S)
      H = (H << 5) + H + C;
    return H;
  }
  // Helper method for hash testing
  bool TestHash(uint32_t hash) const {
    assert(IsInitialized && "Bloom filter is not initialized!");
    uint32_t hash2 = hash >> BloomShift;
    uint32_t n = (hash >> log2u(Bits)) % BloomSize;
    uint64_t mask = ((1ULL << (hash % Bits)) | (1ULL << (hash2 % Bits)));
    return (mask & BloomTable[n]) == mask;
  }

  // Helper method to add a hash
  void AddHash(uint32_t hash) {
    assert(IsInitialized && "Bloom filter is not initialized!");
    uint32_t hash2 = hash >> BloomShift;
    uint32_t n = (hash >> log2u(Bits)) % BloomSize;
    uint64_t mask = ((1ULL << (hash % Bits)) | (1ULL << (hash2 % Bits)));
    BloomTable[n] |= mask;
  }

  // Resizes the Bloom filter table based on symbol count
  void ResizeTable(uint32_t newSymbolsCount) {
    assert(SymbolsCount == 0 && "Resize not supported after initialization!");
    SymbolsCount = newSymbolsCount;
    BloomSize =
        static_cast<uint32_t>(ceil((-1.44f * SymbolsCount * log2f(P)) / Bits));
    BloomShift = std::min(6u, log2u(SymbolsCount));
    BloomTable.resize(BloomSize, 0);
  }

  friend class shared::SPSSerializationTraits<shared::SPSBloomFilter,
                                              BloomFilter>;

public:
  BloomFilter() = default;
  BloomFilter(const BloomFilter &other) noexcept
      : Initialized(other.Initialized), SymbolsCount(other.SymbolsCount),
        BloomSize(other.BloomSize), BloomShift(other.BloomShift),
        BloomTable(other.BloomTable) {
  }
  BloomFilter &operator=(const BloomFilter &other) = delete;

  BloomFilter(BloomFilter &&other) noexcept
      : Initialized(other.Initialized), SymbolsCount(other.SymbolsCount),
        BloomSize(other.BloomSize), BloomShift(other.BloomShift),
        BloomTable(std::move(other.BloomTable)) {
    other.Initialized = false;
    other.SymbolsCount = 0;
    other.BloomSize = 0;
    other.BloomShift = 0;
  }

  BloomFilter &operator=(BloomFilter &&other) noexcept {
    if (this != &other) {
      Initialized = other.Initialized;
      SymbolsCount = other.SymbolsCount;
      BloomSize = other.BloomSize;
      BloomShift = other.BloomShift;
      BloomTable = std::move(other.BloomTable);

      other.Initialized = false;
      other.SymbolsCount = 0;
      other.BloomSize = 0;
      other.BloomShift = 0;
    }
    return *this;
  }

  void swap(BloomFilter &other) noexcept {
    std::swap(Initialized, other.Initialized);
    std::swap(SymbolsCount, other.SymbolsCount);
    std::swap(BloomSize, other.BloomSize);
    std::swap(BloomShift, other.BloomShift);
    std::swap(BloomTable, other.BloomTable);
  }

  void Initialize(uint32_t newSymbolsCount) {
    assert(!Initialized && "Cannot reinitialize the Bloom filter!");
    Initialized = true;
    ResizeTable(newSymbolsCount);
  }

  bool IsEmpty() const { return SymbolsCount == 0; }

  uint32_t getSymCount() const { return SymbolsCount; }

  bool IsInitialized() const { return Initialized; }

  bool MayContain(uint32_t hash) const {
    if (IsEmpty())
      return false;
    return TestHash(hash);
  }

  bool MayContain(StringRef symbol) const {
    return MayContain(GNUHash(symbol));
  }

  void AddSymbol(StringRef symbol) { AddHash(GNUHash(symbol)); }
};

struct ResolveResult {
  std::optional<BloomFilter> Filter;
  std::vector<ExecutorSymbolDef> SymbolDef;
};

namespace shared {

template <> class SPSSerializationTraits<SPSBloomFilter, BloomFilter> {
public:
  static size_t size(const BloomFilter &Filter) {
    return SPSBloomFilter::AsArgList::size(
        Filter.Initialized, Filter.SymbolsCount, Filter.BloomSize,
        Filter.BloomShift, Filter.BloomTable);
  }

  static bool serialize(SPSOutputBuffer &OB, const BloomFilter &Filter) {
    return SPSBloomFilter::AsArgList::serialize(
        OB, Filter.Initialized, Filter.SymbolsCount, Filter.BloomSize,
        Filter.BloomShift, Filter.BloomTable);
  }

  static bool deserialize(SPSInputBuffer &IB, BloomFilter &Filter) {
    bool IsInitialized;
    uint32_t SymbolsCount = 0, BloomSize = 0, BloomShift = 0;
    std::vector<uint64_t> BloomTable;

    if (!SPSBloomFilter::AsArgList::deserialize(
            IB, IsInitialized, SymbolsCount, BloomSize, BloomShift, BloomTable))
      return false;

    Filter.Initialized = IsInitialized;
    Filter.SymbolsCount = SymbolsCount;
    Filter.BloomSize = BloomSize;
    Filter.BloomShift = BloomShift;
    Filter.BloomTable = std::move(BloomTable);

    return true;
  }
};

using SPSResolveResult =
    SPSTuple<SPSOptional<SPSBloomFilter>, SPSSequence<SPSExecutorSymbolDef>>;
template <> class SPSSerializationTraits<SPSResolveResult, ResolveResult> {
public:
  static size_t size(const ResolveResult &Result) {
    return SPSResolveResult::AsArgList::size(Result.Filter, Result.SymbolDef);
  }

  static bool serialize(SPSOutputBuffer &OB, const ResolveResult &Result) {
    return SPSResolveResult::AsArgList::serialize(OB, Result.Filter,
                                                  Result.SymbolDef);
  }

  static bool deserialize(SPSInputBuffer &IB, ResolveResult &Result) {
    return SPSResolveResult::AsArgList::deserialize(IB, Result.Filter,
                                                    Result.SymbolDef);
  }
};

} // end namespace shared
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_AUTOLOADDYLIBUTILS_H