//===- SymbolFilter.h - Utilities for Symbol Filtering ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_SYMBOLFILTER_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_SYMBOLFILTER_H

#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"

#include <cmath>
#include <type_traits>
#include <vector>

namespace llvm {
namespace orc {

namespace shared {
using SPSBloomFilter =
    SPSTuple<bool, uint32_t, uint32_t, uint32_t, SPSSequence<uint64_t>>;
}

class BloomFilter {
public:
  using HashFunc = std::function<uint32_t(StringRef)>;

  BloomFilter() = default;
  BloomFilter(BloomFilter &&) noexcept = default;
  BloomFilter &operator=(BloomFilter &&) noexcept = default;
  BloomFilter(const BloomFilter &) = delete;
  BloomFilter &operator=(const BloomFilter &) = delete;

  BloomFilter(uint32_t SymbolCount, float FalsePositiveRate, HashFunc hashFn)
      : HashFn(std::move(hashFn)) {
    initialize(SymbolCount, FalsePositiveRate);
  }
  bool isInitialized() const { return Initialized; }

  void add(StringRef Sym) {
    assert(Initialized);
    addHash(HashFn(Sym));
  }

  bool mayContain(StringRef Sym) const {
    return !isEmpty() && testHash(HashFn(Sym));
  }

  bool isEmpty() const { return SymbolCount == 0; }

private:
  friend class shared::SPSSerializationTraits<shared::SPSBloomFilter,
                                              BloomFilter>;
  static constexpr uint32_t BitsPerEntry = 64;

  bool Initialized = false;
  uint32_t SymbolCount = 0;
  uint32_t BloomSize = 0;
  uint32_t BloomShift = 0;
  std::vector<uint64_t> BloomTable;
  HashFunc HashFn;

  void initialize(uint32_t SymCount, float FalsePositiveRate) {
    assert(SymCount > 0);
    SymbolCount = SymCount;
    Initialized = true;

    float ln2 = std::log(2.0f);
    float M = -1.0f * SymbolCount * std::log(FalsePositiveRate) / (ln2 * ln2);
    BloomSize = static_cast<uint32_t>(std::ceil(M / BitsPerEntry));
    BloomShift = std::min(6u, log2ceil(SymbolCount));
    BloomTable.resize(BloomSize, 0);
  }

  void addHash(uint32_t Hash) {
    uint32_t Hash2 = Hash >> BloomShift;
    uint32_t N = (Hash / BitsPerEntry) % BloomSize;
    uint64_t Mask =
        (1ULL << (Hash % BitsPerEntry)) | (1ULL << (Hash2 % BitsPerEntry));
    BloomTable[N] |= Mask;
  }

  bool testHash(uint32_t Hash) const {
    uint32_t Hash2 = Hash >> BloomShift;
    uint32_t N = (Hash / BitsPerEntry) % BloomSize;
    uint64_t Mask =
        (1ULL << (Hash % BitsPerEntry)) | (1ULL << (Hash2 % BitsPerEntry));
    return (BloomTable[N] & Mask) == Mask;
  }

  static constexpr uint32_t log2ceil(uint32_t V) {
    return V <= 1 ? 0 : 32 - countl_zero(V - 1);
  }
};

class BloomFilterBuilder {
public:
  using HashFunc = BloomFilter::HashFunc;

  BloomFilterBuilder() = default;

  BloomFilterBuilder &setFalsePositiveRate(float Rate) {
    assert(Rate > 0.0f && Rate < 1.0f);
    FalsePositiveRate = Rate;
    return *this;
  }

  BloomFilterBuilder &setHashFunction(HashFunc Fn) {
    HashFn = std::move(Fn);
    return *this;
  }

  BloomFilter build(ArrayRef<StringRef> Symbols) const {
    assert(!Symbols.empty() && "Cannot build filter from empty symbol list.");
    BloomFilter F(static_cast<uint32_t>(Symbols.size()), FalsePositiveRate,
                  HashFn);
    for (const auto &Sym : Symbols)
      F.add(Sym);

    return F;
  }

private:
  float FalsePositiveRate = 0.02f;
  HashFunc HashFn = [](StringRef S) -> uint32_t {
    uint32_t H = 5381;
    for (char C : S)
      H = ((H << 5) + H) + static_cast<uint8_t>(C); // H * 33 + C
    return H;
  };
};

namespace shared {

template <> class SPSSerializationTraits<SPSBloomFilter, BloomFilter> {
public:
  static size_t size(const BloomFilter &Filter) {
    return SPSBloomFilter::AsArgList::size(
        Filter.Initialized, Filter.SymbolCount, Filter.BloomSize,
        Filter.BloomShift, Filter.BloomTable);
  }

  static bool serialize(SPSOutputBuffer &OB, const BloomFilter &Filter) {
    return SPSBloomFilter::AsArgList::serialize(
        OB, Filter.Initialized, Filter.SymbolCount, Filter.BloomSize,
        Filter.BloomShift, Filter.BloomTable);
  }

  static bool deserialize(SPSInputBuffer &IB, BloomFilter &Filter) {
    bool IsInitialized;
    uint32_t SymbolCount = 0, BloomSize = 0, BloomShift = 0;
    std::vector<uint64_t> BloomTable;

    if (!SPSBloomFilter::AsArgList::deserialize(
            IB, IsInitialized, SymbolCount, BloomSize, BloomShift, BloomTable))
      return false;

    Filter.Initialized = IsInitialized;
    Filter.SymbolCount = SymbolCount;
    Filter.BloomSize = BloomSize;
    Filter.BloomShift = BloomShift;
    Filter.BloomTable = std::move(BloomTable);

    return true;
  }
};

} // end namespace shared
} // end namespace orc
} // end namespace llvm
#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SYMBOLFILTER_H
