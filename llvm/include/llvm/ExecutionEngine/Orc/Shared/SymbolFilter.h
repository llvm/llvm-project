//===--- SymbolFilter.h - Utils for Symbol Filter ---*- C++ -*-===//
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

  BloomFilter(uint32_t symbolCount, float falsePositiveRate, HashFunc hashFn)
      : hashFunc(std::move(hashFn)) {
    initialize(symbolCount, falsePositiveRate);
  }
  bool IsInitialized() const { return initialized; }

  void add(StringRef symbol) {
    assert(initialized);
    addHash(hashFunc(symbol));
  }

  bool mayContain(StringRef symbol) const {
    return !isEmpty() && testHash(hashFunc(symbol));
  }

  bool isEmpty() const { return symbolCount_ == 0; }

private:
  friend class shared::SPSSerializationTraits<shared::SPSBloomFilter,
                                              BloomFilter>;
  static constexpr uint32_t bitsPerEntry = 64;

  bool initialized = false;
  uint32_t symbolCount_ = 0;
  uint32_t bloomSize = 0;
  uint32_t bloomShift = 0;
  std::vector<uint64_t> bloomTable;
  HashFunc hashFunc;

  void initialize(uint32_t symbolCount, float falsePositiveRate) {
    assert(symbolCount > 0);
    symbolCount_ = symbolCount;
    initialized = true;

    float ln2 = std::log(2.0f);
    float m = -1.0f * symbolCount * std::log(falsePositiveRate) / (ln2 * ln2);
    bloomSize = static_cast<uint32_t>(std::ceil(m / bitsPerEntry));
    bloomShift = std::min(6u, log2ceil(symbolCount));
    bloomTable.resize(bloomSize, 0);
  }

  void addHash(uint32_t hash) {
    uint32_t hash2 = hash >> bloomShift;
    uint32_t n = (hash / bitsPerEntry) % bloomSize;
    uint64_t mask =
        (1ULL << (hash % bitsPerEntry)) | (1ULL << (hash2 % bitsPerEntry));
    bloomTable[n] |= mask;
  }

  bool testHash(uint32_t hash) const {
    uint32_t hash2 = hash >> bloomShift;
    uint32_t n = (hash / bitsPerEntry) % bloomSize;
    uint64_t mask =
        (1ULL << (hash % bitsPerEntry)) | (1ULL << (hash2 % bitsPerEntry));
    return (bloomTable[n] & mask) == mask;
  }

  static constexpr uint32_t log2ceil(uint32_t v) {
    return v <= 1 ? 0 : 32 - countl_zero(v - 1);
  }
};

class BloomFilterBuilder {
public:
  using HashFunc = BloomFilter::HashFunc;

  BloomFilterBuilder() = default;

  BloomFilterBuilder &setFalsePositiveRate(float rate) {
    assert(rate > 0.0f && rate < 1.0f);
    falsePositiveRate = rate;
    return *this;
  }

  BloomFilterBuilder &setHashFunction(HashFunc func) {
    hashFunc = std::move(func);
    return *this;
  }

  BloomFilter build(ArrayRef<StringRef> Symbols) const {
    assert(!Symbols.empty() && "Cannot build filter from empty symbol list.");
    BloomFilter filter(static_cast<uint32_t>(Symbols.size()), falsePositiveRate,
                       hashFunc);
    for (const auto &sym : Symbols)
      filter.add(sym);

    return filter;
  }

private:
  float falsePositiveRate = 0.02f;
  HashFunc hashFunc = [](StringRef s) -> uint32_t {
    uint32_t h = 5381;
    for (char c : s)
      h = ((h << 5) + h) + static_cast<uint8_t>(c); // h * 33 + c
    return h;
  };
};

namespace shared {

template <> class SPSSerializationTraits<SPSBloomFilter, BloomFilter> {
public:
  static size_t size(const BloomFilter &Filter) {
    return SPSBloomFilter::AsArgList::size(
        Filter.initialized, Filter.symbolCount_, Filter.bloomSize,
        Filter.bloomShift, Filter.bloomTable);
  }

  static bool serialize(SPSOutputBuffer &OB, const BloomFilter &Filter) {
    return SPSBloomFilter::AsArgList::serialize(
        OB, Filter.initialized, Filter.symbolCount_, Filter.bloomSize,
        Filter.bloomShift, Filter.bloomTable);
  }

  static bool deserialize(SPSInputBuffer &IB, BloomFilter &Filter) {
    bool IsInitialized;
    uint32_t symbolCount_ = 0, bloomSize = 0, bloomShift = 0;
    std::vector<uint64_t> bloomTable;

    if (!SPSBloomFilter::AsArgList::deserialize(
            IB, IsInitialized, symbolCount_, bloomSize, bloomShift, bloomTable))
      return false;

    Filter.initialized = IsInitialized;
    Filter.symbolCount_ = symbolCount_;
    Filter.bloomSize = bloomSize;
    Filter.bloomShift = bloomShift;
    Filter.bloomTable = std::move(bloomTable);

    return true;
  }
};

} // end namespace shared
} // end namespace orc
} // end namespace llvm
#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SYMBOLFILTER_H
