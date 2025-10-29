//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file implements the underlying ActionCache implementations.
///
//===----------------------------------------------------------------------===//

#include "BuiltinCAS.h"
#include "llvm/ADT/TrieRawHashMap.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/Support/BLAKE3.h"

#define DEBUG_TYPE "cas-action-caches"

using namespace llvm;
using namespace llvm::cas;

namespace {

using HasherT = BLAKE3;
using HashType = decltype(HasherT::hash(std::declval<ArrayRef<uint8_t> &>()));

template <size_t Size> class CacheEntry {
public:
  CacheEntry() = default;
  CacheEntry(ArrayRef<uint8_t> Hash) { llvm::copy(Hash, Value.data()); }
  CacheEntry(const CacheEntry &Entry) { llvm::copy(Entry.Value, Value.data()); }
  ArrayRef<uint8_t> getValue() const { return Value; }

private:
  std::array<uint8_t, Size> Value;
};

/// Builtin InMemory ActionCache that stores the mapping in memory.
class InMemoryActionCache final : public ActionCache {
public:
  InMemoryActionCache()
      : ActionCache(builtin::BuiltinCASContext::getDefaultContext()) {}

  Error putImpl(ArrayRef<uint8_t> ActionKey, const CASID &Result,
                bool CanBeDistributed) final;
  Expected<std::optional<CASID>> getImpl(ArrayRef<uint8_t> ActionKey,
                                         bool CanBeDistributed) const final;

private:
  using DataT = CacheEntry<sizeof(HashType)>;
  using InMemoryCacheT = ThreadSafeTrieRawHashMap<DataT, sizeof(HashType)>;

  InMemoryCacheT Cache;
};
} // end namespace

static Error createResultCachePoisonedError(ArrayRef<uint8_t> KeyHash,
                                            const CASContext &Context,
                                            CASID Output,
                                            ArrayRef<uint8_t> ExistingOutput) {
  std::string Existing =
      CASID::create(&Context, toStringRef(ExistingOutput)).toString();
  SmallString<64> Key;
  toHex(KeyHash, /*LowerCase=*/true, Key);
  return createStringError(std::make_error_code(std::errc::invalid_argument),
                           "cache poisoned for '" + Key + "' (new='" +
                               Output.toString() + "' vs. existing '" +
                               Existing + "')");
}

Expected<std::optional<CASID>>
InMemoryActionCache::getImpl(ArrayRef<uint8_t> Key,
                             bool /*CanBeDistributed*/) const {
  auto Result = Cache.find(Key);
  if (!Result)
    return std::nullopt;
  return CASID::create(&getContext(), toStringRef(Result->Data.getValue()));
}

Error InMemoryActionCache::putImpl(ArrayRef<uint8_t> Key, const CASID &Result,
                                   bool /*CanBeDistributed*/) {
  DataT Expected(Result.getHash());
  const InMemoryCacheT::value_type &Cached = *Cache.insertLazy(
      Key, [&](auto ValueConstructor) { ValueConstructor.emplace(Expected); });

  const DataT &Observed = Cached.Data;
  if (Expected.getValue() == Observed.getValue())
    return Error::success();

  return createResultCachePoisonedError(Key, getContext(), Result,
                                        Observed.getValue());
}

namespace llvm::cas {

std::unique_ptr<ActionCache> createInMemoryActionCache() {
  return std::make_unique<InMemoryActionCache>();
}

} // namespace llvm::cas
