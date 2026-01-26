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
#include "llvm/CAS/OnDiskCASLogger.h"
#include "llvm/CAS/OnDiskKeyValueDB.h"
#include "llvm/CAS/UnifiedOnDiskCache.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Errc.h"

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

  Error validate() const final {
    return createStringError("InMemoryActionCache doesn't support validate()");
  }

private:
  using DataT = CacheEntry<sizeof(HashType)>;
  using InMemoryCacheT = ThreadSafeTrieRawHashMap<DataT, sizeof(HashType)>;

  InMemoryCacheT Cache;
};

/// Builtin basic OnDiskActionCache that uses one underlying OnDiskKeyValueDB.
class OnDiskActionCache final : public ActionCache {
public:
  Error putImpl(ArrayRef<uint8_t> ActionKey, const CASID &Result,
                bool CanBeDistributed) final;
  Expected<std::optional<CASID>> getImpl(ArrayRef<uint8_t> ActionKey,
                                         bool CanBeDistributed) const final;

  static Expected<std::unique_ptr<OnDiskActionCache>> create(StringRef Path);

  Error validate() const final;

private:
  static StringRef getHashName() { return "BLAKE3"; }

  OnDiskActionCache(std::unique_ptr<ondisk::OnDiskKeyValueDB> DB);

  std::unique_ptr<ondisk::OnDiskKeyValueDB> DB;
  using DataT = CacheEntry<sizeof(HashType)>;
};

/// Builtin unified ActionCache that wraps around UnifiedOnDiskCache to provide
/// access to its ActionCache.
class UnifiedOnDiskActionCache final : public ActionCache {
public:
  Error putImpl(ArrayRef<uint8_t> ActionKey, const CASID &Result,
                bool CanBeDistributed) final;
  Expected<std::optional<CASID>> getImpl(ArrayRef<uint8_t> ActionKey,
                                         bool CanBeDistributed) const final;

  UnifiedOnDiskActionCache(std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB);

  Error validate() const final;

private:
  std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB;
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

OnDiskActionCache::OnDiskActionCache(
    std::unique_ptr<ondisk::OnDiskKeyValueDB> DB)
    : ActionCache(builtin::BuiltinCASContext::getDefaultContext()),
      DB(std::move(DB)) {}

Expected<std::unique_ptr<OnDiskActionCache>>
OnDiskActionCache::create(StringRef AbsPath) {
  std::shared_ptr<ondisk::OnDiskCASLogger> Logger;
#ifndef _WIN32
  if (Error E =
          ondisk::OnDiskCASLogger::openIfEnabled(AbsPath).moveInto(Logger))
    return std::move(E);
#endif
  std::unique_ptr<ondisk::OnDiskKeyValueDB> DB;
  if (Error E = ondisk::OnDiskKeyValueDB::open(
                    AbsPath, getHashName(), sizeof(HashType), getHashName(),
                    sizeof(DataT), /*UnifiedCache=*/nullptr, std::move(Logger))
                    .moveInto(DB))
    return std::move(E);
  return std::unique_ptr<OnDiskActionCache>(
      new OnDiskActionCache(std::move(DB)));
}

Expected<std::optional<CASID>>
OnDiskActionCache::getImpl(ArrayRef<uint8_t> Key,
                           bool /*CanBeDistributed*/) const {
  std::optional<ArrayRef<char>> Val;
  if (Error E = DB->get(Key).moveInto(Val))
    return std::move(E);
  if (!Val)
    return std::nullopt;
  return CASID::create(&getContext(), toStringRef(*Val));
}

Error OnDiskActionCache::putImpl(ArrayRef<uint8_t> Key, const CASID &Result,
                                 bool /*CanBeDistributed*/) {
  auto ResultHash = Result.getHash();
  ArrayRef Expected((const char *)ResultHash.data(), ResultHash.size());
  ArrayRef<char> Observed;
  if (Error E = DB->put(Key, Expected).moveInto(Observed))
    return E;

  if (Expected == Observed)
    return Error::success();

  return createResultCachePoisonedError(
      Key, getContext(), Result,
      ArrayRef((const uint8_t *)Observed.data(), Observed.size()));
}

Error OnDiskActionCache::validate() const {
  // FIXME: without the matching CAS there is nothing we can check about the
  // cached values. The hash size is already validated by the DB validator.
  return DB->validate(nullptr);
}

UnifiedOnDiskActionCache::UnifiedOnDiskActionCache(
    std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB)
    : ActionCache(builtin::BuiltinCASContext::getDefaultContext()),
      UniDB(std::move(UniDB)) {}

Expected<std::optional<CASID>>
UnifiedOnDiskActionCache::getImpl(ArrayRef<uint8_t> Key,
                                  bool /*CanBeDistributed*/) const {
  std::optional<ArrayRef<char>> Val;
  if (Error E = UniDB->getKeyValueDB().get(Key).moveInto(Val))
    return std::move(E);
  if (!Val)
    return std::nullopt;
  auto ID = ondisk::UnifiedOnDiskCache::getObjectIDFromValue(*Val);
  return CASID::create(&getContext(),
                       toStringRef(UniDB->getGraphDB().getDigest(ID)));
}

Error UnifiedOnDiskActionCache::putImpl(ArrayRef<uint8_t> Key,
                                        const CASID &Result,
                                        bool /*CanBeDistributed*/) {
  auto Expected = UniDB->getGraphDB().getReference(Result.getHash());
  if (LLVM_UNLIKELY(!Expected))
    return Expected.takeError();

  auto Value = ondisk::UnifiedOnDiskCache::getValueFromObjectID(*Expected);
  std::optional<ArrayRef<char>> Observed;
  if (Error E = UniDB->getKeyValueDB().put(Key, Value).moveInto(Observed))
    return E;

  auto ObservedID = ondisk::UnifiedOnDiskCache::getObjectIDFromValue(*Observed);
  if (*Expected == ObservedID)
    return Error::success();

  return createResultCachePoisonedError(
      Key, getContext(), Result, UniDB->getGraphDB().getDigest(ObservedID));
}

Error UnifiedOnDiskActionCache::validate() const {
  return UniDB->validateActionCache();
}

Expected<std::unique_ptr<ActionCache>>
cas::createOnDiskActionCache(StringRef Path) {
#if LLVM_ENABLE_ONDISK_CAS
  return OnDiskActionCache::create(Path);
#else
  return createStringError(inconvertibleErrorCode(), "OnDiskCache is disabled");
#endif
}

std::unique_ptr<ActionCache>
cas::builtin::createActionCacheFromUnifiedOnDiskCache(
    std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB) {
  return std::make_unique<UnifiedOnDiskActionCache>(std::move(UniDB));
}
