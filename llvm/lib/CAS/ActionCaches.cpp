//===- ActionCaches.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BuiltinCAS.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/HashMappedTrie.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/OnDiskGraphDB.h"
#include "llvm/CAS/OnDiskHashMappedTrie.h"
#include "llvm/CAS/OnDiskKeyValueDB.h"
#include "llvm/CAS/UnifiedOnDiskCache.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "action-caches"

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

class InMemoryActionCache final : public ActionCache {
public:
  InMemoryActionCache()
      : ActionCache(builtin::BuiltinCASContext::getDefaultContext()) {}

  Error putImpl(ArrayRef<uint8_t> ActionKey, const CASID &Result,
                bool Globally) final;
  Expected<std::optional<CASID>> getImpl(ArrayRef<uint8_t> ActionKey,
                                         bool Globally) const final;

private:
  using DataT = CacheEntry<sizeof(HashType)>;
  using InMemoryCacheT = ThreadSafeHashMappedTrie<DataT, sizeof(HashType)>;

  InMemoryCacheT Cache;
};

class OnDiskActionCache final : public ActionCache {
public:
  Error putImpl(ArrayRef<uint8_t> ActionKey, const CASID &Result,
                bool Globally) final;
  Expected<std::optional<CASID>> getImpl(ArrayRef<uint8_t> ActionKey,
                                         bool Globally) const final;

  static Expected<std::unique_ptr<OnDiskActionCache>> create(StringRef Path);

private:
  static StringRef getHashName() { return "BLAKE3"; }

  OnDiskActionCache(std::unique_ptr<ondisk::OnDiskKeyValueDB> DB);

  std::unique_ptr<ondisk::OnDiskKeyValueDB> DB;
  using DataT = CacheEntry<sizeof(HashType)>;
};

class UnifiedOnDiskActionCache final : public ActionCache {
public:
  Error putImpl(ArrayRef<uint8_t> ActionKey, const CASID &Result,
                bool Globally) final;
  Expected<std::optional<CASID>> getImpl(ArrayRef<uint8_t> ActionKey,
                                         bool Globally) const final;

  UnifiedOnDiskActionCache(std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB);

private:
  std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB;
};
} // end namespace

static std::string hashToString(ArrayRef<uint8_t> Hash) {
  SmallString<64> Str;
  toHex(Hash, /*LowerCase=*/true, Str);
  return Str.str().str();
}

static Error createResultCachePoisonedError(StringRef Key,
                                            const CASContext &Context,
                                            CASID Output,
                                            ArrayRef<uint8_t> ExistingOutput) {
  std::string Existing =
      CASID::create(&Context, toStringRef(ExistingOutput)).toString();
  return createStringError(std::make_error_code(std::errc::invalid_argument),
                           "cache poisoned for '" + Key + "' (new='" +
                               Output.toString() + "' vs. existing '" +
                               Existing + "')");
}

Expected<std::optional<CASID>>
InMemoryActionCache::getImpl(ArrayRef<uint8_t> Key, bool /*Globally*/) const {
  auto Result = Cache.find(Key);
  if (!Result)
    return std::nullopt;
  return CASID::create(&getContext(), toStringRef(Result->Data.getValue()));
}

Error InMemoryActionCache::putImpl(ArrayRef<uint8_t> Key, const CASID &Result,
                                   bool /*Globally*/) {
  DataT Expected(Result.getHash());
  const InMemoryCacheT::value_type &Cached = *Cache.insertLazy(
      Key, [&](auto ValueConstructor) { ValueConstructor.emplace(Expected); });

  const DataT &Observed = Cached.Data;
  if (Expected.getValue() == Observed.getValue())
    return Error::success();

  return createResultCachePoisonedError(hashToString(Key), getContext(), Result,
                                        Observed.getValue());
}

static constexpr StringLiteral DefaultName = "actioncache";

namespace llvm {
namespace cas {

std::string getDefaultOnDiskActionCachePath() {
  SmallString<128> Path;
  if (!llvm::sys::path::cache_directory(Path))
    report_fatal_error("cannot get default cache directory");
  llvm::sys::path::append(Path, builtin::DefaultDir, DefaultName);
  return Path.str().str();
}

std::unique_ptr<ActionCache> createInMemoryActionCache() {
  return std::make_unique<InMemoryActionCache>();
}

} // namespace cas
} // namespace llvm

OnDiskActionCache::OnDiskActionCache(
    std::unique_ptr<ondisk::OnDiskKeyValueDB> DB)
    : ActionCache(builtin::BuiltinCASContext::getDefaultContext()),
      DB(std::move(DB)) {}

Expected<std::unique_ptr<OnDiskActionCache>>
OnDiskActionCache::create(StringRef AbsPath) {
  std::unique_ptr<ondisk::OnDiskKeyValueDB> DB;
  if (Error E = ondisk::OnDiskKeyValueDB::open(AbsPath, getHashName(),
                                               sizeof(HashType), getHashName(),
                                               sizeof(DataT))
                    .moveInto(DB))
    return std::move(E);
  return std::unique_ptr<OnDiskActionCache>(
      new OnDiskActionCache(std::move(DB)));
}

Expected<std::optional<CASID>>
OnDiskActionCache::getImpl(ArrayRef<uint8_t> Key, bool /*Globally*/) const {
  std::optional<ArrayRef<char>> Val;
  if (Error E = DB->get(Key).moveInto(Val))
    return std::move(E);
  if (!Val)
    return std::nullopt;
  return CASID::create(&getContext(), toStringRef(*Val));
}

Error OnDiskActionCache::putImpl(ArrayRef<uint8_t> Key, const CASID &Result,
                                 bool /*Globally*/) {
  auto ResultHash = Result.getHash();
  ArrayRef Expected((const char *)ResultHash.data(), ResultHash.size());
  ArrayRef<char> Observed;
  if (Error E = DB->put(Key, Expected).moveInto(Observed))
    return E;

  if (Expected == Observed)
    return Error::success();

  return createResultCachePoisonedError(
      hashToString(Key), getContext(), Result,
      ArrayRef((const uint8_t *)Observed.data(), Observed.size()));
}

UnifiedOnDiskActionCache::UnifiedOnDiskActionCache(
    std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB)
    : ActionCache(builtin::BuiltinCASContext::getDefaultContext()),
      UniDB(std::move(UniDB)) {}

Expected<std::optional<CASID>>
UnifiedOnDiskActionCache::getImpl(ArrayRef<uint8_t> Key,
                                  bool /*Globally*/) const {
  std::optional<ondisk::ObjectID> Val;
  if (Error E = UniDB->KVGet(Key).moveInto(Val))
    return std::move(E);
  if (!Val)
    return std::nullopt;
  return CASID::create(&getContext(),
                       toStringRef(UniDB->getGraphDB().getDigest(*Val)));
}

Error UnifiedOnDiskActionCache::putImpl(ArrayRef<uint8_t> Key,
                                        const CASID &Result,
                                        bool /*Globally*/) {
  auto Expected = UniDB->getGraphDB().getReference(Result.getHash());
  if (LLVM_UNLIKELY(!Expected))
    return Expected.takeError();
  std::optional<ondisk::ObjectID> Observed;
  if (Error E = UniDB->KVPut(Key, *Expected).moveInto(Observed))
    return E;

  if (*Expected == Observed)
    return Error::success();

  return createResultCachePoisonedError(
      hashToString(Key), getContext(), Result,
      UniDB->getGraphDB().getDigest(*Observed));
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
