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
#include "llvm/CAS/OnDiskHashMappedTrie.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/BLAKE3.h"
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
  InMemoryActionCache(ObjectStore &CAS);

  Error putImpl(ArrayRef<uint8_t> ActionKey, const ObjectRef &Result) final;
  Expected<Optional<CASID>> getImpl(ArrayRef<uint8_t> ActionKey) const final;

private:
  using DataT = CacheEntry<sizeof(HashType)>;
  using InMemoryCacheT = ThreadSafeHashMappedTrie<DataT, sizeof(HashType)>;

  InMemoryCacheT Cache;
};

#if LLVM_ENABLE_ONDISK_CAS
class OnDiskActionCache final : public ActionCache {
public:
  Error putImpl(ArrayRef<uint8_t> ActionKey, const ObjectRef &Result) final;
  Expected<Optional<CASID>> getImpl(ArrayRef<uint8_t> ActionKey) const final;

  static Expected<std::unique_ptr<OnDiskActionCache>> create(ObjectStore &CAS,
                                                             StringRef Path);

private:
  static StringRef getHashName() { return "BLAKE3"; }
  static StringRef getActionCacheTableName() {
    static const std::string Name =
        ("llvm.actioncache[" + getHashName() + "->" + getHashName() + "]")
            .str();
    return Name;
  }
  static constexpr StringLiteral ActionCacheFile = "actions";
  static constexpr StringLiteral FilePrefix = "v1.";

  OnDiskActionCache(ObjectStore &CAS, StringRef RootPath,
                    OnDiskHashMappedTrie ActionCache);

  std::string Path;
  OnDiskHashMappedTrie Cache;
  using DataT = CacheEntry<sizeof(HashType)>;
};
#endif /* LLVM_ENABLE_ONDISK_CAS */
} // end namespace

static std::string hashToString(ArrayRef<uint8_t> Hash) {
  SmallString<64> Str;
  toHex(Hash, /*LowerCase=*/true, Str);
  return Str.str().str();
}

static Error createResultCachePoisonedError(StringRef Key, ObjectStore &CAS,
                                            ObjectRef Output,
                                            ArrayRef<uint8_t> ExistingOutput) {
  std::string OutID = CAS.getID(Output).toString();
  std::string Existing =
      CASID::create(&CAS, toStringRef(ExistingOutput)).toString();
  return createStringError(std::make_error_code(std::errc::invalid_argument),
                           "cache poisoned for '" + Key + "' (new='" + OutID +
                               "' vs. existing '" + Existing + "')");
}

// TODO: Check the hash schema is the same between action cache and CAS. If we
// can derive that from static type information, that would be even better.
InMemoryActionCache::InMemoryActionCache(ObjectStore &CAS) : ActionCache(CAS) {}

Expected<Optional<CASID>>
InMemoryActionCache::getImpl(ArrayRef<uint8_t> Key) const {
  auto Result = Cache.find(Key);
  if (!Result)
    return None;
  return CASID::create(&getCAS(), toStringRef(Result->Data.getValue()));
}

Error InMemoryActionCache::putImpl(ArrayRef<uint8_t> Key,
                                   const ObjectRef &Result) {
  DataT Expected(getCAS().getID(Result).getHash());
  const InMemoryCacheT::value_type &Cached = *Cache.insertLazy(
      Key, [&](auto ValueConstructor) { ValueConstructor.emplace(Expected); });

  const DataT &Observed = Cached.Data;
  if (Expected.getValue() == Observed.getValue())
    return Error::success();

  return createResultCachePoisonedError(hashToString(Key), getCAS(), Result,
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

std::unique_ptr<ActionCache> createInMemoryActionCache(ObjectStore &CAS) {
  return std::make_unique<InMemoryActionCache>(CAS);
}

} // namespace cas
} // namespace llvm

#if LLVM_ENABLE_ONDISK_CAS
constexpr StringLiteral OnDiskActionCache::ActionCacheFile;
constexpr StringLiteral OnDiskActionCache::FilePrefix;

OnDiskActionCache::OnDiskActionCache(ObjectStore &CAS, StringRef Path,
                                     OnDiskHashMappedTrie Cache)
    : ActionCache(CAS), Path(Path.str()), Cache(std::move(Cache)) {}

Expected<std::unique_ptr<OnDiskActionCache>>
OnDiskActionCache::create(ObjectStore &CAS, StringRef AbsPath) {
  if (std::error_code EC = sys::fs::create_directories(AbsPath))
    return createFileError(AbsPath, EC);

  SmallString<256> CachePath(AbsPath);
  sys::path::append(CachePath, FilePrefix + ActionCacheFile);
  constexpr uint64_t MB = 1024ull * 1024ull;
  constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;

  Optional<OnDiskHashMappedTrie> ActionCache;
  if (Error E = OnDiskHashMappedTrie::create(
                    CachePath, getActionCacheTableName(), sizeof(HashType) * 8,
                    /*DataSize=*/sizeof(DataT), /*MaxFileSize=*/GB,
                    /*MinFileSize=*/MB)
                    .moveInto(ActionCache))
    return std::move(E);

  return std::unique_ptr<OnDiskActionCache>(
      new OnDiskActionCache(CAS, AbsPath, std::move(*ActionCache)));
}

Expected<Optional<CASID>>
OnDiskActionCache::getImpl(ArrayRef<uint8_t> Key) const {
  // Check the result cache.
  OnDiskHashMappedTrie::const_pointer ActionP = Cache.find(Key);
  if (!ActionP)
    return None;

  const DataT *Output = reinterpret_cast<const DataT *>(ActionP->Data.data());
  return CASID::create(&getCAS(), toStringRef(Output->getValue()));
}

Error OnDiskActionCache::putImpl(ArrayRef<uint8_t> Key,
                                 const ObjectRef &Result) {
  DataT Expected(getCAS().getID(Result).getHash());
  OnDiskHashMappedTrie::pointer ActionP = Cache.insertLazy(
      Key, [&](FileOffset TentativeOffset,
               OnDiskHashMappedTrie::ValueProxy TentativeValue) {
        assert(TentativeValue.Data.size() == sizeof(DataT));
        assert(isAddrAligned(Align::Of<DataT>(), TentativeValue.Data.data()));
        new (TentativeValue.Data.data()) DataT{Expected};
      });
  const DataT *Observed = reinterpret_cast<const DataT *>(ActionP->Data.data());

  if (Expected.getValue() == Observed->getValue())
    return Error::success();

  return createResultCachePoisonedError(hashToString(Key), getCAS(), Result,
                                        Observed->getValue());
}

namespace llvm {
namespace cas {

Expected<std::unique_ptr<ActionCache>>
createOnDiskActionCache(ObjectStore &CAS, StringRef Path) {
  return OnDiskActionCache::create(CAS, Path);
}

} // namespace cas
} // namespace llvm
# else

namespace llvm {
namespace cas {

Expected<std::unique_ptr<ActionCache>> createOnDiskActionCache(ObjectStore &CAS,
                                                               StringRef Path) {
  return createStringError(inconvertibleErrorCode(), "OnDiskCache is disabled");
}

} // namespace cas
} // namespace llvm
#endif /* LLVM_ENABLE_ONDISK_CAS */
