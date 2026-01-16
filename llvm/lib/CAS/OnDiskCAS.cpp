//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BuiltinCAS.h"
#include "llvm/CAS/BuiltinCASContext.h"
#include "llvm/CAS/BuiltinObjectHasher.h"
#include "llvm/CAS/OnDiskCASLogger.h"
#include "llvm/CAS/OnDiskGraphDB.h"
#include "llvm/CAS/UnifiedOnDiskCache.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::builtin;

namespace {

class OnDiskCAS : public BuiltinCAS {
public:
  Expected<ObjectRef> storeImpl(ArrayRef<uint8_t> ComputedHash,
                                ArrayRef<ObjectRef> Refs,
                                ArrayRef<char> Data) final;

  Expected<std::optional<ObjectHandle>> loadIfExists(ObjectRef Ref) final;

  CASID getID(ObjectRef Ref) const final;

  std::optional<ObjectRef> getReference(const CASID &ID) const final;

  Expected<bool> isMaterialized(ObjectRef Ref) const final;

  ArrayRef<char> getDataConst(ObjectHandle Node) const final;

  void print(raw_ostream &OS) const final;
  Error validate(bool CheckHash) const final;

  static Expected<std::unique_ptr<OnDiskCAS>> open(StringRef Path);

  OnDiskCAS(std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB)
      : UnifiedDB(std::move(UniDB)), DB(&UnifiedDB->getGraphDB()) {}

private:
  ObjectHandle convertHandle(ondisk::ObjectHandle Node) const {
    return makeObjectHandle(Node.getOpaqueData());
  }

  ondisk::ObjectHandle convertHandle(ObjectHandle Node) const {
    return ondisk::ObjectHandle(Node.getInternalRef(*this));
  }

  ObjectRef convertRef(ondisk::ObjectID Ref) const {
    return makeObjectRef(Ref.getOpaqueData());
  }

  ondisk::ObjectID convertRef(ObjectRef Ref) const {
    return ondisk::ObjectID::fromOpaqueData(Ref.getInternalRef(*this));
  }

  size_t getNumRefs(ObjectHandle Node) const final {
    auto RefsRange = DB->getObjectRefs(convertHandle(Node));
    return llvm::size(RefsRange);
  }

  ObjectRef readRef(ObjectHandle Node, size_t I) const final {
    auto RefsRange = DB->getObjectRefs(convertHandle(Node));
    return convertRef(RefsRange.begin()[I]);
  }

  Error forEachRef(ObjectHandle Node,
                   function_ref<Error(ObjectRef)> Callback) const final;

  Error setSizeLimit(std::optional<uint64_t> SizeLimit) final;
  Expected<std::optional<uint64_t>> getStorageSize() const final;
  Error pruneStorageData() final;

  OnDiskCAS(std::unique_ptr<ondisk::OnDiskGraphDB> GraphDB)
      : OwnedDB(std::move(GraphDB)), DB(OwnedDB.get()) {}

  std::unique_ptr<ondisk::OnDiskGraphDB> OwnedDB;
  std::shared_ptr<ondisk::UnifiedOnDiskCache> UnifiedDB;
  ondisk::OnDiskGraphDB *DB;
};

} // end anonymous namespace

void OnDiskCAS::print(raw_ostream &OS) const { DB->print(OS); }
Error OnDiskCAS::validate(bool CheckHash) const {
  auto Hasher = [](ArrayRef<ArrayRef<uint8_t>> Refs, ArrayRef<char> Data,
                   SmallVectorImpl<uint8_t> &Result) {
    auto Hash = BuiltinObjectHasher<llvm::cas::builtin::HasherT>::hashObject(
        Refs, Data);
    Result.assign(Hash.begin(), Hash.end());
  };

  if (auto E = DB->validate(CheckHash, Hasher))
    return E;

  return Error::success();
}

CASID OnDiskCAS::getID(ObjectRef Ref) const {
  ArrayRef<uint8_t> Hash = DB->getDigest(convertRef(Ref));
  return CASID::create(&getContext(), toStringRef(Hash));
}

std::optional<ObjectRef> OnDiskCAS::getReference(const CASID &ID) const {
  std::optional<ondisk::ObjectID> ObjID =
      DB->getExistingReference(ID.getHash());
  if (!ObjID)
    return std::nullopt;
  return convertRef(*ObjID);
}

Expected<bool> OnDiskCAS::isMaterialized(ObjectRef ExternalRef) const {
  return DB->isMaterialized(convertRef(ExternalRef));
}

ArrayRef<char> OnDiskCAS::getDataConst(ObjectHandle Node) const {
  return DB->getObjectData(convertHandle(Node));
}

Expected<std::optional<ObjectHandle>>
OnDiskCAS::loadIfExists(ObjectRef ExternalRef) {
  Expected<std::optional<ondisk::ObjectHandle>> ObjHnd =
      DB->load(convertRef(ExternalRef));
  if (!ObjHnd)
    return ObjHnd.takeError();
  if (!*ObjHnd)
    return std::nullopt;
  return convertHandle(**ObjHnd);
}

Expected<ObjectRef> OnDiskCAS::storeImpl(ArrayRef<uint8_t> ComputedHash,
                                         ArrayRef<ObjectRef> Refs,
                                         ArrayRef<char> Data) {
  SmallVector<ondisk::ObjectID, 64> IDs;
  IDs.reserve(Refs.size());
  for (ObjectRef Ref : Refs) {
    IDs.push_back(convertRef(Ref));
  }

  auto StoredID = DB->getReference(ComputedHash);
  if (LLVM_UNLIKELY(!StoredID))
    return StoredID.takeError();
  if (Error E = DB->store(*StoredID, IDs, Data))
    return std::move(E);
  return convertRef(*StoredID);
}

Error OnDiskCAS::forEachRef(ObjectHandle Node,
                            function_ref<Error(ObjectRef)> Callback) const {
  auto RefsRange = DB->getObjectRefs(convertHandle(Node));
  for (ondisk::ObjectID Ref : RefsRange) {
    if (Error E = Callback(convertRef(Ref)))
      return E;
  }
  return Error::success();
}

Error OnDiskCAS::setSizeLimit(std::optional<uint64_t> SizeLimit) {
  UnifiedDB->setSizeLimit(SizeLimit);
  return Error::success();
}

Expected<std::optional<uint64_t>> OnDiskCAS::getStorageSize() const {
  return UnifiedDB->getStorageSize();
}

Error OnDiskCAS::pruneStorageData() { return UnifiedDB->collectGarbage(); }

Expected<std::unique_ptr<OnDiskCAS>> OnDiskCAS::open(StringRef AbsPath) {
  std::shared_ptr<ondisk::OnDiskCASLogger> Logger;
#ifndef _WIN32
  if (Error E =
          ondisk::OnDiskCASLogger::openIfEnabled(AbsPath).moveInto(Logger))
    return std::move(E);
#endif

  Expected<std::unique_ptr<ondisk::OnDiskGraphDB>> DB =
      ondisk::OnDiskGraphDB::open(AbsPath, BuiltinCASContext::getHashName(),
                                  sizeof(HashType), /*UpstreamDB=*/nullptr,
                                  std::move(Logger));
  if (!DB)
    return DB.takeError();
  return std::unique_ptr<OnDiskCAS>(new OnDiskCAS(std::move(*DB)));
}

bool cas::isOnDiskCASEnabled() {
#if LLVM_ENABLE_ONDISK_CAS
  return true;
#else
  return false;
#endif
}

Expected<std::unique_ptr<ObjectStore>> cas::createOnDiskCAS(const Twine &Path) {
#if LLVM_ENABLE_ONDISK_CAS
  // FIXME: An absolute path isn't really good enough. Should open a directory
  // and use openat() for files underneath.
  SmallString<256> AbsPath;
  Path.toVector(AbsPath);
  sys::fs::make_absolute(AbsPath);

  return OnDiskCAS::open(AbsPath);
#else
  return createStringError(inconvertibleErrorCode(), "OnDiskCAS is disabled");
#endif /* LLVM_ENABLE_ONDISK_CAS */
}

std::unique_ptr<ObjectStore>
cas::builtin::createObjectStoreFromUnifiedOnDiskCache(
    std::shared_ptr<ondisk::UnifiedOnDiskCache> UniDB) {
  return std::make_unique<OnDiskCAS>(std::move(UniDB));
}
