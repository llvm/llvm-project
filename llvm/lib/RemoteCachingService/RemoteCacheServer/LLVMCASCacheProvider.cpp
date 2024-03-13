//===-- LLVMCASCacheProvider.cpp - LLVMCAS Remote Cache Service -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A RemoteCacheProvider implementation using on-disk \p cas::ObjectStore and
// \p cas::ActionCache. While this is intended for testing purposes, it also
// functions as a reference implementation for a \p RemoteCacheProvider.
//
//===----------------------------------------------------------------------===//

#include "llvm/RemoteCachingService/LLVMCASCacheProvider.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/RemoteCachingService/RemoteCacheProvider.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::remote;

static cas::CacheKey cacheKeyFromString(StringRef Data) {
  // FIXME: Enhance \p cas::ActionCache to be able to pass data for key, without
  // needing to store the key into a \p cas::ObjectStore.
  static ExitOnError ExitOnErr("LLVMCASCacheProvider: ");
  auto CAS = cas::createInMemoryCAS();
  return CAS->getID(ExitOnErr(CAS->storeFromString({}, Data)));
}

namespace {

class LLVMCASCacheProvider final : public RemoteCacheProvider {
  DefaultThreadPool Pool{hardware_concurrency()};
  std::string TempDir;
  std::unique_ptr<cas::ObjectStore> CAS;
  std::unique_ptr<cas::ActionCache> ActCache;

public:
  void initialize(StringRef TempPath, std::unique_ptr<cas::ObjectStore> CAS,
                  std::unique_ptr<cas::ActionCache> Cache);

  void GetValueAsync(std::string Key,
                     std::function<void(Expected<std::optional<std::string>>)>
                         Receiver) override;
  void PutValueAsync(std::string Key, std::string Value,
                     std::function<void(Error)> Receiver) override;

  void
  CASLoadAsync(std::string CASID, bool WriteToDisk,
               std::function<void(Expected<LoadResponse>)> Receiver) override;
  void
  CASSaveAsync(BlobContents Blob,
               std::function<void(Expected<std::string>)> Receiver) override;

  void
  CASGetAsync(std::string CASID, bool WriteToDisk,
              std::function<void(Expected<GetResponse>)> Receiver) override;
  void
  CASPutAsync(BlobContents Blob, SmallVector<std::string> Refs,
              std::function<void(Expected<std::string>)> Receiver) override;

  Expected<std::optional<std::string>> GetValue(StringRef Key);
  Error PutValue(StringRef Key, StringRef Value);

  Expected<LoadResponse> CASLoad(StringRef CASID, bool WriteToDisk);
  Expected<std::string> CASSave(const BlobContents &Blob);

  Expected<GetResponse> CASGet(StringRef CASID, bool WriteToDisk);
  Expected<std::string> CASPut(const BlobContents &Blob,
                               ArrayRef<std::string> Refs);
};

} // namespace

void LLVMCASCacheProvider::initialize(StringRef TempPath,
                                      std::unique_ptr<cas::ObjectStore> CAS,
                                      std::unique_ptr<cas::ActionCache> Cache) {
  TempDir = TempPath.str();
  this->CAS = std::move(CAS);
  this->ActCache = std::move(Cache);
}

void LLVMCASCacheProvider::GetValueAsync(
    std::string Key,
    std::function<void(Expected<std::optional<std::string>>)> Receiver) {
  Pool.async([this, Key = std::move(Key), Receiver = std::move(Receiver)]() {
    Receiver(GetValue(Key));
  });
}

void LLVMCASCacheProvider::PutValueAsync(std::string Key, std::string Value,
                                         std::function<void(Error)> Receiver) {
  Pool.async(
      [this, Key = std::move(Key), Value = std::move(Value),
       Receiver = std::move(Receiver)]() { Receiver(PutValue(Key, Value)); });
}

void LLVMCASCacheProvider::CASLoadAsync(
    std::string CASID, bool WriteToDisk,
    std::function<void(Expected<LoadResponse>)> Receiver) {
  Pool.async([this, CASID = std::move(CASID), WriteToDisk,
              Receiver = std::move(Receiver)]() {
    Receiver(CASLoad(CASID, WriteToDisk));
  });
}

void LLVMCASCacheProvider::CASSaveAsync(
    BlobContents Blob, std::function<void(Expected<std::string>)> Receiver) {
  Pool.async([this, Blob = std::move(Blob), Receiver = std::move(Receiver)]() {
    Receiver(CASSave(Blob));
  });
}

void LLVMCASCacheProvider::CASGetAsync(
    std::string CASID, bool WriteToDisk,
    std::function<void(Expected<GetResponse>)> Receiver) {
  Pool.async([this, CASID = std::move(CASID), WriteToDisk,
              Receiver = std::move(Receiver)]() {
    Receiver(CASGet(CASID, WriteToDisk));
  });
}

void LLVMCASCacheProvider::CASPutAsync(
    BlobContents Blob, SmallVector<std::string> Refs,
    std::function<void(Expected<std::string>)> Receiver) {
  Pool.async(
      [this, Blob = std::move(Blob), Refs = std::move(Refs),
       Receiver = std::move(Receiver)]() { Receiver(CASPut(Blob, Refs)); });
}

Expected<std::optional<std::string>>
LLVMCASCacheProvider::GetValue(StringRef RawKey) {
  cas::CacheKey Key = cacheKeyFromString(RawKey);
  Expected<std::optional<cas::CASID>> ID = ActCache->get(Key);
  if (!ID)
    return ID.takeError();

  if (*ID) {
    Expected<cas::ObjectProxy> Obj = CAS->getProxy(**ID);
    if (!Obj)
      return Obj.takeError();
    return Obj->getData().str();
  } else {
    return std::nullopt;
  }
}

Error LLVMCASCacheProvider::PutValue(StringRef RawKey, StringRef Value) {
  Expected<cas::ObjectRef> Obj = CAS->storeFromString({}, Value);
  if (!Obj)
    return Obj.takeError();
  cas::CacheKey Key = cacheKeyFromString(RawKey);
  return ActCache->put(Key, CAS->getID(*Obj));
}

Expected<RemoteCacheProvider::LoadResponse>
LLVMCASCacheProvider::CASLoad(StringRef CASID, bool WriteToDisk) {
  Expected<cas::CASID> ID = CAS->parseID(CASID);
  if (!ID)
    return ID.takeError();
  Expected<cas::ObjectProxy> Obj = CAS->getProxy(*ID);
  if (!Obj)
    return Obj.takeError();
  StringRef BlobData = Obj->getData();

  if (WriteToDisk) {
    SmallString<128> ModelPath{TempDir};
    sys::path::append(ModelPath, "%%%%%%%.blob");
    int FD;
    SmallString<256> TempPath;
    std::error_code EC = sys::fs::createUniqueFile(ModelPath, FD, TempPath);
    if (EC)
      return createStringError(EC, "failed creating '" + ModelPath +
                                       "': " + EC.message());
    raw_fd_ostream OS(FD, /*shouldClose=*/true);
    OS << BlobData;
    OS.close();
    return LoadResponse{
        /*KeyNotFound*/ false,
        BlobContents{/*IsFilePath*/ true, TempPath.str().str()}};
  } else {
    return LoadResponse{/*KeyNotFound*/ false,
                        BlobContents{/*IsFilePath*/ false, BlobData.str()}};
  }
}

Expected<std::string> LLVMCASCacheProvider::CASSave(const BlobContents &Blob) {
  std::optional<cas::ObjectRef> Ref;
  if (Blob.IsFilePath) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileBuf =
        MemoryBuffer::getFile(Blob.DataOrPath);
    if (!FileBuf)
      return createStringError(FileBuf.getError(),
                               "failed reading '" + Blob.DataOrPath +
                                   "': " + FileBuf.getError().message());
    if (Error E =
            CAS->storeFromString({}, (*FileBuf)->getBuffer()).moveInto(Ref))
      return std::move(E);
  } else {
    if (Error E = CAS->storeFromString({}, Blob.DataOrPath).moveInto(Ref))
      return std::move(E);
  }

  return CAS->getID(*Ref).toString();
}

Expected<RemoteCacheProvider::GetResponse>
LLVMCASCacheProvider::CASGet(StringRef CASID, bool WriteToDisk) {
  Expected<cas::CASID> ID = CAS->parseID(CASID);
  if (!ID)
    return ID.takeError();
  Expected<cas::ObjectProxy> Obj = CAS->getProxy(*ID);
  if (!Obj)
    return Obj.takeError();
  StringRef BlobData = Obj->getData();

  SmallVector<std::string> Refs;
  Refs.reserve(Obj->getNumReferences());
  auto Err = Obj->forEachReference([&](ObjectRef Ref) {
    Refs.push_back(toStringRef(CAS->getID(Ref).getHash()).str());
    return Error::success();
  });
  if (Err)
    return std::move(Err);

  if (WriteToDisk) {
    SmallString<128> ModelPath{TempDir};
    sys::path::append(ModelPath, "%%%%%%%.blob");
    int FD;
    SmallString<256> TempPath;
    std::error_code EC = sys::fs::createUniqueFile(ModelPath, FD, TempPath);
    if (EC)
      return createStringError(EC, "failed creating '" + ModelPath +
                                       "': " + EC.message());
    raw_fd_ostream OS(FD, /*shouldClose=*/true);
    OS << BlobData;
    OS.close();
    return GetResponse{/*KeyNotFound*/ false,
                       BlobContents{/*IsFilePath*/ true, TempPath.str().str()},
                       std::move(Refs)};
  } else {
    return GetResponse{/*KeyNotFound*/ false,
                       BlobContents{/*IsFilePath*/ false, BlobData.str()},
                       std::move(Refs)};
  }
}

Expected<std::string>
LLVMCASCacheProvider::CASPut(const BlobContents &Blob,
                             ArrayRef<std::string> RawRefs) {
  SmallVector<ObjectRef> Refs;
  for (const auto &Ref : RawRefs) {
    auto ID = CAS->parseID(Ref);
    if (!ID)
      return ID.takeError();
    auto CASRef = CAS->getReference(*ID);
    if (!CASRef)
      return createStringError(inconvertibleErrorCode(),
                               "cannot get ObjectRef from CASID ");

    Refs.push_back(*CASRef);
  }

  std::optional<cas::ObjectRef> Ref;
  if (Blob.IsFilePath) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileBuf =
        MemoryBuffer::getFile(Blob.DataOrPath);
    if (!FileBuf)
      return createStringError(FileBuf.getError(),
                               "failed reading '" + Blob.DataOrPath +
                                   "': " + FileBuf.getError().message());
    if (Error E =
            CAS->storeFromString(Refs, (*FileBuf)->getBuffer()).moveInto(Ref))
      return std::move(E);
  } else {
    if (Error E = CAS->storeFromString(Refs, Blob.DataOrPath).moveInto(Ref))
      return std::move(E);
  }

  return CAS->getID(*Ref).toString();
}

std::unique_ptr<RemoteCacheProvider>
cas::remote::createLLVMCASCacheProvider(StringRef TempPath,
                                        std::unique_ptr<ObjectStore> CAS,
                                        std::unique_ptr<ActionCache> Cache) {
  auto Provider = std::make_unique<LLVMCASCacheProvider>();
  Provider->initialize(TempPath, std::move(CAS), std::move(Cache));
  return Provider;
}
