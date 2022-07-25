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

#include "RemoteCacheProvider.h"
#include "compilation_caching_kv.grpc.pb.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/RemoteCachingService/RemoteCacheServer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::remote;
using namespace compilation_cache_service::cas::v1;
using namespace compilation_cache_service::keyvalue::v1;

static cas::CacheKey cacheKeyFromString(StringRef Data) {
  // FIXME: Enhance \p cas::ActionCache to be able to pass data for key, without
  // needing to store the key into a \p cas::ObjectStore.
  static ExitOnError ExitOnErr("LLVMCASCacheProvider: ");
  auto CAS = cas::createInMemoryCAS();
  return CAS->getID(ExitOnErr(CAS->storeFromString({}, Data)));
}

namespace {

class LLVMCASCacheProvider final : public RemoteCacheProvider {
  ThreadPool Pool{hardware_concurrency()};
  std::string TempDir;
  std::unique_ptr<cas::ObjectStore> CAS;
  std::unique_ptr<cas::ActionCache> ActCache;

public:
  void initialize(StringRef TempPath, std::unique_ptr<cas::ObjectStore> CAS,
                  std::unique_ptr<cas::ActionCache> Cache);

  void GetValueAsync(
      const GetValueRequest &Request,
      std::function<void(const GetValueResponse &)> Receiver) override;
  void PutValueAsync(
      const PutValueRequest &Request,
      std::function<void(const PutValueResponse &)> Receiver) override;

  void
  CASLoadAsync(const CASLoadRequest &Request,
               std::function<void(const CASLoadResponse &)> Receiver) override;
  void
  CASSaveAsync(const CASSaveRequest &Request,
               std::function<void(const CASSaveResponse &)> Receiver) override;
  void
  CASGetAsync(const CASGetRequest &Request,
              std::function<void(const CASGetResponse &)> Receiver) override;
  void
  CASPutAsync(const CASPutRequest &Request,
              std::function<void(const CASPutResponse &)> Receiver) override;

  GetValueResponse GetValue(const GetValueRequest &Request);
  PutValueResponse PutValue(const PutValueRequest &Request);

  CASLoadResponse CASLoad(const CASLoadRequest &Request);
  CASSaveResponse CASSave(const CASSaveRequest &Request);
  CASGetResponse CASGet(const CASGetRequest &Request);
  CASPutResponse CASPut(const CASPutRequest &Request);
};

} // namespace

static GetValueResponse GetValueWithError(Error &&E) {
  GetValueResponse Response;
  Response.set_outcome(GetValueResponse_Outcome_ERROR);
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

static PutValueResponse PutValueWithError(Error &&E) {
  PutValueResponse Response;
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

static CASLoadResponse CASLoadWithError(Error &&E) {
  CASLoadResponse Response;
  Response.set_outcome(CASLoadResponse_Outcome_ERROR);
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

static CASSaveResponse CASSaveWithError(Error &&E) {
  CASSaveResponse Response;
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

static CASGetResponse CASGetWithError(Error &&E) {
  CASGetResponse Response;
  Response.set_outcome(CASGetResponse_Outcome_ERROR);
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

static CASPutResponse CASPutWithError(Error &&E) {
  CASPutResponse Response;
  Response.mutable_error()->set_description(toString(std::move(E)));
  return Response;
}

void LLVMCASCacheProvider::initialize(StringRef TempPath,
                                      std::unique_ptr<cas::ObjectStore> CAS,
                                      std::unique_ptr<cas::ActionCache> Cache) {
  TempDir = TempPath.str();
  this->CAS = std::move(CAS);
  this->ActCache = std::move(Cache);
}

void LLVMCASCacheProvider::GetValueAsync(
    const GetValueRequest &Request,
    std::function<void(const GetValueResponse &)> Receiver) {
  Pool.async([this, Request, Receiver]() { Receiver(GetValue(Request)); });
}

void LLVMCASCacheProvider::PutValueAsync(
    const PutValueRequest &Request,
    std::function<void(const PutValueResponse &)> Receiver) {
  Pool.async([this, Request, Receiver]() { Receiver(PutValue(Request)); });
}

void LLVMCASCacheProvider::CASLoadAsync(
    const CASLoadRequest &Request,
    std::function<void(const CASLoadResponse &)> Receiver) {
  Pool.async([this, Request, Receiver]() { Receiver(CASLoad(Request)); });
}

void LLVMCASCacheProvider::CASSaveAsync(
    const CASSaveRequest &Request,
    std::function<void(const CASSaveResponse &)> Receiver) {
  Pool.async([this, Request, Receiver]() { Receiver(CASSave(Request)); });
}

void LLVMCASCacheProvider::CASGetAsync(
    const CASGetRequest &Request,
    std::function<void(const CASGetResponse &)> Receiver) {
  Pool.async([this, Request, Receiver]() { Receiver(CASGet(Request)); });
}

void LLVMCASCacheProvider::CASPutAsync(
    const CASPutRequest &Request,
    std::function<void(const CASPutResponse &)> Receiver) {
  Pool.async([this, Request, Receiver]() { Receiver(CASPut(Request)); });
}

GetValueResponse
LLVMCASCacheProvider::GetValue(const GetValueRequest &Request) {
  cas::CacheKey Key = cacheKeyFromString(Request.key());
  Expected<Optional<cas::CASID>> ID = ActCache->get(Key);
  if (!ID)
    return GetValueWithError(ID.takeError());

  GetValueResponse Response;
  if (*ID) {
    Response.set_outcome(GetValueResponse_Outcome_SUCCESS);
    Expected<cas::ObjectProxy> Obj = CAS->getProxy(**ID);
    if (!Obj)
      return GetValueWithError(Obj.takeError());
    StringRef ValData = Obj->getData();
    Response.mutable_value()->ParseFromArray(ValData.data(), ValData.size());
  } else {
    Response.set_outcome(GetValueResponse_Outcome_KEY_NOT_FOUND);
  }
  return Response;
}

PutValueResponse
LLVMCASCacheProvider::PutValue(const PutValueRequest &Request) {
  Expected<cas::ObjectRef> Obj =
      CAS->storeFromString({}, Request.value().SerializeAsString());
  if (!Obj)
    return PutValueWithError(Obj.takeError());
  cas::CacheKey Key = cacheKeyFromString(Request.key());
  if (Error E = ActCache->put(Key, CAS->getID(*Obj)))
    return PutValueWithError(std::move(E));
  return PutValueResponse();
}

CASLoadResponse LLVMCASCacheProvider::CASLoad(const CASLoadRequest &Request) {
  Expected<cas::CASID> ID = CAS->parseID(Request.cas_id().id());
  if (!ID)
    return CASLoadWithError(ID.takeError());
  Expected<cas::ObjectProxy> Obj = CAS->getProxy(*ID);
  if (!Obj)
    return CASLoadWithError(Obj.takeError());
  StringRef BlobData = Obj->getData();

  CASLoadResponse Response;
  Response.set_outcome(CASLoadResponse_Outcome_SUCCESS);

  if (Request.write_to_disk()) {
    SmallString<128> ModelPath{TempDir};
    sys::path::append(ModelPath, "%%%%%%%.blob");
    int FD;
    SmallString<256> TempPath;
    std::error_code EC = sys::fs::createUniqueFile(ModelPath, FD, TempPath);
    if (EC)
      return CASLoadWithError(createStringError(
          EC, "failed creating '" + ModelPath + "': " + EC.message()));
    raw_fd_ostream OS(FD, /*shouldClose=*/true);
    OS << BlobData;
    OS.close();
    Response.mutable_data()->mutable_blob()->set_file_path(
        TempPath.str().str());
  } else {
    Response.mutable_data()->mutable_blob()->set_data(BlobData.str());
  }

  return Response;
}

CASSaveResponse LLVMCASCacheProvider::CASSave(const CASSaveRequest &Request) {
  const CASBytes &Blob = Request.data().blob();

  Optional<cas::ObjectRef> Ref;
  if (Blob.has_file_path()) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileBuf =
        MemoryBuffer::getFile(Blob.file_path());
    if (!FileBuf)
      return CASSaveWithError(createStringError(
          FileBuf.getError(), "failed reading '" + Blob.file_path() +
                                  "': " + FileBuf.getError().message()));
    if (Error E =
            CAS->storeFromString({}, (*FileBuf)->getBuffer()).moveInto(Ref))
      return CASSaveWithError(std::move(E));
  } else {
    if (Error E = CAS->storeFromString({}, Blob.data()).moveInto(Ref))
      return CASSaveWithError(std::move(E));
  }

  CASSaveResponse Response;
  Response.mutable_cas_id()->set_id(CAS->getID(*Ref).toString());
  return Response;
}

CASGetResponse LLVMCASCacheProvider::CASGet(const CASGetRequest &Request) {
  Expected<cas::CASID> ID = CAS->parseID(Request.cas_id().id());
  if (!ID)
    return CASGetWithError(ID.takeError());
  Expected<cas::ObjectProxy> Obj = CAS->getProxy(*ID);
  if (!Obj)
    return CASGetWithError(Obj.takeError());
  StringRef BlobData = Obj->getData();

  CASGetResponse Response;
  Response.set_outcome(CASGetResponse_Outcome_SUCCESS);

  if (Request.write_to_disk()) {
    SmallString<128> ModelPath{TempDir};
    sys::path::append(ModelPath, "%%%%%%%.blob");
    int FD;
    SmallString<256> TempPath;
    std::error_code EC = sys::fs::createUniqueFile(ModelPath, FD, TempPath);
    if (EC)
      return CASGetWithError(createStringError(
          EC, "failed creating '" + ModelPath + "': " + EC.message()));
    raw_fd_ostream OS(FD, /*shouldClose=*/true);
    OS << BlobData;
    OS.close();
    Response.mutable_data()->mutable_blob()->set_file_path(
        TempPath.str().str());
  } else {
    Response.mutable_data()->mutable_blob()->set_data(BlobData.str());
  }

  auto Err = Obj->forEachReference([&](ObjectRef Ref) {
    std::string Hash = toStringRef(CAS->getID(Ref).getHash()).str();
    Response.mutable_data()->add_references()->set_id(Hash);
    return Error::success();
  });

  if (Err)
    return CASGetWithError(std::move(Err));

  return Response;
}

CASPutResponse LLVMCASCacheProvider::CASPut(const CASPutRequest &Request) {
  const CASBytes &Blob = Request.data().blob();

  SmallVector<ObjectRef> Refs;
  for (auto &Ref : Request.data().references()) {
    auto ID = CAS->parseID(Ref.id());
    if (!ID)
      return CASPutWithError(ID.takeError());
    auto CASRef = CAS->getReference(*ID);
    if (!CASRef)
      return CASPutWithError(createStringError(
          inconvertibleErrorCode(), "cannot get ObjectRef from CASID "));

    Refs.push_back(*CASRef);
  }

  Optional<cas::ObjectRef> Ref;
  if (Blob.has_file_path()) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileBuf =
        MemoryBuffer::getFile(Blob.file_path());
    if (!FileBuf)
      return CASPutWithError(createStringError(
          FileBuf.getError(), "failed reading '" + Blob.file_path() +
                                  "': " + FileBuf.getError().message()));
    if (Error E =
            CAS->storeFromString(Refs, (*FileBuf)->getBuffer()).moveInto(Ref))
      return CASPutWithError(std::move(E));
  } else {
    if (Error E = CAS->storeFromString(Refs, Blob.data()).moveInto(Ref))
      return CASPutWithError(std::move(E));
  }

  CASPutResponse Response;
  Response.mutable_cas_id()->set_id(CAS->getID(*Ref).toString());
  return Response;
}

std::unique_ptr<RemoteCacheProvider>
cas::remote::createLLVMCASCacheProvider(StringRef TempPath,
                                        std::unique_ptr<ObjectStore> CAS,
                                        std::unique_ptr<ActionCache> Cache) {
  auto Provider = std::make_unique<LLVMCASCacheProvider>();
  Provider->initialize(TempPath, std::move(CAS), std::move(Cache));
  return Provider;
}
