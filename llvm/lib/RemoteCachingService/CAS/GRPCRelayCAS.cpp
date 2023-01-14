//===- GRPCRelayCAS.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/LazyAtomicPointer.h"
#include "llvm/CAS/CASID.h"
#include "llvm/CAS/HashMappedTrie.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/ThreadSafeAllocator.h"
#include "llvm/Config/config.h"
#include "llvm/RemoteCachingService/Client.h"
#include "llvm/RemoteCachingService/RemoteCachingService.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

#define DEBUG_TYPE "grpc-cas"

using namespace llvm;
using namespace llvm::cas;

namespace {

class InMemoryCASData;
// The in memory HashMappedTrie to store CASData from Service.
// This implementation assumes 80 byte hash max.
using InMemoryIndexT =
    ThreadSafeHashMappedTrie<LazyAtomicPointer<const InMemoryCASData>, 80>;
using InMemoryIndexValueT = InMemoryIndexT::value_type;

// InMemoryCASData.
// Store CASData together with the hash in the index.
class InMemoryCASData {
public:
  ArrayRef<uint8_t> getHash() const { return Index.Hash; }

  InMemoryCASData() = delete;
  InMemoryCASData(InMemoryCASData &&) = delete;
  InMemoryCASData(const InMemoryCASData &) = delete;

  ArrayRef<const InMemoryIndexValueT *> getRefs() const {
    return makeArrayRef(
        reinterpret_cast<const InMemoryIndexValueT *const *>(this + 1),
        NumRefs);
  }

  ArrayRef<char> getData() const {
    ArrayRef<const InMemoryIndexValueT *> Refs = getRefs();
    return makeArrayRef(
        reinterpret_cast<const char *>(Refs.data() + Refs.size()), DataSize);
  }

  uint64_t getDataSize() const { return DataSize; }
  size_t getNumRefs() const { return NumRefs; }

  const InMemoryIndexValueT &getIndex() const { return Index; }

  static InMemoryCASData &create(function_ref<void *(size_t Size)> Allocate,
                                 const InMemoryIndexValueT &I,
                                 ArrayRef<const InMemoryIndexValueT *> Refs,
                                 ArrayRef<char> Data) {
    void *Mem = Allocate(sizeof(InMemoryCASData) +
                         sizeof(uintptr_t) * Refs.size() + Data.size() + 1);
    return *new (Mem) InMemoryCASData(I, Refs, Data);
  }

private:
  InMemoryCASData(const InMemoryIndexValueT &I,
                  ArrayRef<const InMemoryIndexValueT *> Refs,
                  ArrayRef<char> Data)
      : Index(I), NumRefs(Refs.size()), DataSize(Data.size()) {
    auto *BeginRefs = reinterpret_cast<const InMemoryIndexValueT **>(this + 1);
    llvm::copy(Refs, BeginRefs);
    auto *BeginData = reinterpret_cast<char *>(BeginRefs + NumRefs);
    llvm::copy(Data, BeginData);
    BeginData[Data.size()] = 0;
  }

  const InMemoryIndexValueT &Index;
  uint32_t NumRefs;
  uint32_t DataSize;
};

class GRPCCASContext : public CASContext {
  void printIDImpl(raw_ostream &OS, const CASID &ID) const final;
  void anchor() override;

public:
  static StringRef getHashName() { return "RemoteCAS"; }
  StringRef getHashSchemaIdentifier() const final {
    static const std::string ID =
        ("grpc::cas::service::v1[" + getHashName() + "]").str();
    return ID;
  }

  static const GRPCCASContext &getDefaultContext();

  GRPCCASContext() = default;
};

class GRPCRelayCAS : public ObjectStore {
public:
  GRPCRelayCAS(StringRef Path, Error &Err);

  ArrayRef<uint8_t> getHashImpl(const CASID &ID) const;

  // ObjectStore interfaces.
  Expected<CASID> parseID(StringRef ID) final;
  Expected<ObjectRef> store(ArrayRef<ObjectRef> Refs,
                               ArrayRef<char> Data) final;
  CASID getID(ObjectRef Ref) const final;
  CASID getID(ObjectHandle Handle) const final;
  Optional<ObjectRef> getReference(const CASID &ID) const final;
  Expected<ObjectHandle> load(ObjectRef Ref) final;
  Error validate(const CASID &ID) final {
    // Not supported yet. Always return success.
    return Error::success();
  }
  uint64_t getDataSize(ObjectHandle Node) const final;
  Error forEachRef(ObjectHandle Node,
                   function_ref<Error(ObjectRef)> Callback) const final;
  ObjectRef readRef(ObjectHandle Node, size_t I) const final;
  size_t getNumRefs(ObjectHandle Node) const final;
  ArrayRef<char> getData(ObjectHandle Node,
                         bool RequiresNullTerminator = false) const final;

  // For sending file path through grpc.
  Expected<ObjectRef>
  storeFromOpenFileImpl(sys::fs::file_t FD,
                        Optional<sys::fs::file_status> Status) override;

private:
  InMemoryIndexValueT &indexHash(ArrayRef<uint8_t> Hash) const {
    assert(Hash.size() == ServiceHashSize && "unexpected hash size");
    SmallVector<uint8_t, sizeof(InMemoryIndexT::HashT)> ExtendedHash(
        sizeof(InMemoryIndexT::HashT));
    llvm::copy(Hash, ExtendedHash.begin());
    return *Index.insertLazy(ExtendedHash, [](auto ValueConstructor) {
      ValueConstructor.emplace(nullptr);
    });
  }

  CASID getID(const InMemoryIndexValueT &I) const {
    ArrayRef<uint8_t> Hash = I.Data->getHash().take_front(ServiceHashSize);
    return CASID::create(&getContext(), toStringRef(Hash));
  }

  const InMemoryCASData &asInMemoryCASData(ObjectHandle Ref) const {
    uintptr_t P = Ref.getInternalRef(*this);
    return *reinterpret_cast<const InMemoryCASData *>(P);
  }

  InMemoryIndexValueT &asInMemoryIndexValue(ObjectRef Ref) const {
    uintptr_t P = Ref.getInternalRef(*this);
    return *reinterpret_cast<InMemoryIndexValueT *>(P);
  }

  ObjectRef toReference(const InMemoryCASData &O) const {
    return toReference(O.getIndex());
  }

  ObjectRef toReference(const InMemoryIndexValueT &I) const {
    return makeObjectRef(reinterpret_cast<uintptr_t>(&I));
  }

  const InMemoryCASData &getInMemoryCASData(ObjectHandle OH) const {
    return *reinterpret_cast<const InMemoryCASData *>(
        (uintptr_t)OH.getInternalRef(*this));
  }

  ObjectHandle getObjectHandle(const InMemoryCASData &Node) const {
    assert(!(reinterpret_cast<uintptr_t>(&Node) & 0x1ULL));
    return makeObjectHandle(reinterpret_cast<uintptr_t>(&Node));
  }

  const InMemoryCASData &
  storeObjectImpl(InMemoryIndexValueT &I,
                  ArrayRef<const InMemoryIndexValueT *> Refs,
                  ArrayRef<char> Data) {
    // Load or generate.
    auto Allocator = [&](size_t Size) -> void * {
      return Alloc.Allocate(Size, alignof(InMemoryCASData));
    };
    auto Generator = [&]() -> const InMemoryCASData * {
      return &InMemoryCASData::create(Allocator, I, Refs, Data);
    };
    return I.Data.loadOrGenerate(Generator);
  }

  std::string getDataIDFromRef(ObjectRef Ref) {
    StringRef Hash =
        toStringRef(asInMemoryIndexValue(Ref).Hash).take_front(ServiceHashSize);
    return Hash.str();
  }

  // Stub for compile_cache_service.
  std::unique_ptr<remote::CASDBClient> CASDB;

  // Index to manage the remote CAS.
  mutable InMemoryIndexT Index;
  // Allocator for CAS content.
  ThreadSafeAllocator<BumpPtrAllocator> Alloc;

  // Store the size of the hash.
  size_t ServiceHashSize;
};

class GRPCActionCache : public ActionCache {
public:
  GRPCActionCache(StringRef Path, Error &Err);

  Expected<Optional<CASID>> getImpl(ArrayRef<uint8_t> ResolvedKey) const final;
  Error putImpl(ArrayRef<uint8_t> ResolvedKey, const CASID &Result) final;

private:
  std::unique_ptr<remote::KeyValueDBClient> KVDB;
};

} // namespace

const GRPCCASContext &GRPCCASContext::getDefaultContext() {
  static GRPCCASContext DefaultContext;
  return DefaultContext;
}
void GRPCCASContext::anchor() {}

GRPCRelayCAS::GRPCRelayCAS(StringRef Path, Error &Err)
    : ObjectStore(GRPCCASContext::getDefaultContext()) {
  ErrorAsOutParameter ErrAsOutParam(&Err);
  auto DB = remote::createRemoteCASDBClient(Path);
  if (!DB) {
    Err = DB.takeError();
    return;
  }
  CASDB = std::move(*DB);

  // Send a put request to the CAS to determine hash size.
  auto Response = CASDB->saveDataSync("");
  if (!Response) {
    Err = Response.takeError();
    return;
  }
  ServiceHashSize = Response->size();
  if (ServiceHashSize > 80)
    Err = createStringError(std::make_error_code(std::errc::message_size),
                            "Hash from CASService is too long");
}

ArrayRef<uint8_t> GRPCRelayCAS::getHashImpl(const CASID &ID) const {
  ArrayRef<uint8_t> ExtendedHash(ID.getHash());
  return ExtendedHash.take_front(ServiceHashSize);
}

// FIXME: Just print out as base64 encode for now. Should move this into GRPC
// protocol.
void GRPCCASContext::printIDImpl(raw_ostream &OS, const CASID &ID) const {
  assert(&ID.getContext() == this);
  OS << encodeBase64(ID.getHash());
}

Expected<CASID> GRPCRelayCAS::parseID(StringRef Reference) {
  std::vector<char> Binary;
  if (Error Err = decodeBase64(Reference, Binary))
    return std::move(Err);
  return getID(indexHash(arrayRefFromStringRef(toStringRef(Binary))));
}

Expected<ObjectRef> GRPCRelayCAS::store(ArrayRef<ObjectRef> Refs,
                                        ArrayRef<char> Data) {
  SmallVector<std::string> RefIDs;
  for (auto Ref: Refs)
    RefIDs.emplace_back(getDataIDFromRef(Ref));

  auto Response = CASDB->putDataSync(toStringRef(Data).str(), RefIDs);
  if (!Response)
    return Response.takeError();

  auto &I = indexHash(arrayRefFromStringRef(*Response));
  // Create the node.
  SmallVector<const InMemoryIndexValueT *> InternalRefs;
  for (ObjectRef Ref : Refs)
    InternalRefs.push_back(&asInMemoryIndexValue(Ref));

  return toReference(storeObjectImpl(I, InternalRefs, Data));
}

CASID GRPCRelayCAS::getID(ObjectRef Ref) const {
  return getID(asInMemoryIndexValue(Ref));
}

CASID GRPCRelayCAS::getID(ObjectHandle Handle) const {
  return getID(asInMemoryCASData(Handle).getIndex());
}

Optional<ObjectRef> GRPCRelayCAS::getReference(const CASID &ID) const {
  assert(ID.getContext().getHashSchemaIdentifier() ==
             getContext().getHashSchemaIdentifier() &&
         "Expected ID from same hash schema");
  auto &I = indexHash(ID.getHash());
  return toReference(I);
}

Expected<ObjectHandle> GRPCRelayCAS::load(ObjectRef Ref) {
  auto &I = asInMemoryIndexValue(Ref);

  // Return the existing value if we have one.
  if (const InMemoryCASData *Existing = I.Data.load())
    return getObjectHandle(*Existing);

  // Send the request to remote to load the object. Since Generator function
  // can't return nullptr to indicate failed load now, we can't lock the entry
  // into busy state. we will send parallel loads and put one of the result into
  // the local CAS.
  auto Response = CASDB->getSync(getDataIDFromRef(Ref));
  if (!Response)
    return Response.takeError();

  SmallVector<const InMemoryIndexValueT *> InternalRefs;
  for (auto &ID : Response->Refs) {
    auto &Ref = indexHash(arrayRefFromStringRef(ID));
    InternalRefs.push_back(&Ref);
  }
  if (!Response->BlobData)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Expect result to be returned in buffer");

  ArrayRef<char> Content(Response->BlobData->data(),
                         Response->BlobData->size());
  return getObjectHandle(storeObjectImpl(I, InternalRefs, Content));
}

uint64_t GRPCRelayCAS::getDataSize(ObjectHandle Handle) const {
  return getInMemoryCASData(Handle).getDataSize();
}

Error GRPCRelayCAS::forEachRef(ObjectHandle Handle,
                               function_ref<Error(ObjectRef)> Callback) const {
  auto &Node = getInMemoryCASData(Handle);
  for (const InMemoryIndexValueT *Ref : Node.getRefs())
    if (Error E = Callback(toReference(*Ref)))
      return E;
  return Error::success();
}

ObjectRef GRPCRelayCAS::readRef(ObjectHandle Node, size_t I) const {
  return toReference(*getInMemoryCASData(Node).getRefs()[I]);
}

size_t GRPCRelayCAS::getNumRefs(ObjectHandle Handle) const {
  return getInMemoryCASData(Handle).getNumRefs();
}

ArrayRef<char> GRPCRelayCAS::getData(ObjectHandle Handle,
                                     bool RequiresNullTerminator) const {
  return getInMemoryCASData(Handle).getData();
}

Expected<ObjectRef>
GRPCRelayCAS::storeFromOpenFileImpl(sys::fs::file_t FD,
                                    Optional<sys::fs::file_status> FS) {
  std::error_code EC;
  sys::fs::mapped_file_region Map(FD, sys::fs::mapped_file_region::readonly,
                                  FS->getSize(),
                                  /*offset=*/0, EC);
  if (EC)
    return errorCodeToError(EC);

  ArrayRef<char> Data(Map.data(), Map.size());
  SmallString<128> Path;
  Optional<std::string> Response;
  if (sys::fs::getRealPathFromHandle(FD, Path)) {
    if (auto Err = CASDB->saveFileSync(Path.str().str()).moveInto(Response))
      return std::move(Err);
  } else {
    if (auto Err =
            CASDB->saveDataSync(toStringRef(Data).str()).moveInto(Response))
      return std::move(Err);
  }

  auto &I = indexHash(arrayRefFromStringRef(*Response));
  // TODO: we can avoid the copy by implementing InMemoryRef object like
  // InMemoryCAS.
  return toReference(storeObjectImpl(I, None, Data));
}

GRPCActionCache::GRPCActionCache(StringRef Path, Error &Err)
    : ActionCache(GRPCCASContext::getDefaultContext()) {
  ErrorAsOutParameter ErrAsOutParam(&Err);
  auto Cache = remote::createRemoteKeyValueClient(Path);
  if (!Cache) {
    Err = Cache.takeError();
    return;
  }
  KVDB = std::move(*Cache);
}

Expected<Optional<CASID>>
GRPCActionCache::getImpl(ArrayRef<uint8_t> ResolvedKey) const {
  auto Response = KVDB->getValueSync(ResolvedKey);
  if (!Response)
    return Response.takeError();
  if (!*Response)
    return None;

  auto Result = (*Response)->find("CASID");
  if (Result == (*Response)->end())
    return createStringError(
        inconvertibleErrorCode(),
        "malformed value object returned by remote service");
  std::string Value = Result->getValue();
  return CASID::create(&getContext(), Value);
}

Error GRPCActionCache::putImpl(ArrayRef<uint8_t> ResolvedKey,
                               const CASID &Result) {
  remote::KeyValueDBClient::ValueTy CompResult;
  CompResult["CASID"] = toStringRef(Result.getHash()).str();
  if (auto Err = KVDB->putValueSync(ResolvedKey, CompResult))
    return Err;

  return Error::success();
}

Expected<std::unique_ptr<ObjectStore>>
cas::createGRPCRelayCAS(const Twine &Path) {
  Error Err = Error::success();
  auto CAS = std::make_unique<GRPCRelayCAS>(Path.str(), Err);
  if (Err)
    return std::move(Err);

  return CAS;
}

Expected<std::unique_ptr<cas::ActionCache>>
cas::createGRPCActionCache(StringRef Path) {
  Error Err = Error::success();
  auto Cache = std::make_unique<GRPCActionCache>(Path, Err);
  if (Err)
    return std::move(Err);
  return Cache;
}
