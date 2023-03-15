//===- ObjectStore.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ObjectStore.h"
#include "BuiltinCAS.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CAS/UnifiedOnDiskCache.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"

using namespace llvm;
using namespace llvm::cas;

void CASContext::anchor() {}
void ObjectStore::anchor() {}

LLVM_DUMP_METHOD void CASID::dump() const { print(dbgs()); }
LLVM_DUMP_METHOD void ObjectStore::dump() const { print(dbgs()); }
LLVM_DUMP_METHOD void ObjectRef::dump() const { print(dbgs()); }
LLVM_DUMP_METHOD void ObjectHandle::dump() const { print(dbgs()); }

std::string CASID::toString() const {
  std::string S;
  raw_string_ostream(S) << *this;
  return S;
}

static void printReferenceBase(raw_ostream &OS, StringRef Kind,
                               uint64_t InternalRef, Optional<CASID> ID) {
  OS << Kind << "=" << InternalRef;
  if (ID)
    OS << "[" << *ID << "]";
}

void ReferenceBase::print(raw_ostream &OS, const ObjectHandle &This) const {
  assert(this == &This);
  printReferenceBase(OS, "object-handle", InternalRef, std::nullopt);
}

void ReferenceBase::print(raw_ostream &OS, const ObjectRef &This) const {
  assert(this == &This);

  Optional<CASID> ID;
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  if (CAS)
    ID = CAS->getID(This);
#endif
  printReferenceBase(OS, "object-ref", InternalRef, ID);
}

void ObjectStore::loadIfExistsAsync(
    ObjectRef Ref,
    unique_function<void(Expected<std::optional<ObjectHandle>>)> Callback) {
  // The default implementation is synchronous.
  Callback(loadIfExists(Ref));
}

Expected<ObjectHandle> ObjectStore::load(ObjectRef Ref) {
  std::optional<ObjectHandle> Handle;
  if (Error E = loadIfExists(Ref).moveInto(Handle))
    return std::move(E);
  if (!Handle)
    return createStringError(errc::invalid_argument,
                             "missing object '" + getID(Ref).toString() + "'");
  return *Handle;
}

std::unique_ptr<MemoryBuffer>
ObjectStore::getMemoryBuffer(ObjectHandle Node, StringRef Name,
                             bool RequiresNullTerminator) {
  return MemoryBuffer::getMemBuffer(
      toStringRef(getData(Node, RequiresNullTerminator)), Name,
      RequiresNullTerminator);
}

void ObjectStore::readRefs(ObjectHandle Node,
                           SmallVectorImpl<ObjectRef> &Refs) const {
  consumeError(forEachRef(Node, [&Refs](ObjectRef Ref) -> Error {
    Refs.push_back(Ref);
    return Error::success();
  }));
}

Expected<ObjectProxy> ObjectStore::getProxy(const CASID &ID) {
  Optional<ObjectRef> Ref = getReference(ID);
  if (!Ref)
    return createUnknownObjectError(ID);

  return getProxy(*Ref);
}

Expected<ObjectProxy> ObjectStore::getProxy(ObjectRef Ref) {
  Optional<ObjectHandle> H;
  if (Error E = load(Ref).moveInto(H))
    return std::move(E);

  return ObjectProxy::load(*this, Ref, *H);
}

Expected<std::optional<ObjectProxy>>
ObjectStore::getProxyIfExists(ObjectRef Ref) {
  Optional<ObjectHandle> H;
  if (Error E = loadIfExists(Ref).moveInto(H))
    return std::move(E);
  if (!H)
    return std::nullopt;
  return ObjectProxy::load(*this, Ref, *H);
}

std::future<AsyncProxyValue> ObjectStore::getProxyAsync(ObjectRef Ref) {
  std::promise<AsyncProxyValue> Promise;
  auto Future = Promise.get_future();
  // FIXME: there is potential for use-after-free for the 'this' pointer.
  // Either we should always allocate shared pointers for \c ObjectStore objects
  // and pass \c shared_from_this() or expect that the caller will not release
  // the \c ObjectStore before the callback returns.
  loadIfExistsAsync(Ref, [this, Ref, Promise = std::move(Promise)](
                             Expected<std::optional<ObjectHandle>> H) mutable {
    if (!H)
      Promise.set_value(H.takeError());
    else if (!*H)
      Promise.set_value(std::nullopt);
    else
      Promise.set_value(ObjectProxy::load(*this, Ref, **H));
  });
  return Future;
}

Error ObjectStore::createUnknownObjectError(const CASID &ID) {
  return createStringError(std::make_error_code(std::errc::invalid_argument),
                           "unknown object '" + ID.toString() + "'");
}

Expected<ObjectProxy> ObjectStore::createProxy(ArrayRef<ObjectRef> Refs,
                                               StringRef Data) {
  Expected<ObjectRef> Ref = store(Refs, arrayRefFromStringRef<char>(Data));
  if (!Ref)
    return Ref.takeError();
  return getProxy(*Ref);
}

Expected<ObjectRef>
ObjectStore::storeFromOpenFileImpl(sys::fs::file_t FD,
                                   Optional<sys::fs::file_status> Status) {
  // Copy the file into an immutable memory buffer and call \c store on that.
  // Using \c mmap would be unsafe because there's a race window between when we
  // get the digest hash for the \c mmap contents and when we store the data; if
  // the file changes in-between we will create an invalid object.

  // FIXME: For the on-disk CAS implementation use cloning to store it as a
  // standalone file if the file-system supports it and the file is large.

  constexpr size_t ChunkSize = 4 * 4096;
  SmallString<0> Data;
  Data.reserve(ChunkSize * 2);
  if (Error E = sys::fs::readNativeFileToEOF(FD, Data, ChunkSize))
    return std::move(E);
  return store(std::nullopt, ArrayRef(Data.data(), Data.size()));
}

Error ObjectStore::validateTree(ObjectRef Root) {
  SmallDenseSet<ObjectRef> ValidatedRefs;
  SmallVector<ObjectRef, 16> RefsToValidate;
  RefsToValidate.push_back(Root);

  while (!RefsToValidate.empty()) {
    ObjectRef Ref = RefsToValidate.pop_back_val();
    auto [I, Inserted] = ValidatedRefs.insert(Ref);
    if (!Inserted)
      continue; // already validated.
    if (Error E = validate(getID(Ref)))
      return E;
    Expected<ObjectHandle> Obj = load(Ref);
    if (!Obj)
      return Obj.takeError();
    if (Error E = forEachRef(*Obj, [&RefsToValidate](ObjectRef R) -> Error {
          RefsToValidate.push_back(R);
          return Error::success();
        }))
      return E;
  }
  return Error::success();
}

std::unique_ptr<MemoryBuffer>
ObjectProxy::getMemoryBuffer(StringRef Name,
                             bool RequiresNullTerminator) const {
  return CAS->getMemoryBuffer(H, Name, RequiresNullTerminator);
}

static Expected<std::unique_ptr<ObjectStore>>
createInMemoryCASImpl(const Twine &) {
  return createInMemoryCAS();
}

static ManagedStatic<StringMap<ObjectStoreCreateFuncTy *>> RegisteredScheme;

static StringMap<ObjectStoreCreateFuncTy *> &getRegisteredScheme() {
  if (!RegisteredScheme.isConstructed()) {
    RegisteredScheme->insert({"mem://", &createInMemoryCASImpl});
    RegisteredScheme->insert({"file://", &createOnDiskCAS});
  }
  return *RegisteredScheme;
}

Expected<std::unique_ptr<ObjectStore>>
cas::createCASFromIdentifier(StringRef Path) {
  for (auto &Scheme : getRegisteredScheme()) {
    if (Path.consume_front(Scheme.getKey()))
      return Scheme.getValue()(Path);
  }

  if (Path.empty())
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "No CAS identifier is provided");

  // FIXME: some current default behavior.
  if (Path == "auto")
    return createOnDiskCAS(getDefaultOnDiskCASPath());

  // Fallback is to create UnifiedOnDiskCache.
  auto UniDB = builtin::createBuiltinUnifiedOnDiskCache(Path);
  if (!UniDB)
    return UniDB.takeError();
  return builtin::createObjectStoreFromUnifiedOnDiskCache(std::move(*UniDB));
}

void cas::registerCASURLScheme(StringRef Prefix,
                               ObjectStoreCreateFuncTy *Func) {
  getRegisteredScheme().insert({Prefix, Func});
}
