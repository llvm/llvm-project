//===- CASDB.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASDB.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"

using namespace llvm;
using namespace llvm::cas;

void CASIDContext::anchor() {}
void CASDB::anchor() {}

LLVM_DUMP_METHOD void CASID::dump() const { print(dbgs()); }
LLVM_DUMP_METHOD void CASDB::dump() const { print(dbgs()); }
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

  Optional<CASID> ID;
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  if (CAS)
    ID = CAS->getID(This);
#endif
  printReferenceBase(OS, "object-handle", InternalRef, ID);
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

/// Default implementation opens the file and calls \a createBlob().
Expected<ObjectHandle>
CASDB::storeFromOpenFileImpl(sys::fs::file_t FD,
                             Optional<sys::fs::file_status> Status) {
  // Check whether we can trust the size from stat.
  int64_t FileSize = -1;
  if (Status->type() == sys::fs::file_type::regular_file ||
      Status->type() == sys::fs::file_type::block_file)
    FileSize = Status->getSize();

  // No need for a null terminator since the buffer will be dropped.
  ErrorOr<std::unique_ptr<MemoryBuffer>> ExpectedContent =
      MemoryBuffer::getOpenFile(FD, /*Filename=*/"", FileSize,
                                /*RequiresNullTerminator=*/false);
  if (!ExpectedContent)
    return errorCodeToError(ExpectedContent.getError());

  return store(None,
               arrayRefFromStringRef<char>((*ExpectedContent)->getBuffer()));
}

Expected<Optional<ObjectHandle>> CASDB::load(const CASID &ID) {
  if (Optional<ObjectRef> Ref = getReference(ID))
    return load(*Ref);
  return None;
}

void CASDB::readRefs(ObjectHandle Node,
                     SmallVectorImpl<ObjectRef> &Refs) const {
  consumeError(forEachRef(Node, [&Refs](ObjectRef Ref) -> Error {
    Refs.push_back(Ref);
    return Error::success();
  }));
}

Expected<ObjectProxy> CASDB::getProxy(CASID ID) {
  Optional<ObjectHandle> H;
  if (Error E = load(ID).moveInto(H))
    return std::move(E);
  if (!H)
    return createUnknownObjectError(ID);
  return ObjectProxy::load(*this, *H);
}

Expected<ObjectProxy> CASDB::getProxy(ObjectRef Ref) {
  return getProxy(load(Ref));
}

Expected<ObjectProxy> CASDB::getProxy(Expected<ObjectHandle> H) {
  if (!H)
    return H.takeError();
  return ObjectProxy::load(*this, *H);
}

Error CASDB::createUnknownObjectError(CASID ID) {
  return createStringError(std::make_error_code(std::errc::invalid_argument),
                           "unknown object '" + ID.toString() + "'");
}

Expected<ObjectProxy> CASDB::createProxy(ArrayRef<ObjectRef> Refs,
                                         StringRef Data) {
  return getProxy(store(Refs, arrayRefFromStringRef<char>(Data)));
}

Expected<std::unique_ptr<MemoryBuffer>>
CASDB::loadIndependentDataBuffer(ObjectHandle Node, const Twine &Name,
                                 bool NullTerminate) const {
  return loadIndependentDataBufferImpl(Node, Name, NullTerminate);
}

Expected<std::unique_ptr<MemoryBuffer>>
CASDB::loadIndependentDataBufferImpl(ObjectHandle Node, const Twine &Name,
                                     bool NullTerminate) const {
  SmallString<256> Bytes;
  raw_svector_ostream OS(Bytes);
  readData(Node, OS);
  return std::make_unique<SmallVectorMemoryBuffer>(std::move(Bytes), Name.str(),
                                                   NullTerminate);
}
