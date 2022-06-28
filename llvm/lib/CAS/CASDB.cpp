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
    ID = CAS->getObjectID(This);
#endif
  printReferenceBase(OS, "object-handle", InternalRef, ID);
}

void ReferenceBase::print(raw_ostream &OS, const ObjectRef &This) const {
  assert(this == &This);

  Optional<CASID> ID;
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  if (CAS)
    ID = CAS->getObjectID(This);
#endif
  printReferenceBase(OS, "object-ref", InternalRef, ID);
}

/// Default implementation opens the file and calls \a createBlob().
Expected<NodeHandle>
CASDB::storeNodeFromOpenFileImpl(sys::fs::file_t FD,
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

  return storeNode(
      None, arrayRefFromStringRef<char>((*ExpectedContent)->getBuffer()));
}

Expected<Optional<AnyObjectHandle>> CASDB::loadObject(const CASID &ID) {
  if (Optional<ObjectRef> Ref = getReference(ID))
    return loadObject(*Ref);
  return None;
}

void CASDB::readRefs(NodeHandle Node, SmallVectorImpl<ObjectRef> &Refs) const {
  consumeError(forEachRef(Node, [&Refs](ObjectRef Ref) -> Error {
    Refs.push_back(Ref);
    return Error::success();
  }));
}

template <class ProxyT, class HandleT>
Expected<ProxyT> CASDB::loadObjectProxy(CASID ID) {
  Optional<AnyObjectHandle> H;
  if (Error E = loadObject(ID).moveInto(H))
    return std::move(E);
  if (!H)
    return createUnknownObjectError(ID);
  if (Optional<HandleT> Casted = H->dyn_cast<HandleT>())
    return ProxyT::load(*this, *Casted);
  return createWrongKindError(ID);
}

template <class ProxyT, class HandleT>
Expected<ProxyT> CASDB::loadObjectProxy(Expected<HandleT> H) {
  if (!H)
    return H.takeError();
  return ProxyT::load(*this, *H);
}

Expected<LeafNodeProxy> CASDB::getBlob(CASID ID) {
  Optional<AnyObjectHandle> Object;
  if (Error E = loadObject(ID).moveInto(Object))
    return std::move(E);
  if (!Object || !Object->is<NodeHandle>())
    return createWrongKindError(ID);
  Optional<NodeHandle> Node = Object->get<NodeHandle>();
  if (getNumRefs(*Node) > 0)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "node '" + ID.toString() + "' is not a leaf");
  return LeafNodeProxy(NodeProxy::load(*this, *Node));
}

Expected<TreeProxy> CASDB::loadTree(ObjectRef Ref) {
  Expected<AnyObjectHandle> Object = loadObject(Ref);
  if (!Object)
    return Object.takeError();
  Optional<TreeHandle> Tree = Object->dyn_cast<TreeHandle>();
  if (!Tree)
    return createWrongKindError(getObjectID(Ref));
  return TreeProxy::load(*this, *Tree);
}

Expected<LeafNodeProxy> CASDB::loadBlob(ObjectRef Ref) {
  Expected<AnyObjectHandle> Object = loadObject(Ref);
  if (!Object)
    return Object.takeError();
  Optional<NodeHandle> Node = Object->dyn_cast<NodeHandle>();
  if (!Node)
    return createWrongKindError(getObjectID(Ref));
  if (getNumRefs(*Node) > 0)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "node '" + getObjectID(Ref).toString() +
                                 "' is not a leaf");
  return LeafNodeProxy(NodeProxy::load(*this, *Node));
}

Expected<NodeProxy> CASDB::loadNode(ObjectRef Ref) {
  Expected<AnyObjectHandle> Object = loadObject(Ref);
  if (!Object)
    return Object.takeError();
  Optional<NodeHandle> Node = Object->dyn_cast<NodeHandle>();
  if (!Node)
    return createWrongKindError(getObjectID(Ref));
  return NodeProxy::load(*this, *Node);
}

Expected<TreeProxy> CASDB::getTree(CASID ID) {
  return loadObjectProxy<TreeProxy, TreeHandle>(ID);
}
Expected<NodeProxy> CASDB::getNode(CASID ID) {
  return loadObjectProxy<NodeProxy, NodeHandle>(ID);
}

Error CASDB::createUnknownObjectError(CASID ID) {
  return createStringError(std::make_error_code(std::errc::invalid_argument),
                           "unknown object '" + ID.toString() + "'");
}

Error CASDB::createWrongKindError(CASID ID) {
  return createStringError(std::make_error_code(std::errc::invalid_argument),
                           "wrong object kind '" + ID.toString() + "'");
}

Expected<LeafNodeProxy> CASDB::createBlob(StringRef Data) {
  Optional<NodeHandle> Node;
  if (Error E =
          storeNode(None, arrayRefFromStringRef<char>(Data)).moveInto(Node))
    return std::move(E);
  return LeafNodeProxy(NodeProxy::load(*this, *Node));
}

Expected<TreeProxy> CASDB::createTree(ArrayRef<NamedTreeEntry> Entries) {
  return loadObjectProxy<TreeProxy>(storeTree(Entries));
}

Expected<NodeProxy> CASDB::createNode(ArrayRef<CASID> IDs, StringRef Data) {
  SmallVector<ObjectRef> Refs;
  for (CASID ID : IDs) {
    if (Optional<ObjectRef> Ref = getReference(ID))
      Refs.push_back(*Ref);
    else
      return createUnknownObjectError(ID);
  }
  return loadObjectProxy<NodeProxy>(
      storeNode(Refs, arrayRefFromStringRef<char>(Data)));
}

Expected<std::unique_ptr<MemoryBuffer>>
CASDB::loadIndependentDataBuffer(NodeHandle Node, const Twine &Name,
                                 bool NullTerminate) const {
  return loadIndependentDataBufferImpl(Node, Name, NullTerminate);
}

Expected<std::unique_ptr<MemoryBuffer>>
CASDB::loadIndependentDataBufferImpl(NodeHandle Node, const Twine &Name,
                                     bool NullTerminate) const {
  SmallString<256> Bytes;
  raw_svector_ostream OS(Bytes);
  readData(Node, OS);
  return std::make_unique<SmallVectorMemoryBuffer>(std::move(Bytes), Name.str(),
                                                   NullTerminate);
}
