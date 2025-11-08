//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ObjectStore.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <deque>

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
                               uint64_t InternalRef, std::optional<CASID> ID) {
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

  std::optional<CASID> ID;
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  if (CAS)
    ID = CAS->getID(This);
#endif
  printReferenceBase(OS, "object-ref", InternalRef, ID);
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
  std::optional<ObjectRef> Ref = getReference(ID);
  if (!Ref)
    return createUnknownObjectError(ID);

  return getProxy(*Ref);
}

Expected<ObjectProxy> ObjectStore::getProxy(ObjectRef Ref) {
  std::optional<ObjectHandle> H;
  if (Error E = load(Ref).moveInto(H))
    return std::move(E);

  return ObjectProxy::load(*this, Ref, *H);
}

Expected<std::optional<ObjectProxy>>
ObjectStore::getProxyIfExists(ObjectRef Ref) {
  std::optional<ObjectHandle> H;
  if (Error E = loadIfExists(Ref).moveInto(H))
    return std::move(E);
  if (!H)
    return std::nullopt;
  return ObjectProxy::load(*this, Ref, *H);
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
                                   std::optional<sys::fs::file_status> Status) {
  // TODO: For the on-disk CAS implementation use cloning to store it as a
  // standalone file if the file-system supports it and the file is large.
  uint64_t Size = Status ? Status->getSize() : -1;
  auto Buffer = MemoryBuffer::getOpenFile(FD, /*Filename=*/"", Size);
  if (!Buffer)
    return errorCodeToError(Buffer.getError());

  return store({}, arrayRefFromStringRef<char>((*Buffer)->getBuffer()));
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
    if (Error E = validateObject(getID(Ref)))
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

Expected<ObjectRef> ObjectStore::importObject(ObjectStore &Upstream,
                                              ObjectRef Other) {
  // Copy the full CAS tree from upstream with depth-first ordering to ensure
  // all the child nodes are available in downstream CAS before inserting
  // current object. This uses a similar algorithm as
  // `OnDiskGraphDB::importFullTree` but doesn't assume the upstream CAS schema
  // so it can be used to import from any other ObjectStore reguardless of the
  // CAS schema.

  // There is no work to do if importing from self.
  if (this == &Upstream)
    return Other;

  /// Keeps track of the state of visitation for current node and all of its
  /// parents. Upstream Cursor holds information only from upstream CAS.
  struct UpstreamCursor {
    ObjectRef Ref;
    ObjectHandle Node;
    size_t RefsCount;
    std::deque<ObjectRef> Refs;
  };
  SmallVector<UpstreamCursor, 16> CursorStack;
  /// PrimaryNodeStack holds the ObjectRef of the current CAS, with nodes either
  /// just stored in the CAS or nodes already exists in the current CAS.
  SmallVector<ObjectRef, 128> PrimaryRefStack;
  /// A map from upstream ObjectRef to current ObjectRef.
  llvm::DenseMap<ObjectRef, ObjectRef> CreatedObjects;

  auto enqueueNode = [&](ObjectRef Ref, ObjectHandle Node) {
    unsigned NumRefs = Upstream.getNumRefs(Node);
    std::deque<ObjectRef> Refs;
    for (unsigned I = 0; I < NumRefs; ++I)
      Refs.push_back(Upstream.readRef(Node, I));

    CursorStack.push_back({Ref, Node, NumRefs, std::move(Refs)});
  };

  auto UpstreamHandle = Upstream.load(Other);
  if (!UpstreamHandle)
    return UpstreamHandle.takeError();
  enqueueNode(Other, *UpstreamHandle);

  while (!CursorStack.empty()) {
    UpstreamCursor &Cur = CursorStack.back();
    if (Cur.Refs.empty()) {
      // Copy the node data into the primary store.
      // The bottom of \p PrimaryRefStack contains the ObjectRef for the
      // current node.
      assert(PrimaryRefStack.size() >= Cur.RefsCount);
      auto Refs = ArrayRef(PrimaryRefStack)
                      .slice(PrimaryRefStack.size() - Cur.RefsCount);
      auto NewNode = store(Refs, Upstream.getData(Cur.Node));
      if (!NewNode)
        return NewNode.takeError();

      // Remove the current node and its IDs from the stack.
      PrimaryRefStack.truncate(PrimaryRefStack.size() - Cur.RefsCount);

      // Push new node into created objects.
      PrimaryRefStack.push_back(*NewNode);
      CreatedObjects.try_emplace(Cur.Ref, *NewNode);

      // Pop the cursor in the end after all uses.
      CursorStack.pop_back();
      continue;
    }

    // Check if the node exists already.
    auto CurrentID = Cur.Refs.front();
    Cur.Refs.pop_front();
    auto Ref = CreatedObjects.find(CurrentID);
    if (Ref != CreatedObjects.end()) {
      // If exists already, just need to enqueue the primary node.
      PrimaryRefStack.push_back(Ref->second);
      continue;
    }

    // Load child.
    auto PrimaryID = Upstream.load(CurrentID);
    if (LLVM_UNLIKELY(!PrimaryID))
      return PrimaryID.takeError();

    enqueueNode(CurrentID, *PrimaryID);
  }

  assert(PrimaryRefStack.size() == 1);
  return PrimaryRefStack.front();
}

std::unique_ptr<MemoryBuffer>
ObjectProxy::getMemoryBuffer(StringRef Name,
                             bool RequiresNullTerminator) const {
  return CAS->getMemoryBuffer(H, Name, RequiresNullTerminator);
}
