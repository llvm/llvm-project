//===- TreeSchema.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/TreeSchema.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::cas;

char TreeSchema::ID = 0;
constexpr StringLiteral TreeSchema::SchemaName;

void TreeSchema::anchor() {}

bool TreeSchema::isNode(const ObjectProxy &Node) const {
  // Load the first ref to check its content.
  if (Node.getNumReferences() < 1)
    return false;

  // If can't load the first ref, consume error and return false.
  auto FirstRef = Node.getReference(0);
  return FirstRef == getKindRef();
}

TreeSchema::TreeSchema(cas::ObjectStore &CAS) : TreeSchema::RTTIExtends(CAS) {
  TreeKindRef = cantFail(CAS.storeFromString(std::nullopt, SchemaName));
}

ObjectRef TreeSchema::getKindRef() const { return *TreeKindRef; }

size_t TreeSchema::getNumTreeEntries(TreeProxy Tree) const {
  return Tree.getNumReferences() - 1;
}

Error TreeSchema::forEachTreeEntry(
    TreeProxy Tree,
    function_ref<Error(const NamedTreeEntry &)> Callback) const {
  for (size_t I = 0, IE = getNumTreeEntries(Tree); I != IE; ++I)
    if (Error E = Callback(loadTreeEntry(Tree, I)))
      return E;

  return Error::success();
}

Error TreeSchema::walkFileTreeRecursively(
    ObjectStore &CAS, ObjectRef Root,
    function_ref<Error(const NamedTreeEntry &, std::optional<TreeProxy>)>
        Callback) {
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  SmallString<128> PathStorage;
  SmallVector<NamedTreeEntry> Stack;
  Stack.emplace_back(Root, TreeEntry::Tree, "/");

  while (!Stack.empty()) {
    if (Stack.back().getKind() != TreeEntry::Tree) {
      if (Error E = Callback(Stack.pop_back_val(), std::nullopt))
        return E;
      continue;
    }

    NamedTreeEntry Parent = Stack.pop_back_val();
    Expected<TreeProxy> ExpTree = load(Parent.getRef());
    if (Error E = ExpTree.takeError())
      return E;
    TreeProxy Tree = *ExpTree;
    if (Error E = Callback(Parent, Tree))
      return E;
    for (int I = Tree.size(), E = 0; I != E; --I) {
      std::optional<NamedTreeEntry> Child = Tree.get(I - 1);
      assert(Child && "Expected no corruption");

      PathStorage = Parent.getName();
      sys::path::append(PathStorage, sys::path::Style::posix, Child->getName());
      Stack.emplace_back(Child->getRef(), Child->getKind(),
                         Saver.save(StringRef(PathStorage)));
    }
  }

  return Error::success();
}

NamedTreeEntry TreeSchema::loadTreeEntry(TreeProxy Tree, size_t I) const {
  // Load entry from TreeNode.
  TreeEntry::EntryKind Kind =
      (TreeEntry::EntryKind)
          Tree.getData()[I + (Tree.size() + 1) * sizeof(uint32_t)];

  StringRef Name = Tree.getName(I);
  auto ObjectRef = Tree.getReference(I + 1);

  return {ObjectRef, Kind, Name};
}

std::optional<size_t> TreeSchema::lookupTreeEntry(TreeProxy Tree,
                                                  StringRef Name) const {
  size_t NumNames = Tree.size();
  if (!NumNames)
    return std::nullopt;

  // Start with a binary search, if there are enough entries.
  //
  // FIXME: Should just use std::lower_bound, but we need the actual iterators
  // to know the index in the NameCache...
  const size_t MaxLinearSearchSize = 4;
  size_t Last = NumNames;
  size_t First = 0;
  while (Last - First > MaxLinearSearchSize) {
    auto I = First + (Last - First) / 2;
    StringRef NameI = Tree.getName(I);
    switch (Name.compare(NameI)) {
    case 0:
      return I;
    case -1:
      Last = I;
      break;
    case 1:
      First = I + 1;
      break;
    }
  }

  // Use a linear search for small trees.
  for (; First != Last; ++First)
    if (Name == Tree.getName(First))
      return First;

  return std::nullopt;
}

Expected<TreeProxy> TreeSchema::load(ObjectRef Object) const {
  auto TreeNode = CAS.getProxy(Object);
  if (!TreeNode)
    return TreeNode.takeError();

  return load(*TreeNode);
}

Expected<TreeProxy> TreeSchema::load(ObjectProxy Object) const {
  if (!isNode(Object))
    return createStringError(inconvertibleErrorCode(), "not a tree object");

  return TreeProxy::get(*this, Object);
}

Expected<TreeProxy>
TreeSchema::create(ArrayRef<NamedTreeEntry> Entries) {
  return TreeProxy::create(*this, Entries);
}

Expected<TreeProxy> TreeProxy::get(const TreeSchema &Schema,
                                           Expected<ObjectProxy> Ref) {
  if (!Ref)
    return Ref.takeError();
  return TreeProxy(Schema, *Ref);
}

Expected<TreeProxy> TreeProxy::create(TreeSchema &Schema,
                                      ArrayRef<NamedTreeEntry> Entries) {
  auto B = Builder::startNode(Schema);
  if (!B)
    return B.takeError();

  return B->build(Entries);
}

StringRef TreeProxy::getName(size_t I) const {
  uint32_t StartIdx =
      support::endian::read32le(getData().data() + sizeof(uint32_t) * I);
  uint32_t EndIdx =
      support::endian::read32le(getData().data() + sizeof(uint32_t) * (I + 1));

  return StringRef(getData().data() + StartIdx, EndIdx - StartIdx);
}

Expected<TreeProxy::Builder>
TreeProxy::Builder::startNode(TreeSchema &Schema) {
  Builder B(Schema);
  B.Refs.push_back(Schema.getKindRef());
  return std::move(B);
}

Expected<TreeProxy>
TreeProxy::Builder::build(ArrayRef<NamedTreeEntry> Entries) {
  // Ensure a stable order for tree entries and ignore name collisions.
  SmallVector<NamedTreeEntry> Sorted(Entries.begin(), Entries.end());
  std::stable_sort(Sorted.begin(), Sorted.end());
  Sorted.erase(std::unique(Sorted.begin(), Sorted.end()), Sorted.end());

  raw_svector_ostream OS(Data);
  support::endian::Writer Writer(OS, endianness::little);
  // Encode the entires in the Data. The layout of the tree schema object is:
  // * Name offset table: The offset of in the data blob for where to find the
  //   string. It has N + 1 entries and you can find the name of n-th entry at
  //   offset[n] -> offset[n+1]. Each offset is encoded as little-endian
  //   uint32_t.
  // * Kind: uint8_t for each entry.
  // * Object: ObjectRef for each entry is at n + 1 refs for the object (with
  //   the first one being the tree kind ID).

  // Write Name.
  // The start of the string table index.
  uint32_t StrIdx =
      sizeof(uint8_t) * Sorted.size() + sizeof(uint32_t) * (Sorted.size() + 1);
  for (auto &Entry : Sorted) {
    Writer.write(StrIdx);
    StrIdx += Entry.getName().size();

    // Append refs.
    Refs.push_back(Entry.getRef());
  }
  // Write the end index for the last string.
  Writer.write(StrIdx);

  // Write Kind.
  for (auto &Entry : Sorted)
    Writer.write((uint8_t)Entry.getKind());

  // Write names in the end of the block.
  for (auto &Entry : Sorted)
    OS << Entry.getName();

  return TreeProxy::get(*Schema, Schema->CAS.createProxy(Refs, Data));
}
