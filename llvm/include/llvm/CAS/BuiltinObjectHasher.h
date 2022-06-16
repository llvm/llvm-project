//===- BuiltinObjectHasher.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_BUILTINOBJECTHASHER_H
#define LLVM_CAS_BUILTINOBJECTHASHER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/TreeEntry.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace cas {

enum class StableObjectKind : uint8_t {
  Node = 1,
  Blob = 2,
  Tree = 3,
  String = 4,
};

enum class StableTreeEntryKind : uint8_t {
  Tree = 1,
  Regular = 2,
  Executable = 3,
  Symlink = 4,
};

inline StableTreeEntryKind getStableKind(TreeEntry::EntryKind Kind) {
  switch (Kind) {
  case TreeEntry::Tree:
    return StableTreeEntryKind::Tree;
  case TreeEntry::Regular:
    return StableTreeEntryKind::Regular;
  case TreeEntry::Executable:
    return StableTreeEntryKind::Executable;
  case TreeEntry::Symlink:
    return StableTreeEntryKind::Symlink;
  }
}

inline TreeEntry::EntryKind getUnstableKind(StableTreeEntryKind Kind) {
  switch (Kind) {
  case StableTreeEntryKind::Tree:
    return TreeEntry::Tree;
  case StableTreeEntryKind::Regular:
    return TreeEntry::Regular;
  case StableTreeEntryKind::Executable:
    return TreeEntry::Executable;
  case StableTreeEntryKind::Symlink:
    return TreeEntry::Symlink;
  }
}

template <class HasherT> class BuiltinObjectHasher {
public:
  using HashT = decltype(HasherT::hash(std::declval<ArrayRef<uint8_t> &>()));

  static HashT hashBlob(ArrayRef<char> Data) {
    BuiltinObjectHasher H;
    H.start(StableObjectKind::Blob);
    H.updateArray(Data);
    return H.finish();
  }

  static HashT hashString(StringRef String) {
    BuiltinObjectHasher H;
    H.start(StableObjectKind::String);
    H.updateString(String);
    return H.finish();
  }

  static HashT hashTree(const CASDB &CAS, ArrayRef<NamedTreeEntry> Entries) {
    BuiltinObjectHasher H;
    H.start(StableObjectKind::Tree);
    H.updateSize(Entries.size());
    for (const NamedTreeEntry &Entry : Entries) {
      H.updateRef(CAS, Entry.getRef());
      H.updateString(Entry.getName());
      H.updateKind(getStableKind(Entry.getKind()));
    }
    return H.finish();
  }

  static HashT hashNode(const CASDB &CAS, ArrayRef<ObjectRef> Refs,
                        ArrayRef<char> Data) {
    BuiltinObjectHasher H;
    H.start(StableObjectKind::Node);
    H.updateSize(Refs.size());
    for (const ObjectRef &Ref : Refs)
      H.updateRef(CAS, Ref);
    H.updateArray(Data);
    return H.finish();
  }

private:
  void start(StableObjectKind Kind) {
    updateKind(Kind);
  }

  HashT finish() { return Hasher.final(); }

  void updateKind(StableObjectKind Kind) {
    static_assert(sizeof(Kind) == 1, "Expected kind to be 1-byte");
    Hasher.update((uint8_t)Kind);
  }

  void updateKind(StableTreeEntryKind Kind) {
    static_assert(sizeof(Kind) == 1, "Expected kind to be 1-byte");
    Hasher.update((uint8_t)Kind);
  }

  void updateString(StringRef String) {
    updateArray(makeArrayRef(String.data(), String.size()));
  }

  void updateRef(const CASDB &CAS, ObjectRef Ref) {
    updateID(CAS.getObjectID(Ref));
  }

  void updateID(const CASID &ID) {
    // NOTE: Does not hash the size of the hash. That's a CAS implementation
    // detail that shouldn't leak into the UUID for an object.
    ArrayRef<uint8_t> Hash = ID.getHash();
    assert(Hash.size() == sizeof(HashT) &&
           "Expected object ref to match the hash size");
    Hasher.update(Hash);
  }

  void updateArray(ArrayRef<uint8_t> Bytes) {
    updateSize(Bytes.size());
    Hasher.update(Bytes);
  }

  void updateArray(ArrayRef<char> Bytes) {
    updateArray(makeArrayRef(reinterpret_cast<const uint8_t *>(Bytes.data()),
                             Bytes.size()));
  }

  void updateSize(uint64_t Size) {
    Size = support::endian::byte_swap(Size, support::endianness::little);
    Hasher.update(makeArrayRef(reinterpret_cast<const uint8_t *>(&Size),
                                sizeof(Size)));
  }

  BuiltinObjectHasher() = default;
  ~BuiltinObjectHasher() = default;
  HasherT Hasher;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_BUILTINOBJECTHASHER_H
