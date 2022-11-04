//===- llvm/CAS/TreeEntry.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_TREEENTRY_H
#define LLVM_CAS_TREEENTRY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/CASReference.h"

namespace llvm {
namespace cas {

class ObjectStore;

class TreeEntry {
public:
  enum EntryKind {
    Regular,    /// A file.
    Executable, /// A file that's executable.
    Symlink,    /// A symbolic link.
    Tree,       /// A filesystem tree.
  };

  EntryKind getKind() const { return Kind; }
  bool isRegular() const { return Kind == Regular; }
  bool isExecutable() const { return Kind == Executable; }
  bool isFile() const { return isRegular() || isExecutable(); }
  bool isSymlink() const { return Kind == Symlink; }
  bool isTree() const { return Kind == Tree; }

  ObjectRef getRef() const { return Ref; }

  friend bool operator==(const TreeEntry &LHS, const TreeEntry &RHS) {
    return LHS.Kind == RHS.Kind && LHS.Ref == RHS.Ref;
  }

  TreeEntry(ObjectRef Ref, EntryKind Kind) : Kind(Kind), Ref(Ref) {}

private:
  EntryKind Kind;
  ObjectRef Ref;
};

class NamedTreeEntry : public TreeEntry {
public:
  StringRef getName() const { return Name; }

  friend bool operator==(const NamedTreeEntry &LHS, const NamedTreeEntry &RHS) {
    return static_cast<const TreeEntry &>(LHS) == RHS && LHS.Name == RHS.Name;
  }

  friend bool operator<(const NamedTreeEntry &LHS, const NamedTreeEntry &RHS) {
    return LHS.Name < RHS.Name;
  }

  NamedTreeEntry(ObjectRef Ref, EntryKind Kind, StringRef Name)
      : TreeEntry(Ref, Kind), Name(Name) {}

  void print(raw_ostream &OS, ObjectStore &CAS) const;

private:
  StringRef Name;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_TREEENTRY_H
