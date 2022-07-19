//===- llvm/CAS/TreeSchema.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_TREESCHEMA_H
#define LLVM_CAS_TREESCHEMA_H

#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/CASNodeSchema.h"
#include "llvm/CAS/TreeEntry.h"

namespace llvm {
namespace cas {

class TreeProxy;

class TreeSchema : public RTTIExtends<TreeSchema, NodeSchema> {
  void anchor() override;

public:
  static char ID;
  bool isRootNode(const ObjectHandle &Node) const final {
    return false; // TreeSchema doesn't have a root node.
  }
  bool isNode(const ObjectHandle &Node) const final;

  TreeSchema(CASDB &CAS);

  size_t getNumTreeEntries(TreeProxy Tree) const;

  Error
  forEachTreeEntry(TreeProxy Tree,
                   function_ref<Error(const NamedTreeEntry &)> Callback) const;

  /// Visit each file entry in order, returning an error from \p Callback to
  /// stop early.
  ///
  /// The \p NamedTreeEntry, that the \p Callback receives, points to a name
  /// string that may not live beyond the return of the callback function.
  ///
  /// Passes the \p TreeNodeProxy if the entry is a \p TreeEntry::Tree,
  /// otherwise passes \p None.
  Error walkFileTreeRecursively(
      CASDB &CAS, const ObjectHandle &Root,
      function_ref<Error(const NamedTreeEntry &, Optional<TreeProxy>)>
          Callback);

  Optional<size_t> lookupTreeEntry(TreeProxy Tree, StringRef Name) const;
  NamedTreeEntry loadTreeEntry(TreeProxy Tree, size_t I) const;

  Expected<TreeProxy> load(ObjectRef Object) const;
  Expected<TreeProxy> load(ObjectHandle Object) const;

  Expected<TreeProxy> create(ArrayRef<NamedTreeEntry> Entries = None);

private:
  static constexpr StringLiteral SchemaName = "llvm::cas::schema::tree::v1";
  Optional<ObjectRef> TreeKindRef;

  friend class TreeProxy;

  ObjectRef getKindRef() const;
};

class TreeProxy : public ObjectProxy {
public:
  static Expected<TreeProxy> get(const TreeSchema &Schema,
                                     Expected<ObjectProxy> Ref);

  static Expected<TreeProxy> create(TreeSchema &Schema,
                                        ArrayRef<NamedTreeEntry> Entries);

  const TreeSchema &getSchema() const { return *Schema; }

  bool operator==(const TreeProxy &RHS) const {
    return Schema == RHS.Schema && cas::CASID(*this) == cas::CASID(RHS);
  }

  Error
  forEachEntry(function_ref<Error(const NamedTreeEntry &)> Callback) const {
    return Schema->forEachTreeEntry(*this, Callback);
  }

  bool empty() const { return size() == 0; }
  size_t size() const { return Schema->getNumTreeEntries(*this); }

  Optional<NamedTreeEntry> lookup(StringRef Name) const {
    if (auto I = Schema->lookupTreeEntry(*this, Name))
      return get(*I);
    return None;
  }

  StringRef getName(size_t I) const;

  NamedTreeEntry get(size_t I) const { return Schema->loadTreeEntry(*this, I); }

  TreeProxy() = delete;

private:
  TreeProxy(const TreeSchema &Schema, const ObjectProxy &Node)
      : ObjectProxy(Node), Schema(&Schema) {}

  class Builder {
  public:
    static Expected<Builder> startNode(TreeSchema &Schema);

    Expected<TreeProxy> build(ArrayRef<NamedTreeEntry> Entries);

  private:
    Builder(const TreeSchema &Schema) : Schema(&Schema) {}
    const TreeSchema *Schema;

  public:
    SmallString<256> Data;
    SmallVector<ObjectRef, 16> Refs;
  };
  const TreeSchema *Schema;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_TREESCHEMA_H
