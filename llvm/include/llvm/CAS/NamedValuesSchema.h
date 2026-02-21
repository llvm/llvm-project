//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations for the NamedValuesSchema, a schema to
/// represent an array of named nodes inside CAS.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_NAMEDVALUESSCHEMA_H
#define LLVM_CAS_NAMEDVALUESSCHEMA_H

#include "llvm/CAS/CASNodeSchema.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"

namespace llvm::cas {

class NamedValuesProxy;

/// Represents an entry in NamedValuesSchema.
struct NamedValuesEntry {
  NamedValuesEntry(StringRef Name, ObjectRef Ref) : Name(Name), Ref(Ref) {}

  StringRef Name;
  ObjectRef Ref;

  friend bool operator==(const NamedValuesEntry &LHS,
                         const NamedValuesEntry &RHS) {
    return LHS.Ref == RHS.Ref && LHS.Name == RHS.Name;
  }

  /// Ordering the entries by name. Items should have unique names.
  friend bool operator<(const NamedValuesEntry &LHS,
                        const NamedValuesEntry &RHS) {
    return LHS.Name < RHS.Name;
  }
};

/// A schema for representing an array of named nodes in a CAS. The name of the
/// nodes are stored in the root node so child node can be loaded on demand
/// based on name and the name for all nodes need to be unique.
class LLVM_ABI NamedValuesSchema
    : public RTTIExtends<NamedValuesSchema, NodeSchema> {
  void anchor() override;

public:
  static char ID;

  bool isRootNode(const ObjectProxy &Node) const final {
    // NamedValuesSchema only has one node, thus root node.
    return isNode(Node);
  }

  /// Check if a proxy represents a valid node.
  bool isNode(const ObjectProxy &Node) const final;

  /// Create a NamedValuesSchema.
  static Expected<NamedValuesSchema> create(ObjectStore &CAS);

  /// Load NamedValuesProxy from an ObjectRef.
  Expected<NamedValuesProxy> load(ObjectRef Object) const;

  /// Load NamedValuesProxy from an ObjectProxy.
  Expected<NamedValuesProxy> load(ObjectProxy Object) const;

  /// Construct a \c NamedValuesSchema CAS object with the given entries.
  Expected<NamedValuesProxy> construct(ArrayRef<NamedValuesEntry> Entries);

  /// A builder class for creating nodes in NamedValuesSchema.
  class Builder {
  public:
    Builder(ObjectStore &CAS) : CAS(CAS) {}

    /// Add an entry to the builder.
    LLVM_ABI void add(StringRef Name, ObjectRef Ref);

    /// Build the node from added entries.
    LLVM_ABI Expected<NamedValuesProxy> build();

  private:
    ObjectStore &CAS;
    SmallVector<NamedValuesEntry> Nodes;
    BumpPtrAllocator Alloc;
  };

private:
  friend class NamedValuesProxy;

  NamedValuesSchema(ObjectStore &CAS, Error &E);

  /// Get the number of entries.
  size_t getNumEntries(NamedValuesProxy Values) const;

  /// Iterate over entries with a callback.
  Error
  forEachEntry(NamedValuesProxy Values,
               function_ref<Error(const NamedValuesEntry &)> Callback) const;

  /// Lookup an entry by name.
  std::optional<size_t> lookupEntry(NamedValuesProxy Values,
                                    StringRef Name) const;

  /// Load an entry by index.
  NamedValuesEntry loadEntry(NamedValuesProxy Values, size_t I) const;

  /// Name for the schema.
  static constexpr StringLiteral SchemaName =
      "llvm::cas::schema::namedvalues::v1";
  std::optional<ObjectRef> NamedValuesKindRef;
};

/// A proxy for a loaded CAS Object in NamedValuesSchema.
class NamedValuesProxy : public ObjectProxy {
public:
  /// Get the schema associated with this proxy.
  const NamedValuesSchema &getSchema() const { return *Schema; }

  /// Iterate over entries with a callback.
  Error
  forEachEntry(function_ref<Error(const NamedValuesEntry &)> Callback) const {
    return Schema->forEachEntry(*this, Callback);
  }

  /// Check if the object is empty.
  bool empty() const { return size() == 0; }

  /// Get the number of entries in the CAS object.
  size_t size() const { return Schema->getNumEntries(*this); }

  /// Lookup an entry by name.
  std::optional<NamedValuesEntry> lookup(StringRef Name) const {
    if (auto I = Schema->lookupEntry(*this, Name))
      return get(*I);
    return std::nullopt;
  }

  /// Get the name of an entry by index.
  LLVM_ABI StringRef getName(size_t I) const;

  /// Get an entry by index.
  NamedValuesEntry get(size_t I) const { return Schema->loadEntry(*this, I); }

private:
  NamedValuesProxy(const NamedValuesSchema &Schema, const ObjectProxy &Node)
      : ObjectProxy(Node), Schema(&Schema) {}

  friend class NamedValuesSchema;
  const NamedValuesSchema *Schema;
};

} // namespace llvm::cas

#endif
