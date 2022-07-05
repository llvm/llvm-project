//===- llvm/CAS/CASNodeSchema.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASNODESCHEMA_H
#define LLVM_CAS_CASNODESCHEMA_H

#include "llvm/CAS/CASReference.h"
#include "llvm/Support/ExtensibleRTTI.h"

namespace llvm {
namespace cas {
/// A base class for schemas built on top of CAS nodes.
///
/// TODO: Build a FilesystemSchema on top of this for reimplementing Trees on
/// top of the CAS.
class NodeSchema : public RTTIExtends<NodeSchema, RTTIRoot> {
  void anchor() override;

public:
  static char ID;

  /// Check if \a Node is a root (entry node) for the schema. This is a strong
  /// check, since it requires that the first reference matches a complete
  /// type-id DAG.
  virtual bool isRootNode(const cas::ObjectHandle &Node) const = 0;

  virtual bool isNode(const cas::ObjectHandle &Node) const = 0;

  cas::CASDB &CAS;

protected:
  NodeSchema(cas::CASDB &CAS) : CAS(CAS) {}

public:
  virtual ~NodeSchema() = default;
};

/// Creates all the schemas and can be used to retrieve a particular schema
/// based on a CAS root node. A client should aim to create and maximize re-use
/// of an instance of this object.
class SchemaPool {
public:
  /// Look up the schema for the provided root node. Returns \a nullptr if no
  /// schema was found or it's not actually a root node. The returned \p
  /// NodeSchema pointer is owned by the \p SchemaPool instance, therefore it
  /// cannot be used beyond the \p SchemaPool instance's lifetime.
  ///
  /// Thread-safe.
  NodeSchema *getSchemaForRoot(cas::ObjectHandle Node) const;

  /// Add a schema to the pool.
  void addSchema(std::unique_ptr<NodeSchema> S) {
    Schemas.push_back(std::move(S));
  }

  cas::CASDB &getCAS() const { return CAS; }

  explicit SchemaPool(cas::CASDB &CAS) : CAS(CAS) {}

private:
  cas::CASDB &CAS;
  SmallVector<std::unique_ptr<NodeSchema>> Schemas;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_CASNODESCHEMA_H
