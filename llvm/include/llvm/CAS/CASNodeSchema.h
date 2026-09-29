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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ExtensibleRTTI.h"

namespace llvm::cas {

class ObjectProxy;

/// A base class for schemas built on top of CAS nodes.
class LLVM_ABI NodeSchema : public RTTIExtends<NodeSchema, RTTIRoot> {
  void anchor() override;

public:
  static char ID;

  /// Check if \a Node is a root (entry node) for the schema. This is a strong
  /// check, since it requires that the first reference matches a complete
  /// type-id DAG.
  virtual bool isRootNode(const cas::ObjectProxy &Node) const = 0;

  /// Check if \a Node is a node for the schema. This can be any node that
  /// belongs to the schema.
  virtual bool isNode(const cas::ObjectProxy &Node) const = 0;

  cas::ObjectStore &CAS;

protected:
  NodeSchema(cas::ObjectStore &CAS) : CAS(CAS) {}

public:
  virtual ~NodeSchema() = default;
};

} // namespace llvm::cas

#endif // LLVM_CAS_CASNODESCHEMA_H
