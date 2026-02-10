//===- llvm/MCCAS/MCCASFormatSchemaBase.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCCAS_MCCASFORMATSCHEMABASE_H
#define LLVM_MCCAS_MCCASFORMATSCHEMABASE_H

#include "llvm/CAS/ObjectStore.h"
#include "llvm/MC/MCCASFormatSchemaBase.h"
#include "llvm/Support/Casting.h"

namespace llvm {
namespace mccasformats {

/// Creates all the schemas and can be used to retrieve a particular schema
/// based on a CAS root node. A client should aim to create and maximize re-use
/// of an instance of this object.
void addMCFormatSchemas(cas::SchemaPool &Pool);

/// Wrapper for a pool that is preloaded with object file schemas.
class MCFormatSchemaPool {
public:
  /// Creates all the schemas up front.
  explicit MCFormatSchemaPool(cas::ObjectStore &CAS) : Pool(CAS) {
    addMCFormatSchemas(Pool);
  }

  /// Look up the schema for the provided root node. Returns \a nullptr if no
  /// schema was found or it's not actually a root node. The returned \p
  /// ObjectFormatSchemaBase pointer is owned by the \p SchemaPool instance,
  /// therefore it cannot be used beyond the \p SchemaPool instance's lifetime.
  ///
  /// Thread-safe.
  MCFormatSchemaBase *getSchemaForRoot(cas::ObjectProxy Node) const {
    return dyn_cast_or_null<MCFormatSchemaBase>(Pool.getSchemaForRoot(Node));
  }

  cas::SchemaPool &getPool() { return Pool; }
  cas::ObjectStore &getCAS() const { return Pool.getCAS(); }

private:
  cas::SchemaPool Pool;
};

} // namespace mccasformats
} // namespace llvm

#endif // LLVM_MCCAS_MCCASFORMATSCHEMABASE_H
