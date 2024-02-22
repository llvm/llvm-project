//===- llvm/MC/CAS/MCCASFormatSchemaBase.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_CAS_MCCASFORMATSCHEMABASE_H
#define LLVM_MC_CAS_MCCASFORMATSCHEMABASE_H

#include "llvm/CAS/CASNodeSchema.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCMachOCASWriter.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace mccasformats {

class MCFormatSchemaBase
    : public RTTIExtends<MCFormatSchemaBase, cas::NodeSchema> {
  void anchor() override;

public:
  static char ID;

  Expected<cas::ObjectProxy>
  createFromMCAssembler(llvm::MachOCASWriter &ObjectWriter,
                        llvm::MCAssembler &Asm, const llvm::MCAsmLayout &Layout,
                        raw_ostream *DebugOS = nullptr) const {
    return createFromMCAssemblerImpl(ObjectWriter, Asm, Layout, DebugOS);
  }

  virtual Error serializeObjectFile(cas::ObjectProxy RootNode,
                                    llvm::raw_ostream &OS) const = 0;

protected:
  virtual Expected<cas::ObjectProxy> createFromMCAssemblerImpl(
      llvm::MachOCASWriter &ObjectWriter, llvm::MCAssembler &Asm,
      const llvm::MCAsmLayout &Layout, raw_ostream *DebugOS) const = 0;

  MCFormatSchemaBase(cas::ObjectStore &CAS)
      : MCFormatSchemaBase::RTTIExtends(CAS) {}
};

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
    return dyn_cast_or_null<MCFormatSchemaBase>(
        Pool.getSchemaForRoot(Node));
  }

  cas::SchemaPool &getPool() { return Pool; }
  cas::ObjectStore &getCAS() const { return Pool.getCAS(); }

private:
  cas::SchemaPool Pool;
};

} // namespace mccasformats
} // namespace llvm

#endif // LLVM_MC_CAS_MCCASFORMATSCHEMABASE_H
