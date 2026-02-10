//===- llvm/MC/MCCASFormatSchemaBase.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCCASFORMATSCHEMABASE_H
#define LLVM_MC_MCCASFORMATSCHEMABASE_H

#include "llvm/CAS/CASNodeSchema.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MachOCASWriter;
class MCAssembler;

namespace mccasformats {

class MCFormatSchemaBase
    : public RTTIExtends<MCFormatSchemaBase, cas::NodeSchema> {
  void anchor() override;

public:
  static char ID;

  Expected<cas::ObjectProxy>
  createFromMCAssembler(llvm::MachOCASWriter &ObjectWriter,
                        llvm::MCAssembler &Asm,
                        raw_ostream *DebugOS = nullptr) const {
    return createFromMCAssemblerImpl(ObjectWriter, Asm, DebugOS);
  }

  virtual Error serializeObjectFile(cas::ObjectProxy RootNode,
                                    llvm::raw_ostream &OS) const = 0;

protected:
  virtual Expected<cas::ObjectProxy>
  createFromMCAssemblerImpl(llvm::MachOCASWriter &ObjectWriter,
                            llvm::MCAssembler &Asm,
                            raw_ostream *DebugOS) const = 0;

  MCFormatSchemaBase(cas::ObjectStore &CAS)
      : MCFormatSchemaBase::RTTIExtends(CAS) {}
};

using MCCASSchema = MCFormatSchemaBase;
} // namespace mccasformats
} // namespace llvm

#endif // LLVM_MC_MCCASFORMATSCHEMABASE_H
