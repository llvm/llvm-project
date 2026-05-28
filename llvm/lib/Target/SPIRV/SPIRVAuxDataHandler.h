//===-- SPIRVAuxDataHandler.h - NonSemantic.AuxData emitter -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits NonSemantic.AuxData ExtInst annotations preserving LLVM-level info
// with no native SPIR-V representation (currently: available_externally
// linkage). Matches SPIRV-LLVM-Translator's --spirv-preserve-auxdata.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVAUXDATAHANDLER_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVAUXDATAHANDLER_H

#include "SPIRVModuleAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCRegister.h"

namespace llvm {

class AsmPrinter;
class Function;
class Module;
class SPIRVSubtarget;

class SPIRVAuxDataHandler {
public:
  SPIRVAuxDataHandler(AsmPrinter &AP, const Module &M);

  bool hasWork() const { return !LinkagePreservedFns.empty(); }

  /// Register the extension and ext-inst-set in MAI. Must run before
  /// outputGlobalRequirements() / outputOpExtInstImports().
  void prepareModuleOutput(const SPIRVSubtarget &ST,
                           SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit AuxData annotations in module section 10.
  void emitAuxData(SPIRV::ModuleAnalysisInfo &MAI);

private:
  AsmPrinter &Asm;
  SmallVector<const Function *> LinkagePreservedFns;

  void emitMCInst(MCInst &Inst);
  MCRegister findOrEmitOpTypeVoid(SPIRV::ModuleAnalysisInfo &MAI);
  MCRegister findOrEmitOpTypeInt32(SPIRV::ModuleAnalysisInfo &MAI);
  MCRegister emitOpConstantI32(uint32_t Value, MCRegister I32TypeReg,
                               SPIRV::ModuleAnalysisInfo &MAI);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVAUXDATAHANDLER_H
