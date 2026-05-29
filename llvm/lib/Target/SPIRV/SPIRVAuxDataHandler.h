//===-- SPIRVAuxDataHandler.h - NonSemantic.AuxData emitter -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits NonSemantic.AuxData ExtInst annotations (mirrors SPIRV-LLVM-Translator
// --spirv-preserve-auxdata). Linkage records emit unconditionally for AE-tagged
// functions; attribute/metadata records are gated by -spirv-preserve-auxdata.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVAUXDATAHANDLER_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVAUXDATAHANDLER_H

#include "SPIRVModuleAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"

namespace llvm {

class AsmPrinter;
class Function;
class GlobalObject;
class Module;
class SPIRVSubtarget;

// Khronos NonSemantic.AuxData opcodes (int64_t to drop casts at MCOperand
// boundaries).
enum AuxDataOpcode : int64_t {
  FunctionMetadataOpcode = 0,
  FunctionAttributeOpcode = 1,
  GlobalVariableMetadataOpcode = 2,
  GlobalVariableAttributeOpcode = 3,
  LinkageOpcode = 4,
};

class SPIRVAuxDataHandler {
public:
  SPIRVAuxDataHandler(AsmPrinter &AP, const Module &M);

  bool hasWork() const;

  /// Register extension + ext-inst-set; call before output of section 1.
  void prepareModuleOutput(const SPIRVSubtarget &ST,
                           SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit OpStrings and stage ExtInst records; call in module section 7.
  void emitAuxDataStrings(SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit the staged ExtInst records; call in module section 10.
  void emitAuxData(SPIRV::ModuleAnalysisInfo &MAI);

private:
  struct ExtInstRecord {
    AuxDataOpcode Opcode;
    SmallVector<MCRegister, 4> Operands;
  };

  AsmPrinter &Asm;
  const Module &Mod;

  SmallVector<const Function *> LinkagePreservedFns;

  // Backing storage for non-string-attribute strings; StringRegs keys are
  // StringRefs into it.
  BumpPtrAllocator StringAlloc;
  UniqueStringSaver StringPool{StringAlloc};

  DenseMap<StringRef, MCRegister> StringRegs;
  SmallVector<ExtInstRecord> PendingRecords;

  MCRegister getOrEmitString(StringRef S, SPIRV::ModuleAnalysisInfo &MAI);
  void collectAttributesFor(const GlobalObject *GO,
                            function_ref<MCRegister()> GetNameReg,
                            SPIRV::ModuleAnalysisInfo &MAI);
  void collectMetadataFor(const GlobalObject *GO,
                          function_ref<MCRegister()> GetNameReg,
                          ArrayRef<StringRef> MDNames,
                          SPIRV::ModuleAnalysisInfo &MAI);

  void emitMCInst(MCInst &Inst);
  MCRegister findOrEmitOpTypeVoid(SPIRV::ModuleAnalysisInfo &MAI);
  MCRegister findOrEmitOpTypeInt32(SPIRV::ModuleAnalysisInfo &MAI);
  MCRegister emitOpConstantI32(uint32_t Value, MCRegister I32TypeReg,
                               SPIRV::ModuleAnalysisInfo &MAI);
  void emitAuxDataExtInst(AuxDataOpcode Opcode, MCRegister VoidTypeReg,
                          MCRegister ExtSetReg, ArrayRef<MCRegister> Operands,
                          SPIRV::ModuleAnalysisInfo &MAI);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVAUXDATAHANDLER_H
