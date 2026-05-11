//===-- SPIRVNonSemanticDebugHandler.h - NSDI AsmPrinter handler -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares SPIRVNonSemanticDebugHandler, a DebugHandlerBase subclass
// that emits NonSemantic.Shader.DebugInfo.100 instructions in the SPIR-V
// AsmPrinter. It replaces SPIRVEmitNonSemanticDI, which was a
// MachineFunctionPass, with a handler that controls instruction placement
// directly instead of routing through SPIRVModuleAnalysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVNONSEMANTICDEBUGHANDLER_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVNONSEMANTICDEBUGHANDLER_H

#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "SPIRVModuleAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/DebugHandlerBase.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegister.h"

namespace llvm {

class SPIRVSubtarget;

/// AsmPrinter handler that emits NonSemantic.Shader.DebugInfo.100 (NSDI)
/// instructions for the SPIR-V backend. Registered with SPIRVAsmPrinter when
/// the module contains debug info (llvm.dbg.cu).
///
/// Call sequence:
///   beginModule()                    -- collect compile-unit metadata.
///   prepareModuleOutput()            -- add extension + ext inst set to MAI.
///   emitNonSemanticGlobalDebugInfo() -- emit DebugSource,
///                                       DebugCompilationUnit, DebugTypeBasic,
///                                       DebugTypePointer.
///   beginFunctionImpl()              -- no-op (no per-function DI yet).
///   endFunctionImpl()                -- no-op.
class SPIRVNonSemanticDebugHandler : public DebugHandlerBase {
  struct CompileUnitInfo {
    SmallString<128> FilePath;
    unsigned SpirvSourceLanguage = 0; // NonSemantic.Shader.DebugInfo.100 source
                                      // language code (section 4.3)
  };
  // TODO: When per-function NSDI emission is implemented, augment
  // CompileUnitInfo with the originating DICompileUnit pointer so that
  // Parent operands on DebugFunction and similar instructions can resolve
  // the compile unit's result register.
  SmallVector<CompileUnitInfo> CompileUnits;
  int64_t DwarfVersion = 0;

  // Types referenced by debug variable records, collected in beginModule().
  SetVector<const DIBasicType *> BasicTypes;
  SetVector<const DIDerivedType *> PointerTypes;

  // Cache of already-emitted i32 constants, keyed by value. Prevents
  // duplicate OpConstant instructions for the same integer value.
  DenseMap<uint32_t, MCRegister> I32ConstantCache;

  // OpString registers for NSDI instructions, populated by
  // emitNonSemanticDebugStrings() (section 7) and consumed by
  // emitNonSemanticGlobalDebugInfo() (section 10). OpString must appear in
  // section 7 per the SPIR-V module layout; it cannot be emitted alongside the
  // OpExtInst instructions in section 10.
  SmallVector<MCRegister> FileStringRegs;    // one per CompileUnits entry
  SmallVector<MCRegister> BasicTypeNameRegs; // one per BasicTypes entry

  // True once emitNonSemanticGlobalDebugInfo() has run. Both
  // SPIRVAsmPrinter::emitFunctionHeader() and emitEndOfAsmFile() may call
  // outputModuleSections(), each guarded by ModuleSectionsEmitted, so only
  // one fires. This flag provides a secondary guard in case the call sites
  // change.
  bool GlobalDIEmitted = false;

public:
  explicit SPIRVNonSemanticDebugHandler(AsmPrinter &AP);

  /// Collect compile-unit metadata from the module. Called by
  /// AsmPrinter::doInitialization() via the handler list. No emission.
  void beginModule(Module *M) override;

  /// Emit OpString instructions for all NSDI file paths and basic type names
  /// into the debug section (section 7 of the SPIR-V module layout). Must be
  /// called from SPIRVAsmPrinter::outputDebugSourceAndStrings(), after
  /// prepareModuleOutput() has registered the ext inst set. The resulting
  /// registers are cached in FileStringRegs and BasicTypeNameRegs for use by
  /// emitNonSemanticGlobalDebugInfo().
  void emitNonSemanticDebugStrings(SPIRV::ModuleAnalysisInfo &MAI);

  /// Add SPV_KHR_non_semantic_info extension and
  /// NonSemantic.Shader.DebugInfo.100 ext inst set entry to MAI. Must be called
  /// before outputGlobalRequirements() and outputOpExtInstImports() in
  /// SPIRVAsmPrinter::outputModuleSections().
  void prepareModuleOutput(const SPIRVSubtarget &ST,
                           SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit module-scope NSDI instructions (DebugSource, DebugCompilationUnit,
  /// DebugTypeBasic, DebugTypePointer). Called by SPIRVAsmPrinter::
  /// outputModuleSections() at section 10 in place of
  /// outputModuleSection(MB_NonSemanticGlobalDI).
  void emitNonSemanticGlobalDebugInfo(SPIRV::ModuleAnalysisInfo &MAI);

protected:
  // All module-level output is driven by emitNonSemanticGlobalDebugInfo(),
  // called explicitly from SPIRVAsmPrinter::outputModuleSections(). Nothing
  // needs to happen in the AsmPrinterHandler::endModule() callback.
  void endModule() override {}

  // DebugHandlerBase stores MMI as a pointer copy from Asm->MMI at construction
  // time (DebugHandlerBase.cpp: `MMI(Asm->MMI)`). The handler is constructed
  // before AsmPrinter::doInitialization() runs, so Asm->MMI is null at that
  // point and MMI remains null for this handler's entire lifetime. The
  // base-class beginInstruction/endInstruction dereference MMI to create temp
  // symbols for label tracking and would crash. Override them as no-ops.
  // When per-function NSDI is implemented, use Asm->OutStreamer->getContext()
  // for MCContext access rather than MMI->getContext().
  void beginInstruction(const MachineInstr *MI) override {}
  void endInstruction() override {}

  // TODO: Emit DebugFunction and DebugFunctionDefinition here once per-function
  // NSDI emission is implemented. DebugHandlerBase::beginFunction() populates
  // LScopes and DbgValues, which are needed for DebugLine emission. Do not
  // override beginFunction() until that work is in place.
  void beginFunctionImpl(const MachineFunction *MF) override {}
  // TODO: Add per-function cleanup when DebugFunction emission is in place.
  void endFunctionImpl(const MachineFunction *MF) override {}

private:
  void emitMCInst(MCInst &Inst);
  MCRegister emitOpString(StringRef S, SPIRV::ModuleAnalysisInfo &MAI);
  MCRegister emitOpConstantI32(uint32_t Value, MCRegister I32TypeReg,
                               SPIRV::ModuleAnalysisInfo &MAI);
  MCRegister emitExtInst(SPIRV::NonSemanticExtInst::NonSemanticExtInst Opcode,
                         MCRegister VoidTypeReg, MCRegister ExtInstSetReg,
                         ArrayRef<MCRegister> Operands,
                         SPIRV::ModuleAnalysisInfo &MAI);

  /// Find OpTypeVoid in the already-emitted TypeConstVars section, or emit one
  /// if the module does not contain it (e.g. no void-returning functions).
  MCRegister findOrEmitOpTypeVoid(SPIRV::ModuleAnalysisInfo &MAI);

  /// Find OpTypeInt 32 0 in the already-emitted TypeConstVars section, or emit
  /// one if the module does not contain it.
  MCRegister findOrEmitOpTypeInt32(SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit a DebugTypePointer instruction for PT. Skips pointer types that do
  /// not carry a DWARF address space. For pointers whose base type is a
  /// DIBasicType, looks up the base type's DebugTypeBasic register in
  /// BasicTypeRegs. All other pointers (void pointers and pointers whose base
  /// type is not a DIBasicType) use DebugInfoNone as the base type operand.
  void emitDebugTypePointer(
      const DIDerivedType *PT, MCRegister VoidTypeReg, MCRegister I32TypeReg,
      MCRegister ExtInstSetReg, MCRegister I32ZeroReg,
      const DenseMap<const DIBasicType *, MCRegister> &BasicTypeRegs,
      SPIRV::ModuleAnalysisInfo &MAI);

  /// Map a DWARF source language code to a NonSemantic.Shader.DebugInfo.100
  /// source language code.
  static unsigned toNSDISrcLang(unsigned DwarfSrcLang);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVNONSEMANTICDEBUGHANDLER_H
