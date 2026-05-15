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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/DebugHandlerBase.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegister.h"
#include <optional>

namespace llvm {

class SPIRVSubtarget;

/// AsmPrinter handler that emits NonSemantic.Shader.DebugInfo.100 (NSDI)
/// instructions for the SPIR-V backend. Registered with SPIRVAsmPrinter when
/// the module contains debug info (llvm.dbg.cu).
///
/// Call sequence:
///   beginModule()                    -- collect compile-unit metadata.
///   prepareModuleOutput()            -- add extension + ext inst set to MAI.
///   emitNonSemanticDebugStrings()    -- OpString for NSDI strings (sec. 7).
///   emitNonSemanticGlobalDebugInfo() -- emit DebugSource, DebugTypeBasic,
///                                       DebugTypePointer, DebugTypeFunction,
///                                       DebugCompilationUnit.
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

  // DI types partitioned from DebugInfoFinder.types() in beginModule()
  // (basics, pointers, subroutine types NSDI v1 may emit).
  SmallVector<const DIBasicType *> BasicTypes;
  SmallVector<const DIDerivedType *> PointerTypes;
  SmallVector<const DISubroutineType *> SubroutineTypes;

  // Filled in emitNonSemanticGlobalDebugInfo(): DI types to their result
  // registers.
  DenseMap<const DIType *, MCRegister> DebugTypeRegs;

  // Maps OpString contents to result id. Populated only by emitOpStringIfNew()
  // during section 7; section 10 uses getCachedOpStringReg() (lookup only).
  StringMap<MCRegister> OpStringContentCache;

  // True after emitNonSemanticDebugStrings() emitted the NSDI OpStrings for
  // this module. SPIRVAsmPrinter calls that before
  // emitNonSemanticGlobalDebugInfo().
  bool NonSemanticOpStringsSectionEmitted = false;

  MCRegister CachedDebugInfoNoneReg;

  MCRegister CachedOpTypeVoidReg;

  MCRegister CachedOpTypeInt32Reg;

  // Cache of already-emitted i32 constants, keyed by value. Prevents
  // duplicate OpConstant instructions for the same integer value.
  DenseMap<uint32_t, MCRegister> I32ConstantCache;

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
  /// prepareModuleOutput() has registered the ext inst set. Registers are
  /// stored in OpStringContentCache; emitNonSemanticGlobalDebugInfo() resolves
  /// them via getCachedOpStringReg().
  void emitNonSemanticDebugStrings(SPIRV::ModuleAnalysisInfo &MAI);

  /// Add SPV_KHR_non_semantic_info extension and
  /// NonSemantic.Shader.DebugInfo.100 ext inst set entry to MAI. Must be called
  /// before outputGlobalRequirements() and outputOpExtInstImports() in
  /// SPIRVAsmPrinter::outputModuleSections().
  void prepareModuleOutput(const SPIRVSubtarget &ST,
                           SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit module-scope NSDI instructions (DebugSource, DebugCompilationUnit,
  /// DebugTypeBasic, DebugTypePointer, DebugTypeFunction). Called by
  /// SPIRVAsmPrinter::outputModuleSections() at section 10 in place of
  /// outputModuleSection(MB_NonSemanticGlobalDI). Requires
  /// emitNonSemanticDebugStrings() to have run first when NSDI strings apply.
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

  /// Section 7 only: return cached id or emit OpString and cache it. Must not
  /// be called after NonSemanticOpStringsSectionEmitted is set.
  MCRegister emitOpStringIfNew(StringRef S, SPIRV::ModuleAnalysisInfo &MAI);

  /// Section 10 only: lookup OpString id from cache; asserts if missing or if
  /// section 7 did not complete.
  MCRegister getCachedOpStringReg(StringRef S);
  MCRegister emitOpConstantI32(uint32_t Value, MCRegister I32TypeReg,
                               SPIRV::ModuleAnalysisInfo &MAI);
  MCRegister emitExtInst(SPIRV::NonSemanticExtInst::NonSemanticExtInst Opcode,
                         MCRegister VoidTypeReg, MCRegister ExtInstSetReg,
                         ArrayRef<MCRegister> Operands,
                         SPIRV::ModuleAnalysisInfo &MAI);

  /// Return OpTypeVoid id for this module (lazy lookup / emit, then cache).
  MCRegister getOrEmitOpTypeVoidReg(SPIRV::ModuleAnalysisInfo &MAI);

  /// Return OpTypeInt 32 0 id for this module (lazy lookup / emit, then cache).
  MCRegister getOrEmitOpTypeInt32Reg(SPIRV::ModuleAnalysisInfo &MAI);

  /// Find OpTypeVoid in the already-emitted TypeConstVars section, or emit one
  /// if the module does not contain it (e.g. no void-returning functions).
  MCRegister findOrEmitOpTypeVoid(SPIRV::ModuleAnalysisInfo &MAI);

  /// Find OpTypeInt 32 0 in the already-emitted TypeConstVars section, or emit
  /// one if the module does not contain it.
  MCRegister findOrEmitOpTypeInt32(SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit \c DebugTypePointer for pointer metadata \p PT.
  ///
  /// \returns The result id register on success. Returns \c std::nullopt and
  /// emits nothing if \p PT has no DWARF address space (needed to pick the
  /// SPIR-V storage class), or if \p PT has a non-null base DI type that is not
  /// yet in \c DebugTypeRegs (the pointee was not emitted as a debug type).
  ///
  /// Base Type operand: the register from \c DebugTypeRegs for \p PT's base
  /// type when it is set and mapped; \c DebugInfoNone when there is no base
  /// type (e.g. \c void * in IR), consistent with SPIRV-LLVM-Translator.
  std::optional<MCRegister>
  emitDebugTypePointer(const DIDerivedType *PT, MCRegister ExtInstSetReg,
                       SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit one DebugTypeFunction for ST when every DI operand maps to a debug
  /// type id; otherwise emit nothing and return std::nullopt.
  std::optional<MCRegister>
  emitDebugTypeFunctionForSubroutineType(const DISubroutineType *ST,
                                         MCRegister ExtInstSetReg,
                                         SPIRV::ModuleAnalysisInfo &MAI);

  /// Map DISubroutineType slot DI types using DebugTypeRegs.
  std::optional<MCRegister> mapDISignatureTypeToReg(const DIType *Ty,
                                                    MCRegister VoidTypeReg);

  /// Map a DWARF source language code to a NonSemantic.Shader.DebugInfo.100
  /// source language code.
  static unsigned toNSDISrcLang(unsigned DwarfSrcLang);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVNONSEMANTICDEBUGHANDLER_H
