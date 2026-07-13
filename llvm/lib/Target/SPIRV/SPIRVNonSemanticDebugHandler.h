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
///   emitNonSemanticGlobalDebugInfo() -- emit DebugSource,
///                                       DebugCompilationUnit, DebugTypeBasic,
///                                       DebugTypePointer, DebugTypeFunction,
///                                       DebugFunctionDeclaration.
///   beginFunctionImpl()              -- no-op (no per-function DI yet).
///   endFunctionImpl()                -- no-op.
class SPIRVNonSemanticDebugHandler : public DebugHandlerBase {
  struct CompileUnitInfo {
    const DICompileUnit *TheCU = nullptr;
    SmallString<128> FilePath;
    unsigned SpirvSourceLanguage = 0; // NonSemantic.Shader.DebugInfo.100 source
                                      // language code (section 4.3)
  };
  SmallVector<CompileUnitInfo> CompileUnits;
  int64_t DwarfVersion = 0;

  // DI types partitioned from DebugInfoFinder.types() in beginModule()
  // (basics, pointers, vectors, subroutine types NSDI v1 may emit).
  SmallVector<const DIBasicType *> BasicTypes;
  SmallVector<const DIDerivedType *> PointerTypes;
  SmallVector<const DISubroutineType *> SubroutineTypes;
  // DICompositeType nodes with DW_TAG_array_type and DINode::FlagVector,
  // partitioned from DebugInfoFinder.types() in beginModule().
  SmallVector<const DICompositeType *> VectorTypes;

  // Filled in emitNonSemanticGlobalDebugInfo(): DI types to their result
  // registers.
  DenseMap<const DIType *, MCRegister> DebugTypeRegs;

  // DISubprogram nodes that are declarations only (!isDefinition()), collected
  // in beginModule() for DebugFunctionDeclaration emission.
  SmallVector<const DISubprogram *> SubprogramDeclarations;

  // DebugFunctionDeclaration result id per emitted declaration DISubprogram
  // (only entries where emission succeeded).
  DenseMap<const DISubprogram *, MCRegister> DebugFunctionDeclarationRegs;

  // Path \c OpString result id per \c DIScope (CU, \c DIFile, declaration
  // \c DISubprogram, …). Filled during \c emitNonSemanticDebugStrings() using
  // \c getDebugFullPath + \c emitOpStringIfNew; section 10 uses it for
  // \c DebugSource without recomputing path text.
  DenseMap<const DIScope *, MCRegister> ScopeToPathOpStringReg;

  // DebugCompilationUnit result id per DICompileUnit (for Parent operands).
  DenseMap<const DICompileUnit *, MCRegister> CUToCompilationUnitDbgReg;

  // DebugSource result id keyed by path \c OpString id (\c MCRegister::id()),
  // deduplicating when the same file string is reused.
  DenseMap<unsigned, MCRegister> DebugSourceRegByFileStr;

  // Maps OpString contents to result id. Populated only by emitOpStringIfNew()
  // during section 7; section 10 uses getCachedOpStringReg() (lookup only).
  StringMap<MCRegister> OpStringContentCache;

#ifndef NDEBUG // Only declare the variable for debugging purposes.
  // True after emitNonSemanticDebugStrings() emitted the NSDI OpStrings for
  // this module. SPIRVAsmPrinter calls that before
  // emitNonSemanticGlobalDebugInfo().
  bool NonSemanticOpStringsSectionEmitted = false;
#endif

  MCRegister CachedDebugInfoNoneReg;

  MCRegister CachedOpTypeVoidReg;

  MCRegister CachedOpTypeInt32Reg;

  // Cache of already-emitted i32 constants, keyed by value. Prevents
  // duplicate OpConstant instructions for the same integer value.
  DenseMap<uint32_t, MCRegister> I32ConstantCache;

  // Cache of already-emitted DebugTypeFunction instructions, keyed by operand
  // ids (flags, return type, parameters).
  DenseMap<SmallVector<MCRegister, 8>, MCRegister> DebugTypeFunctionCache;

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
  /// stored in \c OpStringContentCache and \c ScopeToPathOpStringReg;
  /// \c emitNonSemanticGlobalDebugInfo() resolves them via
  /// \c getCachedOpStringReg() and path maps.
  void emitNonSemanticDebugStrings(SPIRV::ModuleAnalysisInfo &MAI);

  /// Add SPV_KHR_non_semantic_info extension and
  /// NonSemantic.Shader.DebugInfo.100 ext inst set entry to MAI. Must be called
  /// before outputGlobalRequirements() and outputOpExtInstImports() in
  /// SPIRVAsmPrinter::outputModuleSections().
  void prepareModuleOutput(const SPIRVSubtarget &ST,
                           SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit module-scope NSDI instructions (DebugSource, DebugCompilationUnit,
  /// DebugTypeBasic, DebugTypePointer, DebugTypeFunction,
  /// DebugFunctionDeclaration). Called by
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

  /// Section 7 only: emit OpString and cache it if not already present. Must
  /// not be called after NonSemanticOpStringsSectionEmitted is set. Returns
  /// the path (or string) \c OpString result id.
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

  /// Return a cached DebugTypeFunction id when \p Ops matches a prior emission,
  /// otherwise emit and cache a new instruction.
  MCRegister getOrEmitDebugTypeFunction(ArrayRef<MCRegister> Ops,
                                        MCRegister VoidTypeReg,
                                        MCRegister ExtInstSetReg,
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

  /// Emit \c DebugFunctionDeclaration for a \c DISubprogram that is not a
  /// definition (\p SP must satisfy \c !isDefinition()).
  ///
  /// \returns The result id register on success. Returns \c std::nullopt and
  /// emits nothing if \p SP is null, is a definition, has no \c
  /// DISubroutineType type, the signature type was not emitted in \c
  /// DebugTypeRegs, no path
  /// \c OpString was recorded for \p SP in section 7, or
  /// \c resolveDebugFunctionDeclarationParent returns no id for the \c Parent
  /// operand.
  std::optional<MCRegister>
  emitDebugFunctionDeclaration(const DISubprogram *SP, MCRegister VoidTypeReg,
                               MCRegister I32TypeReg, MCRegister ExtInstSetReg,
                               SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit \c DebugTypeVector for the vector composite type \p VT.
  ///
  /// \returns The result id register on success. Returns \c std::nullopt and
  /// emits nothing if \p VT has no \c DIBasicType base type, if the base type
  /// has not been emitted yet, if \p VT has more than one \c DISubrange
  /// element, or if the component count is not a compile-time constant.
  std::optional<MCRegister> emitDebugTypeVector(const DICompositeType *VT,
                                                MCRegister ExtInstSetReg,
                                                SPIRV::ModuleAnalysisInfo &MAI);

  /// Map a \c DISubroutineType::getTypeArray() element to an operand register
  /// for
  /// \c DebugTypeFunction. Non-null \p Ty resolves via \c DebugTypeRegs; if the
  /// type was never emitted, returns \c std::nullopt.
  ///
  /// LLVM encodes a void return as a null first element (and may use null in
  /// later slots). NonSemantic \c DebugTypeFunction
  /// requires a concrete return-type operand, so when \p ReturnType is true and
  /// \p Ty is null, this returns \p VoidTypeReg (\c OpTypeVoid). When
  /// \p ReturnType is false and \p Ty is null, this returns
  /// \c CachedDebugInfoNoneReg (\c DebugInfoNone).
  std::optional<MCRegister> mapDISignatureTypeToReg(const DIType *Ty,
                                                    MCRegister VoidTypeReg,
                                                    bool ReturnType);

  /// Map a DWARF source language code to a NonSemantic.Shader.DebugInfo.100
  /// source language code.
  static unsigned toNSDISrcLang(unsigned DwarfSrcLang);

  /// Build a full path from debug \p Scope for OpString / DebugSource, matching
  /// SPIRV-LLVM-Translator \c getFullPath (OCLUtil.h): \c DIScope::getFilename,
  /// \c getDirectory, and \c sys::path::Style::native. Works for any \c DIScope
  /// that carries file path fields (e.g. \c DIFile, \c DISubprogram,
  /// \c DICompileUnit). Returns an empty path when \p Scope is null.
  SmallString<128> getDebugFullPath(const DIScope *Scope) const;

  /// Return an existing \c DebugSource id for file path \c OpString \p
  /// FileStrReg or emit \c DebugSource and cache it (keyed by \p FileStrReg
  /// id).
  MCRegister getOrEmitDebugSourceForFileStrReg(MCRegister FileStrReg,
                                               MCRegister VoidTypeReg,
                                               MCRegister ExtInstSetReg,
                                               SPIRV::ModuleAnalysisInfo &MAI);

  /// Resolve the \c Parent operand for \c DebugFunctionDeclaration: an emitted
  /// debug type id when \c SP->getScope() is a \c DIType in \c DebugTypeRegs,
  /// otherwise \c DebugCompilationUnit for \c SP->getUnit() (or the first
  /// module CU when \c unit: is absent).
  /// \returns \c std::nullopt when the scope requires a parent we cannot supply
  /// (non-file scope that is not a mapped \c DIType) or the CU has no emitted
  /// id.
  std::optional<MCRegister>
  resolveDebugFunctionDeclarationParent(const DISubprogram *SP) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVNONSEMANTICDEBUGHANDLER_H
