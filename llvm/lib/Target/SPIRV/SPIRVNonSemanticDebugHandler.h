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
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/DebugHandlerBase.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegister.h"
#include <optional>

namespace llvm {

class GlobalVariable;
class SPIRVSubtarget;

/// AsmPrinter handler that emits NonSemantic.Shader.DebugInfo.100 (NSDI)
/// instructions for the SPIR-V backend. Registered with SPIRVAsmPrinter when
/// the module contains debug info (llvm.dbg.cu).
///
/// Call sequence:
/// - beginModule() collects compile-unit metadata.
/// - prepareModuleOutput() adds the extension and ext-inst set to MAI.
/// - emitNonSemanticDebugStrings() emits NSDI OpStrings in section 7.
/// - emitNonSemanticGlobalDebugInfo() emits module-scope NSDI and sets
///   GlobalNSDIEnabled.
/// - beginFunctionImpl() prepares per-function DebugFunctionDefinition state.
/// - endInstruction() emits DebugFunctionDefinition after the last function-
///   level OpVariable; SPIRVAsmPrinter calls notifyEntryLabelEmitted() after
///   the synthesized entry OpLabel when there are no OpVariables.
/// - endFunctionImpl() resets per-function state.
class SPIRVNonSemanticDebugHandler : public DebugHandlerBase {
  static constexpr unsigned NSSet = static_cast<unsigned>(
      SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100);

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
  // DICompositeType nodes with DW_TAG_array_type that are not vectors,
  // partitioned in beginModule().
  SmallVector<const DICompositeType *> ArrayTypes;
  // DICompositeType nodes with DW_TAG_structure_type, DW_TAG_class_type, or
  // DW_TAG_union_type, partitioned in beginModule() for DebugTypeComposite.
  SmallVector<const DICompositeType *> CompositeTypes;
  // DIDerivedType nodes with DW_TAG_typedef, partitioned in beginModule() for
  // DebugTypedef emission.
  SmallVector<const DIDerivedType *> TypedefTypes;

  // NonSemantic debug instruction result id per emitted scope.
  DenseMap<const DIScope *, MCRegister> DebugScopeRegs;

  // DISubprogram nodes that are declarations only (!isDefinition()), collected
  // in beginModule() for DebugFunctionDeclaration emission.
  SmallVector<const DISubprogram *> SubprogramDeclarations;

  // DISubprogram nodes that are definitions, collected in beginModule() for
  // DebugFunction emission.
  SmallVector<const DISubprogram *> SubprogramDefinitions;

  // Distinct DILocations from instruction !dbg attachments and debug program
  // records (#dbg_declare, #dbg_value, #dbg_assign, #dbg_label).
  SetVector<const DILocation *> UniqueDebugLocations;

  struct GlobalVariableDebugInfo {
    const DIExpression *Expr = nullptr;
    const GlobalVariable *LLVMGV = nullptr;
  };
  DenseMap<const DIGlobalVariable *, GlobalVariableDebugInfo>
      GlobalVariableDebugInfoMap;

  // Distinct DILexicalBlock and DINamespace scopes, parent-before-child
  // order, collected in beginModule() for DebugLexicalBlock emission.
  SetVector<const DIScope *> LexicalBlocks;

  // Path \c OpString result id per \c DIScope (CU, \c DIFile, declaration
  // \c DISubprogram, …). Filled during \c emitNonSemanticDebugStrings() using
  // \c getDebugFullPath + \c emitOpStringIfNew; section 10 uses it for
  // \c DebugSource without recomputing path text.
  DenseMap<const DIScope *, MCRegister> ScopeToPathOpStringReg;

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

  MCRegister CachedEmptyStringReg;

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

  // True when emitNonSemanticGlobalDebugInfo() completed module-scope NSDI
  // emission for this module.
  bool GlobalNSDIEnabled = false;

  SPIRV::ModuleAnalysisInfo *CurrentMAI = nullptr;

  const MachineFunction *CurrentMF = nullptr;

  const MachineInstr *LastFunctionOpVariable = nullptr;

  bool DebugFunctionDefinitionEmitted = false;

  const MachineInstr *LastLineMI = nullptr;

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
  /// DebugFunctionDeclaration, DebugFunction). Called by
  /// SPIRVAsmPrinter::outputModuleSections() at section 10 in place of
  /// outputModuleSection(MB_NonSemanticGlobalDI). Requires
  /// emitNonSemanticDebugStrings() to have run first when NSDI strings apply.
  /// Sets \c GlobalNSDIEnabled when module-scope NSDI emission completes.
  void emitNonSemanticGlobalDebugInfo(SPIRV::ModuleAnalysisInfo &MAI);

  /// Called after the synthesized entry \c OpLabel has been emitted.
  void notifyEntryLabelEmitted(const MachineFunction &MF);

protected:
  // All module-level output is driven by emitNonSemanticGlobalDebugInfo(),
  // called explicitly from SPIRVAsmPrinter::outputModuleSections(). Nothing
  // needs to happen in the AsmPrinterHandler::endModule() callback.
  void endModule() override {}

  // DebugHandlerBase stores MMI as a pointer copy from Asm->MMI at construction
  // time (DebugHandlerBase.cpp: `MMI(Asm->MMI)`). The handler is constructed
  // before AsmPrinter::doInitialization() runs, so Asm->MMI is null at that
  // point and MMI remains null for this handler's entire lifetime. Do not call
  // the base-class beginInstruction/endInstruction — they dereference MMI to
  // create temp symbols for label tracking and would crash.
  // Future local NSDI that needs MCContext must use
  // Asm->OutStreamer->getContext() rather than MMI->getContext().
  void beginInstruction(const MachineInstr *MI) override;
  void endInstruction() override;

  // Override beginFunctionImpl(), not beginFunction():
  // DebugHandlerBase::beginFunction() populates LScopes and DbgValues needed
  // for future DebugLine emission.
  void beginFunctionImpl(const MachineFunction *MF) override;
  void endFunctionImpl(const MachineFunction *MF) override;

private:
  void emitDebugFunctionDefinition(MCRegister DebugFunctionReg,
                                   MCRegister OpFunctionReg,
                                   SPIRV::ModuleAnalysisInfo &MAI);

  void resetPerFunctionDebugState();

  void emitDebugLineForInstruction(const MachineInstr *MI);
  void preparePerFunctionDebug(const MachineFunction *MF);
  void tryEmitDebugFunctionDefinition(SPIRV::ModuleAnalysisInfo &MAI);

  void emitMCInst(MCInst &Inst);
  MCRegister emitOpString(StringRef S, SPIRV::ModuleAnalysisInfo &MAI);

  /// Section 7 only: emit OpString and cache it if not already present. Must
  /// not be called after NonSemanticOpStringsSectionEmitted is set. Returns
  /// the path (or string) \c OpString result id.
  MCRegister emitOpStringIfNew(StringRef S, SPIRV::ModuleAnalysisInfo &MAI);

  /// Section 10 only: lookup OpString id from cache; asserts if missing or if
  /// section 7 did not complete.
  MCRegister getCachedOpStringReg(StringRef S);

  /// Section 7 only: emit the path \c OpString for \p Scope and cache it under
  /// \p Scope. Returns the \c OpString result id. A \p Scope already seen
  /// returns the cached id without rebuilding the path. A null \p Scope maps to
  /// the empty path and is cached like any other, though section 10 reads it
  /// through \c getCachedScopePathOpStringReg, which handles null separately.
  MCRegister emitAndCacheScopePathOpStringReg(const DIScope *Scope,
                                              SPIRV::ModuleAnalysisInfo &MAI);

  /// Section 10 only: lookup path \c OpString id for \p Scope from
  /// \c ScopeToPathOpStringReg; asserts if missing or invalid. When
  /// \p UseEmptyPathIfNullScope is true and \p Scope is null, returns
  /// \c CachedEmptyStringReg instead.
  MCRegister
  getCachedScopePathOpStringReg(const DIScope *Scope,
                                bool UseEmptyPathIfNullScope = false);
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
  /// yet in \c DebugScopeRegs (the pointee was not emitted as a debug type).
  ///
  /// Base Type operand: the register from \c DebugScopeRegs for \p PT's base
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
  /// DebugScopeRegs, no path
  /// \c OpString was recorded for \p SP in section 7, or
  /// \c resolveScope returns no id for the \c Parent operand.
  std::optional<MCRegister>
  emitDebugFunctionDeclaration(const DISubprogram *SP, MCRegister VoidTypeReg,
                               MCRegister I32TypeReg, MCRegister ExtInstSetReg,
                               SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit \c DebugFunction for a defining \c DISubprogram (\p SP must satisfy
  /// \c isDefinition()).
  std::optional<MCRegister> emitDebugFunction(const DISubprogram *SP,
                                              MCRegister VoidTypeReg,
                                              MCRegister I32TypeReg,
                                              MCRegister ExtInstSetReg,
                                              SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit \c DebugGlobalVariable for the source global variable \p GV.
  ///
  /// (\c SPIRVDebug::Operand::GlobalVariable): Name, Type, Source, Line,
  /// Column, Parent, Linkage Name, Variable, Flags, and an optional Static
  /// Member Declaration. Line, Column, and Flags are emitted as \c OpConstant
  /// ids as required for non-semantic debug info.
  ///
  /// \c DebugInfoNone is used for two operands when LLVM has no value to
  /// supply:
  /// \c Type when \p GV is a declaration with no DI type (e.g. \c extern void;
  /// valid IR, \c isDefinition: false); \c Variable when no \c
  /// llvm::GlobalVariable in this module carries \p GV in its \c !dbg metadata.
  ///
  /// \returns The result id register on success. Returns \c std::nullopt and
  /// emits nothing if a non-null \p GV type was not emitted in \c
  /// DebugScopeRegs, or \p GV has a static data member declaration that was not
  /// emitted in \c DebugScopeRegs.
  std::optional<MCRegister> emitDebugGlobalVariable(
      const DIGlobalVariable *GV, const GlobalVariableDebugInfo &Info,
      MCRegister VoidTypeReg, MCRegister I32TypeReg, MCRegister ExtInstSetReg,
      SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit \c DebugExpression for \p Expr. Unimplemented: defined as a no-op
  /// (\returns \c std::nullopt, emits nothing) so \c emitDebugGlobalVariable
  /// can complete Variable-operand resolution for the opcodes we support today.
  std::optional<MCRegister> emitDebugExpression(const DIExpression *Expr,
                                                MCRegister VoidTypeReg,
                                                MCRegister ExtInstSetReg,
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

  /// Emit \c DebugTypeArray for the array composite type \p AT.
  ///
  /// Emits the element (base) type id followed by one Component Count per
  /// \c DISubrange, in DWARF subrange order. A count that is not a
  /// compile-time constant is emitted as 0, matching \c OpTypeRuntimeArray. A
  /// matrix arrives here as a multi-subrange array and is emitted with one
  /// count per dimension.
  ///
  /// \returns The result id register on success. Returns \c std::nullopt and
  /// emits nothing if \p AT's element type has not been emitted into
  /// \c DebugScopeRegs.
  std::optional<MCRegister> emitDebugTypeArray(const DICompositeType *AT,
                                               MCRegister ExtInstSetReg,
                                               SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit \c DebugTypeMember for the data member \p M (a \c DIDerivedType with
  /// \c DW_TAG_member). Operands: Name, Type, Source, Line, Column, Offset,
  /// Size, Flags. NonSemantic \c DebugTypeMember carries no Parent operand: the
  /// enclosing \c DebugTypeComposite references its members, not the reverse.
  ///
  /// \returns The result id register on success. Returns \c std::nullopt and
  /// emits nothing if \p M's type has not been emitted into \c DebugScopeRegs.
  std::optional<MCRegister> emitDebugTypeMember(const DIDerivedType *M,
                                                MCRegister VoidTypeReg,
                                                MCRegister I32TypeReg,
                                                MCRegister ExtInstSetReg,
                                                SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit \c DebugTypeComposite for the struct, class, or union \p CT, listing
  /// the already-emitted \p MemberRegs in its Members operand. A forward
  /// declaration emits \c DebugInfoNone for Size and no members.
  ///
  /// \returns The result id register on success. Returns \c std::nullopt and
  /// emits nothing if the Parent scope cannot be resolved.
  std::optional<MCRegister> emitDebugTypeComposite(
      const DICompositeType *CT, ArrayRef<MCRegister> MemberRegs,
      MCRegister VoidTypeReg, MCRegister I32TypeReg, MCRegister ExtInstSetReg,
      SPIRV::ModuleAnalysisInfo &MAI);

  /// Emit \c DebugTypedef for the typedef derived type \p TD (a \c
  /// DIDerivedType with \c DW_TAG_typedef). Operands: Name, Base Type, Source,
  /// Line, Column, Parent. Parent is the enclosing type when \c TD->getScope()
  /// is an emitted \c DIType, otherwise the first module \c
  /// DebugCompilationUnit.
  ///
  /// \returns The result id register on success. Returns \c std::nullopt and
  /// emits nothing if \p TD's base type has not been emitted into \c
  /// DebugScopeRegs.
  std::optional<MCRegister> emitDebugTypedef(const DIDerivedType *TD,
                                             MCRegister VoidTypeReg,
                                             MCRegister I32TypeReg,
                                             MCRegister ExtInstSetReg,
                                             SPIRV::ModuleAnalysisInfo &MAI);

  /// Map a \c DISubroutineType::getTypeArray() element to an operand register
  /// for
  /// \c DebugTypeFunction. Non-null \p Ty resolves via \c DebugScopeRegs; if
  /// the type was never emitted, returns \c std::nullopt.
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

  /// Map \p Scope to the NonSemantic debug id used as a \c Parent operand.
  ///
  /// Checks \c DebugScopeRegs in order by scope kind. When \p Scope is null, a
  /// \c DIFile, or another scope without a dedicated debug instruction, falls
  /// back to \p FallbackCU or the first module \c DebugCompilationUnit
  /// recorded in \c DebugScopeRegs.
  ///
  /// \returns \c std::nullopt when \p Scope names an emitted scope that has
  /// not been recorded yet, or when no fallback compile unit is available.
  std::optional<MCRegister>
  resolveScope(const DIScope *Scope,
               const DICompileUnit *FallbackCU = nullptr) const;

  /// Emit \c DebugLexicalBlock for \p S, which must be a \c DILexicalBlock or
  /// a \c DINamespace. A \c DILexicalBlock supplies Line/Column
  /// from \c getLine()/getColumn(); a \c DINamespace has neither, so both are
  /// emitted as 0, and its Name is appended as an extra \c OpString operand.
  ///
  /// \returns The result id register on success. Returns \c std::nullopt and
  /// emits nothing if \c resolveScope returns no id for \c S->getScope().
  std::optional<MCRegister>
  emitDebugLexicalBlock(const DIScope *S, MCRegister VoidTypeReg,
                        MCRegister I32TypeReg, MCRegister ExtInstSetReg,
                        SPIRV::ModuleAnalysisInfo &MAI);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVNONSEMANTICDEBUGHANDLER_H
