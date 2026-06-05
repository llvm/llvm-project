//===-- SPIRVNonSemanticDebugHandler.cpp - NSDI AsmPrinter handler -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVNonSemanticDebugHandler.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include <cassert>

using namespace llvm;

namespace {

/// Partition \p Ty into \p BasicTypes, \p PointerTypes, and \p SubroutineTypes
/// for NSDI emission. Used when iterating DebugInfoFinder.types(); each DI
/// node is seen once, so no recursion into pointer bases. Other composites and
/// non-pointer derived kinds are ignored because they are not yet supported.
/// Only types that are supported (later used) are partitioned.
static void
partitionTypes(const DIType *Ty, SmallVector<const DIBasicType *> &BasicTypes,
               SmallVector<const DIDerivedType *> &PointerTypes,
               SmallVector<const DISubroutineType *> &SubroutineTypes) {
  if (const auto *BT = dyn_cast<DIBasicType>(Ty)) {
    BasicTypes.push_back(BT);
    return;
  }
  if (const auto *ST = dyn_cast<DISubroutineType>(Ty)) {
    SubroutineTypes.push_back(ST);
    return;
  }
  const auto *DT = dyn_cast<DIDerivedType>(Ty);
  if (DT && DT->getTag() == dwarf::DW_TAG_pointer_type)
    PointerTypes.push_back(DT);
}

enum : uint32_t {
  NSDIFlagIsProtected = 1u << 0,
  NSDIFlagIsPrivate = 1u << 1,
  NSDIFlagIsPublic = NSDIFlagIsPrivate | NSDIFlagIsProtected,
  NSDIFlagIsLocal = 1u << 2,
  NSDIFlagIsDefinition = 1u << 3,
  NSDIFlagFwdDecl = 1u << 4,
  NSDIFlagArtificial = 1u << 5,
  NSDIFlagExplicit = 1u << 6,
  NSDIFlagPrototyped = 1u << 7,
  NSDIFlagObjectPointer = 1u << 8,
  NSDIFlagStaticMember = 1u << 9,
  NSDIFlagIndirectVariable = 1u << 10,
  NSDIFlagLValueReference = 1u << 11,
  NSDIFlagRValueReference = 1u << 12,
  NSDIFlagIsOptimized = 1u << 13,
  NSDIFlagIsEnumClass = 1u << 14,
  NSDIFlagTypePassByValue = 1u << 15,
  NSDIFlagTypePassByReference = 1u << 16,
  NSDIFlagUnknownPhysicalLayout = 1u << 17,
};

static uint32_t mapDIFlagsToNonSemantic(DINode::DIFlags DFlags) {
  uint32_t Flags = 0;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagPublic)
    Flags |= NSDIFlagIsPublic;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagProtected)
    Flags |= NSDIFlagIsProtected;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagPrivate)
    Flags |= NSDIFlagIsPrivate;
  if (DFlags & DINode::FlagFwdDecl)
    Flags |= NSDIFlagFwdDecl;
  if (DFlags & DINode::FlagArtificial)
    Flags |= NSDIFlagArtificial;
  if (DFlags & DINode::FlagExplicit)
    Flags |= NSDIFlagExplicit;
  if (DFlags & DINode::FlagPrototyped)
    Flags |= NSDIFlagPrototyped;
  if (DFlags & DINode::FlagObjectPointer)
    Flags |= NSDIFlagObjectPointer;
  if (DFlags & DINode::FlagStaticMember)
    Flags |= NSDIFlagStaticMember;
  if (DFlags & DINode::FlagLValueReference)
    Flags |= NSDIFlagLValueReference;
  if (DFlags & DINode::FlagRValueReference)
    Flags |= NSDIFlagRValueReference;
  if (DFlags & DINode::FlagTypePassByValue)
    Flags |= NSDIFlagTypePassByValue;
  if (DFlags & DINode::FlagTypePassByReference)
    Flags |= NSDIFlagTypePassByReference;
  if (DFlags & DINode::FlagEnumClass)
    Flags |= NSDIFlagIsEnumClass;
  return Flags;
}

static uint32_t transDebugFlags(const DINode *DN) {
  uint32_t Flags = 0;
  if (const auto *GV = dyn_cast<DIGlobalVariable>(DN)) {
    if (GV->isLocalToUnit())
      Flags |= NSDIFlagIsLocal;
    if (GV->isDefinition())
      Flags |= NSDIFlagIsDefinition;
  }
  if (const auto *SP = dyn_cast<DISubprogram>(DN)) {
    if (SP->isLocalToUnit())
      Flags |= NSDIFlagIsLocal;
    if (SP->isOptimized())
      Flags |= NSDIFlagIsOptimized;
    if (SP->isDefinition())
      Flags |= NSDIFlagIsDefinition;
    Flags |= mapDIFlagsToNonSemantic(SP->getFlags());
  }
  if (DN->getTag() == dwarf::DW_TAG_reference_type)
    Flags |= NSDIFlagLValueReference;
  if (DN->getTag() == dwarf::DW_TAG_rvalue_reference_type)
    Flags |= NSDIFlagRValueReference;
  if (const auto *Ty = dyn_cast<DIType>(DN))
    Flags |= mapDIFlagsToNonSemantic(Ty->getFlags());
  if (const auto *LV = dyn_cast<DILocalVariable>(DN))
    Flags |= mapDIFlagsToNonSemantic(LV->getFlags());
  return Flags;
}

} // namespace

SPIRVNonSemanticDebugHandler::SPIRVNonSemanticDebugHandler(AsmPrinter &AP)
    : DebugHandlerBase(&AP) {}

// Map DWARF source language codes to NonSemantic.Shader.DebugInfo.100 source
// language codes. Values are from the SourceLanguage enum in the
// NonSemantic.Shader.DebugInfo.100 specification, section 4.3.
unsigned SPIRVNonSemanticDebugHandler::toNSDISrcLang(unsigned DwarfSrcLang) {
  switch (DwarfSrcLang) {
  case dwarf::DW_LANG_OpenCL:
    return 3; // OpenCL_C
  case dwarf::DW_LANG_OpenCL_CPP:
    return 4; // OpenCL_CPP
  case dwarf::DW_LANG_CPP_for_OpenCL:
    return 6; // CPP_for_OpenCL
  case dwarf::DW_LANG_GLSL:
    return 2; // GLSL
  case dwarf::DW_LANG_HLSL:
    return 5; // HLSL
  case dwarf::DW_LANG_SYCL:
    return 7; // SYCL
  case dwarf::DW_LANG_Zig:
    return 12; // Zig
  default:
    return 0; // Unknown
  }
}

void SPIRVNonSemanticDebugHandler::beginModule(Module *M) {
  // The base class sets Asm = nullptr when the module has no compile units,
  // and initializes lexical scope tracking otherwise.
  DebugHandlerBase::beginModule(M);

  if (!Asm)
    return;

  CompileUnits.clear();
  BasicTypes.clear();
  PointerTypes.clear();
  SubroutineTypes.clear();
  DebugTypeRegs.clear();
  OpStringContentCache.clear();
  I32ConstantCache.clear();
  DebugTypeFunctionCache.clear();
  GlobalDIEmitted = false;
#ifndef NDEBUG
  NonSemanticOpStringsSectionEmitted = false;
#endif
  CachedDebugInfoNoneReg = MCRegister();
  CachedOpTypeVoidReg = MCRegister();
  CachedOpTypeInt32Reg = MCRegister();

  // Collect compile-unit info: file paths and source languages.
  for (const DICompileUnit *CU : M->debug_compile_units()) {
    const DIFile *File = CU->getFile();
    CompileUnitInfo Info;
    if (sys::path::is_absolute(File->getFilename()))
      Info.FilePath = File->getFilename();
    else
      sys::path::append(Info.FilePath, File->getDirectory(),
                        File->getFilename());
    // getName() returns the language code regardless of whether the name is
    // versioned. getUnversionedName() would assert on versioned names.
    Info.SpirvSourceLanguage = toNSDISrcLang(CU->getSourceLanguage().getName());
    CompileUnits.push_back(std::move(Info));
  }

  // Collect DWARF version from module flags. For CodeView modules there is no
  // "Dwarf Version" flag; DwarfVersion remains 0, which is the correct value
  // for the DebugCompilationUnit DWARF Version operand in that case.
  if (const NamedMDNode *Flags = M->getNamedMetadata("llvm.module.flags")) {
    for (const auto *Op : Flags->operands()) {
      const MDOperand &NameOp = Op->getOperand(1);
      if (NameOp.equalsStr("Dwarf Version"))
        DwarfVersion =
            cast<ConstantInt>(
                cast<ConstantAsMetadata>(Op->getOperand(2))->getValue())
                ->getSExtValue();
    }
  }

  // Find all debug info types that may be referenced by NSDI instructions.
  DebugInfoFinder Finder;
  Finder.processModule(*M);
  llvm::for_each(Finder.types(), [&](DIType *Ty) {
    partitionTypes(Ty, BasicTypes, PointerTypes, SubroutineTypes);
  });
}

void SPIRVNonSemanticDebugHandler::prepareModuleOutput(
    const SPIRVSubtarget &ST, SPIRV::ModuleAnalysisInfo &MAI) {
  if (CompileUnits.empty())
    return;
  if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_non_semantic_info))
    return;

  // Add the extension to requirements so OpExtension is output.
  MAI.Reqs.addExtension(SPIRV::Extension::SPV_KHR_non_semantic_info);

  // Add the NonSemantic.Shader.DebugInfo.100 entry to ExtInstSetMap so that
  // outputOpExtInstImports() emits the OpExtInstImport instruction. Allocate a
  // fresh result ID for it now; the same ID is used in emitExtInst() operands.
  constexpr unsigned NSSet = static_cast<unsigned>(
      SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100);
  if (!MAI.ExtInstSetMap.count(NSSet))
    MAI.ExtInstSetMap[NSSet] = MAI.getNextIDRegister();
}

void SPIRVNonSemanticDebugHandler::emitMCInst(MCInst &Inst) {
  Asm->OutStreamer->emitInstruction(Inst, Asm->getSubtargetInfo());
}

MCRegister
SPIRVNonSemanticDebugHandler::emitOpString(StringRef S,
                                           SPIRV::ModuleAnalysisInfo &MAI) {
  MCRegister Reg = MAI.getNextIDRegister();
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpString);
  Inst.addOperand(MCOperand::createReg(Reg));
  addStringImm(S, Inst);
  emitMCInst(Inst);
  return Reg;
}

MCRegister SPIRVNonSemanticDebugHandler::emitOpStringIfNew(
    StringRef S, SPIRV::ModuleAnalysisInfo &MAI) {
#ifndef NDEBUG
  assert(!NonSemanticOpStringsSectionEmitted &&
         "emitOpStringIfNew is only valid while emitting SPIR-V section 7");
#endif
  auto [It, Inserted] = OpStringContentCache.try_emplace(S, MCRegister());
  if (!Inserted)
    return It->second;

  MCRegister Reg = emitOpString(S, MAI);
  It->second = Reg;
  return Reg;
}

MCRegister SPIRVNonSemanticDebugHandler::getCachedOpStringReg(StringRef S) {
#ifndef NDEBUG
  assert(NonSemanticOpStringsSectionEmitted &&
         "getCachedOpStringReg requires emitNonSemanticDebugStrings() first");
#endif
  auto It = OpStringContentCache.find(S);
  assert(It != OpStringContentCache.end() &&
         "NSDI OpString missing from cache; emitNonSemanticDebugStrings must "
         "cache every string used in section 10");
  return It->second;
}

MCRegister SPIRVNonSemanticDebugHandler::emitOpConstantI32(
    uint32_t Value, MCRegister I32TypeReg, SPIRV::ModuleAnalysisInfo &MAI) {
  auto [It, Inserted] = I32ConstantCache.try_emplace(Value);
  if (!Inserted)
    return It->second;

  MCRegister Reg = MAI.getNextIDRegister();
  It->second = Reg;
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpConstantI);
  Inst.addOperand(MCOperand::createReg(Reg));
  Inst.addOperand(MCOperand::createReg(I32TypeReg));
  Inst.addOperand(MCOperand::createImm(static_cast<int64_t>(Value)));
  emitMCInst(Inst);
  return Reg;
}

MCRegister SPIRVNonSemanticDebugHandler::emitExtInst(
    SPIRV::NonSemanticExtInst::NonSemanticExtInst Opcode,
    MCRegister VoidTypeReg, MCRegister ExtInstSetReg,
    ArrayRef<MCRegister> Operands, SPIRV::ModuleAnalysisInfo &MAI) {
  MCRegister Reg = MAI.getNextIDRegister();
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpExtInst);
  Inst.addOperand(MCOperand::createReg(Reg));
  Inst.addOperand(MCOperand::createReg(VoidTypeReg));
  Inst.addOperand(MCOperand::createReg(ExtInstSetReg));
  Inst.addOperand(MCOperand::createImm(static_cast<int64_t>(Opcode)));
  for (MCRegister R : Operands)
    Inst.addOperand(MCOperand::createReg(R));
  emitMCInst(Inst);
  return Reg;
}

MCRegister SPIRVNonSemanticDebugHandler::getOrEmitDebugTypeFunction(
    ArrayRef<MCRegister> Ops, MCRegister VoidTypeReg, MCRegister ExtInstSetReg,
    SPIRV::ModuleAnalysisInfo &MAI) {
  auto [It, Inserted] =
      DebugTypeFunctionCache.try_emplace(SmallVector<MCRegister, 8>(Ops));
  if (!Inserted)
    return It->second;

  MCRegister Reg = emitExtInst(SPIRV::NonSemanticExtInst::DebugTypeFunction,
                               VoidTypeReg, ExtInstSetReg, Ops, MAI);
  It->second = Reg;
  return Reg;
}

MCRegister SPIRVNonSemanticDebugHandler::getOrEmitOpTypeVoidReg(
    SPIRV::ModuleAnalysisInfo &MAI) {
  if (!CachedOpTypeVoidReg.isValid())
    CachedOpTypeVoidReg = findOrEmitOpTypeVoid(MAI);
  return CachedOpTypeVoidReg;
}

MCRegister SPIRVNonSemanticDebugHandler::getOrEmitOpTypeInt32Reg(
    SPIRV::ModuleAnalysisInfo &MAI) {
  if (!CachedOpTypeInt32Reg.isValid())
    CachedOpTypeInt32Reg = findOrEmitOpTypeInt32(MAI);
  return CachedOpTypeInt32Reg;
}

MCRegister SPIRVNonSemanticDebugHandler::findOrEmitOpTypeVoid(
    SPIRV::ModuleAnalysisInfo &MAI) {
  for (const MachineInstr *MI : MAI.getMSInstrs(SPIRV::MB_TypeConstVars)) {
    if (MI->getOpcode() == SPIRV::OpTypeVoid)
      return MAI.getRegisterAlias(MI->getMF(), MI->getOperand(0).getReg());
  }
  MCRegister Reg = MAI.getNextIDRegister();
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpTypeVoid);
  Inst.addOperand(MCOperand::createReg(Reg));
  emitMCInst(Inst);
  return Reg;
}

MCRegister SPIRVNonSemanticDebugHandler::findOrEmitOpTypeInt32(
    SPIRV::ModuleAnalysisInfo &MAI) {
  for (const MachineInstr *MI : MAI.getMSInstrs(SPIRV::MB_TypeConstVars)) {
    if (MI->getOpcode() == SPIRV::OpTypeInt &&
        MI->getOperand(1).getImm() == 32 && MI->getOperand(2).getImm() == 0)
      return MAI.getRegisterAlias(MI->getMF(), MI->getOperand(0).getReg());
  }
  MCRegister Reg = MAI.getNextIDRegister();
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpTypeInt);
  Inst.addOperand(MCOperand::createReg(Reg));
  Inst.addOperand(MCOperand::createImm(32)); // width
  Inst.addOperand(MCOperand::createImm(0));  // signedness (unsigned)
  emitMCInst(Inst);
  return Reg;
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDebugTypePointer(
    const DIDerivedType *PT, MCRegister ExtInstSetReg,
    SPIRV::ModuleAnalysisInfo &MAI) {
  // A DWARF address space is required to determine the SPIR-V storage class.
  // Skip pointer types that do not carry one.
  if (!PT->getDWARFAddressSpace().has_value())
    return std::nullopt;

  MCRegister VoidTypeReg = getOrEmitOpTypeVoidReg(MAI);
  MCRegister I32TypeReg = getOrEmitOpTypeInt32Reg(MAI);
  MCRegister DebugTypePointerFlagsReg =
      emitOpConstantI32(transDebugFlags(PT), I32TypeReg, MAI);

  // For SPIR-V targets, Clang sets DwarfAddressSpace to the LLVM IR address
  // space, which addressSpaceToStorageClass expects.
  const auto &ST = static_cast<const SPIRVSubtarget &>(Asm->getSubtargetInfo());
  MCRegister StorageClassReg = emitOpConstantI32(
      addressSpaceToStorageClass(PT->getDWARFAddressSpace().value(), ST),
      I32TypeReg, MAI);

  if (const DIType *BaseTy = PT->getBaseType()) {
    auto BaseIt = DebugTypeRegs.find(BaseTy);
    if (BaseIt != DebugTypeRegs.end())
      return emitExtInst(
          SPIRV::NonSemanticExtInst::DebugTypePointer, VoidTypeReg,
          ExtInstSetReg,
          {BaseIt->second, StorageClassReg, DebugTypePointerFlagsReg}, MAI);
    // Unsupported type, no DebugType* id available.
    return std::nullopt;
  }
  // No getBaseType() (typical for void*): use DebugInfoNone as Base Type,
  // same as SPIRV-LLVM-Translator (see issue #109287 and the DISABLED
  // spirv-val run in debug-type-pointer.ll). spirv-val may still reject this
  // encoding; see https://github.com/KhronosGroup/SPIRV-Registry/pull/287.
  return emitExtInst(
      SPIRV::NonSemanticExtInst::DebugTypePointer, VoidTypeReg, ExtInstSetReg,
      {CachedDebugInfoNoneReg, StorageClassReg, DebugTypePointerFlagsReg}, MAI);
}

std::optional<MCRegister>
SPIRVNonSemanticDebugHandler::emitDebugTypeFunctionForSubroutineType(
    const DISubroutineType *ST, MCRegister ExtInstSetReg,
    SPIRV::ModuleAnalysisInfo &MAI) {
  MCRegister VoidTypeReg = getOrEmitOpTypeVoidReg(MAI);
  MCRegister I32TypeReg = getOrEmitOpTypeInt32Reg(MAI);
  MCRegister DebugTypeFunctionFlagsReg =
      emitOpConstantI32(transDebugFlags(ST), I32TypeReg, MAI);
  DITypeArray TA = ST->getTypeArray();
  SmallVector<MCRegister, 8> Ops;
  Ops.push_back(DebugTypeFunctionFlagsReg);
  // Empty DI type tuple: no explicit return or parameter slots (hand-written IR
  // may use !{}). Emit void-only prototype. Same as SPIRV-LLVM-Translator when
  // DISubroutineType::getTypeArray() has zero elements.
  if (TA.empty()) {
    Ops.push_back(VoidTypeReg);
  } else {
    for (unsigned I = 0, E = TA.size(); I != E; ++I) {
      bool IsReturnType = I == 0;
      auto OptReg = mapDISignatureTypeToReg(TA[I], VoidTypeReg, IsReturnType);
      // No emitted DebugType* id for this slot (e.g., pointer that
      // was skipped due missing address space, etc.).
      if (!OptReg)
        return std::nullopt;
      Ops.push_back(*OptReg);
    }
  }
  return getOrEmitDebugTypeFunction(Ops, VoidTypeReg, ExtInstSetReg, MAI);
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::mapDISignatureTypeToReg(
    const DIType *Ty, MCRegister VoidTypeReg, bool ReturnType) {
  if (!Ty) {
    if (ReturnType)
      return VoidTypeReg;
    assert(CachedDebugInfoNoneReg.isValid() &&
           "DebugInfoNone must be emitted before DISubroutineType operands");
    return CachedDebugInfoNoneReg;
  }
  auto It = DebugTypeRegs.find(Ty);
  if (It != DebugTypeRegs.end())
    return It->second;

  return std::nullopt;
}

void SPIRVNonSemanticDebugHandler::emitNonSemanticDebugStrings(
    SPIRV::ModuleAnalysisInfo &MAI) {
  if (CompileUnits.empty())
    return;
  // Check that prepareModuleOutput() registered the extended instruction set.
  // If the subtarget does not support the extension, neither strings nor ext
  // insts are emitted.
  constexpr unsigned NSSet = static_cast<unsigned>(
      SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100);
  if (!MAI.getExtInstSetReg(NSSet).isValid())
    return;

  for (const CompileUnitInfo &Info : CompileUnits)
    (void)emitOpStringIfNew(Info.FilePath, MAI);

  for (const DIBasicType *BT : BasicTypes)
    (void)emitOpStringIfNew(BT->getName(), MAI);

#ifndef NDEBUG
  NonSemanticOpStringsSectionEmitted = true;
#endif
}

void SPIRVNonSemanticDebugHandler::emitNonSemanticGlobalDebugInfo(
    SPIRV::ModuleAnalysisInfo &MAI) {
  if (GlobalDIEmitted || CompileUnits.empty())
    return;
  GlobalDIEmitted = true;

  // Retrieve the ext inst set register allocated by prepareModuleOutput().
  constexpr unsigned NSSet = static_cast<unsigned>(
      SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100);
  MCRegister ExtInstSetReg = MAI.getExtInstSetReg(NSSet);
  if (!ExtInstSetReg.isValid())
    return; // Extension not available.

#ifndef NDEBUG
  assert(NonSemanticOpStringsSectionEmitted &&
         "emitNonSemanticDebugStrings() must run before "
         "emitNonSemanticGlobalDebugInfo()");
#endif

  MCRegister VoidTypeReg = getOrEmitOpTypeVoidReg(MAI);
  MCRegister I32TypeReg = getOrEmitOpTypeInt32Reg(MAI);

  CachedDebugInfoNoneReg = emitExtInst(SPIRV::NonSemanticExtInst::DebugInfoNone,
                                       VoidTypeReg, ExtInstSetReg, {}, MAI);

  // Emit integer constants shared across all NSDI instructions. The constant
  // cache ensures each value is emitted at most once even when referenced from
  // multiple instructions. All constants are pre-emitted before any DebugSource
  // so that the output order is: constants, then
  // DebugSource+DebugCompilationUnit pairs. This keeps OpConstant instructions
  // grouped before the OpExtInst instructions.

  // The Version operand of DebugCompilationUnit is the version of the
  // NonSemantic.Shader.DebugInfo instruction set, which is 100 for
  // "NonSemantic.Shader.DebugInfo.100" (NonSemanticShaderDebugInfo100Version).
  MCRegister DebugInfoVersionReg = emitOpConstantI32(100, I32TypeReg, MAI);
  MCRegister DwarfVersionReg =
      emitOpConstantI32(static_cast<uint32_t>(DwarfVersion), I32TypeReg, MAI);

  // Pre-emit source language constants for all compile units before entering
  // the DebugSource loop.
  SmallVector<MCRegister> SrcLangRegs =
      map_to_vector(CompileUnits, [&](const CompileUnitInfo &Info) {
        return emitOpConstantI32(Info.SpirvSourceLanguage, I32TypeReg, MAI);
      });

  // Emit DebugSource and DebugCompilationUnit for each compile unit.
  for (auto [Info, SrcLangReg] : llvm::zip(CompileUnits, SrcLangRegs)) {
    MCRegister FileStrReg = getCachedOpStringReg(Info.FilePath);
    MCRegister DebugSourceReg =
        emitExtInst(SPIRV::NonSemanticExtInst::DebugSource, VoidTypeReg,
                    ExtInstSetReg, {FileStrReg}, MAI);
    emitExtInst(
        SPIRV::NonSemanticExtInst::DebugCompilationUnit, VoidTypeReg,
        ExtInstSetReg,
        {DebugInfoVersionReg, DwarfVersionReg, DebugSourceReg, SrcLangReg},
        MAI);
  }

  // Zero constant used as the Flags operand in DebugTypeBasic and
  // DebugTypePointer. Cached with other i32 constants.
  MCRegister I32ZeroReg = emitOpConstantI32(0, I32TypeReg, MAI);

  DebugTypeRegs.clear();

  for (const DIBasicType *BT : BasicTypes) {
    MCRegister NameReg = getCachedOpStringReg(BT->getName());
    MCRegister SizeReg = emitOpConstantI32(
        static_cast<uint32_t>(BT->getSizeInBits()), I32TypeReg, MAI);

    // Map DWARF base type encodings to NSDI encoding codes per
    // NonSemantic.Shader.DebugInfo.100 specification, section 4.5.
    unsigned Encoding = 0; // Unspecified
    switch (BT->getEncoding()) {
    case dwarf::DW_ATE_address:
      Encoding = 1;
      break;
    case dwarf::DW_ATE_boolean:
      Encoding = 2;
      break;
    case dwarf::DW_ATE_float:
      Encoding = 3;
      break;
    case dwarf::DW_ATE_signed:
      Encoding = 4;
      break;
    case dwarf::DW_ATE_signed_char:
      Encoding = 5;
      break;
    case dwarf::DW_ATE_unsigned:
      Encoding = 6;
      break;
    case dwarf::DW_ATE_unsigned_char:
      Encoding = 7;
      break;
    }
    MCRegister EncodingReg = emitOpConstantI32(Encoding, I32TypeReg, MAI);

    MCRegister BTReg = emitExtInst(
        SPIRV::NonSemanticExtInst::DebugTypeBasic, VoidTypeReg, ExtInstSetReg,
        {NameReg, SizeReg, EncodingReg, I32ZeroReg}, MAI);
    DebugTypeRegs[BT] = BTReg;
  }

  // Emit DebugTypePointer for each referenced pointer type.
  for (const DIDerivedType *PT : PointerTypes) {
    if (auto PtrReg = emitDebugTypePointer(PT, ExtInstSetReg, MAI))
      DebugTypeRegs[PT] = *PtrReg;
  }

  // Emit DebugTypeFunction for each distinct DISubroutineType.
  for (const DISubroutineType *ST : SubroutineTypes) {
    if (auto FnTyReg =
            emitDebugTypeFunctionForSubroutineType(ST, ExtInstSetReg, MAI))
      DebugTypeRegs[ST] = *FnTyReg;
  }
}
