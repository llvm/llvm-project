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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include <cassert>

using namespace llvm;

namespace {

/// Look up \p Key in a register map and return its value, or std::nullopt when
/// the key is absent.
template <typename MapT>
static std::optional<MCRegister> lookupOptReg(const MapT &Map,
                                              typename MapT::key_type Key) {
  auto It = Map.find(Key);
  if (It == Map.end())
    return std::nullopt;
  assert(It->second.isValid() && "invalid register stored in map");
  return It->second;
}

/// Partition \p Ty into \p BasicTypes, \p PointerTypes, \p SubroutineTypes,
/// \p VectorTypes, \p ArrayTypes, \p CompositeTypes, and \p TypedefTypes for
/// NSDI emission. Used when iterating DebugInfoFinder.types(); each DI node is
/// seen once, so no recursion into pointer bases. Other composites and the
/// remaining derived kinds are ignored because they are not yet supported.
/// Only types that are supported (later used) are partitioned.
static void
partitionTypes(const DIType *Ty, SmallVector<const DIBasicType *> &BasicTypes,
               SmallVector<const DIDerivedType *> &PointerTypes,
               SmallVector<const DISubroutineType *> &SubroutineTypes,
               SmallVector<const DICompositeType *> &VectorTypes,
               SmallVector<const DICompositeType *> &ArrayTypes,
               SmallVector<const DICompositeType *> &CompositeTypes,
               SmallVector<const DIDerivedType *> &TypedefTypes) {
  if (const auto *BT = dyn_cast<DIBasicType>(Ty)) {
    BasicTypes.push_back(BT);
    return;
  }
  if (const auto *ST = dyn_cast<DISubroutineType>(Ty)) {
    SubroutineTypes.push_back(ST);
    return;
  }
  if (const auto *CT = dyn_cast<DICompositeType>(Ty)) {
    if (CT->getTag() == dwarf::DW_TAG_array_type) {
      // A vector is an array with DINode::FlagVector. A plain array is the
      // same tag without it. A matrix is also lowered to a DW_TAG_array_type
      // (two subranges), so it is indistinguishable from a 2D array here and
      // is emitted as a DebugTypeArray.
      //
      // FIXME: Emitting a matrix as a DebugTypeArray is valid but loses the
      // matrix shape. DWARF has no matrix tag, so distinguishing a matrix needs
      // a new DINode flag analogous to FlagVector, set on the array, plus a way
      // to carry column-major vs row-major traits. Array-of-vectors alone would
      // not disambiguate a matrix from a genuine array of vectors. Once the
      // frontend marks matrices, route them to a DebugTypeMatrix path here.
      if (CT->isVector())
        VectorTypes.push_back(CT);
      else
        ArrayTypes.push_back(CT);
    } else if (CT->getTag() == dwarf::DW_TAG_structure_type ||
               CT->getTag() == dwarf::DW_TAG_class_type ||
               CT->getTag() == dwarf::DW_TAG_union_type) {
      CompositeTypes.push_back(CT);
    }
    return;
  }
  const auto *DT = dyn_cast<DIDerivedType>(Ty);
  if (DT && DT->getTag() == dwarf::DW_TAG_pointer_type)
    PointerTypes.push_back(DT);
  else if (DT && DT->getTag() == dwarf::DW_TAG_typedef)
    TypedefTypes.push_back(DT);
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

// Map a DWARF composite tag to a NonSemantic.Shader.DebugInfo Composite Type
// value: Class 0, Structure 1, Union 2.
static uint32_t mapCompositeTypeTag(unsigned Tag) {
  switch (Tag) {
  case dwarf::DW_TAG_class_type:
    return 0;
  case dwarf::DW_TAG_structure_type:
    return 1;
  case dwarf::DW_TAG_union_type:
    return 2;
  default:
    reportFatalInternalError("unexpected DWARF composite tag " + Twine(Tag) +
                             ". Expecting 0, 1 or 2");
  }
}

static const MachineInstr *
findLastFunctionOpVariableDeclaration(const MachineFunction &MF,
                                      SPIRV::ModuleAnalysisInfo &MAI) {

  // We iterate over the instructions to find the last OpVariable instruction if
  // any. The following SPIRV rule is used to terminate the traversal earlier:
  // SPIR-V 2.16.1, Function Structure: "All OpVariable instructions in a
  // function must be in the first block in the function. These instructions,
  // together with any intermixed OpLine and OpNoLine instructions, must be the
  // first instructions in that block."
  const MachineInstr *LastOpVariable = nullptr;
  bool SeenOpVariable = false;
  for (const MachineInstr &MI : MF.front()) {
    if (MI.getOpcode() == SPIRV::OpVariable) {
      SeenOpVariable = true;
      if (!MAI.getSkipEmission(&MI))
        LastOpVariable = &MI;
      continue;
    }

    bool CanInterleaveWithOpVariable =
        MI.getOpcode() == SPIRV::OpLine || MI.getOpcode() == SPIRV::OpNoLine;
    if (SeenOpVariable && !CanInterleaveWithOpVariable &&
        !MAI.getSkipEmission(&MI))
      break;
  }
  return LastOpVariable;
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

// Collect distinct DILocations from LLVM IR. DebugLine pre-emission and MIR
// lookups assume every machine-instruction debug location already appeared
// here; a codegen-only location would not be collected and emission will be
// skipped.
static void collectUniqueDebugLocations(const Module &M,
                                        SetVector<const DILocation *> &Out) {
  for (const Function &F : M) {
    if (!F.getSubprogram())
      continue;
    for (const Instruction &I : instructions(F)) {
      if (const DILocation *DL = I.getDebugLoc().get())
        Out.insert(DL);
      for (DbgRecord &DR : I.getDbgRecordRange())
        if (const DILocation *DL = DR.getDebugLoc().get())
          Out.insert(DL);
    }
  }
}

// Insert \p S and its enclosing DILexicalBlock/DINamespace chain into \p Out,
// parent before child, so single-pass emission never needs a forward
// reference for the Parent operand.
static void collectLexicalBlockChain(const DIScope *S,
                                     SetVector<const DIScope *> &Out) {
  // Walk up child-first, then insert in reverse to get parents in first.
  SmallVector<const DIScope *, 8> Chain;
  while (S && !Out.contains(S) && isa<DILexicalBlock, DINamespace>(S)) {
    Chain.push_back(S);
    S = S->getScope();
  }
  Out.insert(Chain.rbegin(), Chain.rend());
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
  VectorTypes.clear();
  ArrayTypes.clear();
  CompositeTypes.clear();
  TypedefTypes.clear();
  SubprogramDeclarations.clear();
  SubprogramDefinitions.clear();
  UniqueDebugLocations.clear();
  GlobalVariableDebugInfoMap.clear();
  LexicalBlocks.clear();
  DebugScopeRegs.clear();
  ScopeToPathOpStringReg.clear();
  DebugSourceRegByFileStr.clear();
  OpStringContentCache.clear();
  I32ConstantCache.clear();
  DebugTypeFunctionCache.clear();
  GlobalDIEmitted = false;
  GlobalNSDIEnabled = false;
  CurrentMAI = nullptr;
#ifndef NDEBUG
  NonSemanticOpStringsSectionEmitted = false;
#endif
  CachedDebugInfoNoneReg = MCRegister();
  CachedEmptyStringReg = MCRegister();
  CachedOpTypeVoidReg = MCRegister();
  CachedOpTypeInt32Reg = MCRegister();

  // Collect compile-unit info: file paths and source languages.
  for (const DICompileUnit *CU : M->debug_compile_units()) {
    const DIFile *File = CU->getFile();
    CompileUnitInfo Info;
    Info.TheCU = CU;
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
    partitionTypes(Ty, BasicTypes, PointerTypes, SubroutineTypes, VectorTypes,
                   ArrayTypes, CompositeTypes, TypedefTypes);
  });

  for (const DISubprogram *SP : Finder.subprograms()) {
    if (SP->isDefinition())
      SubprogramDefinitions.push_back(SP);
    else
      SubprogramDeclarations.push_back(SP);
  }

  // Walk LLVM globals to map each DIGlobalVariable to its llvm::GlobalVariable.
  DenseMap<const DIGlobalVariable *, const GlobalVariable *> DIGVToLLVMGV;
  for (const GlobalVariable &G : M->globals()) {
    SmallVector<DIGlobalVariableExpression *> GVEs;
    G.getDebugInfo(GVEs);
    for (DIGlobalVariableExpression *GVE : GVEs) {
      if (const DIGlobalVariable *GV = GVE->getVariable()) {
        DIGVToLLVMGV.try_emplace(GV, &G);
      }
    }
  }

  for (const DIGlobalVariableExpression *GVE : Finder.global_variables()) {
    const DIGlobalVariable *GV = GVE->getVariable();
    const DIExpression *Expr = GVE->getExpression();
    GlobalVariableDebugInfoMap.try_emplace(
        GV, GlobalVariableDebugInfo{Expr, DIGVToLLVMGV.lookup(GV)});
  }

  collectUniqueDebugLocations(*M, UniqueDebugLocations);

  // DILexicalBlock and DINamespace scopes are lowered to DebugLexicalBlock.
  // Collect them in parent-before-child order so they can be later emitted in a
  // single pass.
  for (const DIScope *S : Finder.scopes())
    collectLexicalBlockChain(S, LexicalBlocks);
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
  if (Inserted)
    It->second = emitOpString(S, MAI);

  return It->second;
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

MCRegister SPIRVNonSemanticDebugHandler::emitAndCacheScopePathOpStringReg(
    const DIScope *Scope, SPIRV::ModuleAnalysisInfo &MAI) {
  auto [It, Inserted] = ScopeToPathOpStringReg.try_emplace(Scope, MCRegister());
  if (Inserted)
    It->second = emitOpStringIfNew(getDebugFullPath(Scope), MAI);
  return It->second;
}

MCRegister SPIRVNonSemanticDebugHandler::getCachedScopePathOpStringReg(
    const DIScope *Scope, bool UseEmptyPathIfNullScope) {
  if (!Scope) {
    assert(UseEmptyPathIfNullScope &&
           "null scope path lookup requires UseEmptyPathIfNullScope");
    assert(CachedEmptyStringReg.isValid() &&
           "empty path OpString must be cached in emitNonSemanticDebugStrings");
    return CachedEmptyStringReg;
  }
  auto It = ScopeToPathOpStringReg.find(Scope);
  assert(It != ScopeToPathOpStringReg.end() &&
         "path OpString must be cached in emitNonSemanticDebugStrings");
  MCRegister FileStrReg = It->second;
  assert(FileStrReg.isValid() && "path OpString id must be valid once cached");
  return FileStrReg;
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
    auto BaseIt = DebugScopeRegs.find(BaseTy);
    if (BaseIt != DebugScopeRegs.end())
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
      bool IsReturnType = (I == 0);
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

// Match SPIRV-LLVM-Translator's selection logic for the Parent operand.
std::optional<MCRegister> SPIRVNonSemanticDebugHandler::resolveScope(
    const DIScope *Scope, const DICompileUnit *FallbackCU) const {

  if (isa_and_nonnull<DIType, DILexicalBlock, DINamespace, DISubprogram>(Scope))
    return lookupOptReg(DebugScopeRegs, Scope);

  // For a file, compile-unit, or absent scope, fall back to a compile unit.
  if (FallbackCU)
    return lookupOptReg(DebugScopeRegs, FallbackCU);

  if (CompileUnits.empty())
    return std::nullopt;

  return lookupOptReg(DebugScopeRegs, CompileUnits[0].TheCU);
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDebugLexicalBlock(
    const DIScope *S, MCRegister VoidTypeReg, MCRegister I32TypeReg,
    MCRegister ExtInstSetReg, SPIRV::ModuleAnalysisInfo &MAI) {
  assert((isa<DILexicalBlock, DINamespace>(S)) &&
         "S must be a DILexicalBlock or DINamespace in emitDebugLexicalBlock");
  auto ParentRegOpt = resolveScope(S->getScope());
  if (!ParentRegOpt)
    return std::nullopt;

  MCRegister FileStrReg = getCachedScopePathOpStringReg(
      S->getFile(), /*UseEmptyPathIfNullScope=*/true);
  MCRegister SrcReg = getOrEmitDebugSourceForFileStrReg(FileStrReg, VoidTypeReg,
                                                        ExtInstSetReg, MAI);

  SmallVector<MCRegister, 5> Ops;
  if (const auto *LB = dyn_cast<DILexicalBlock>(S)) {
    MCRegister LineReg = emitOpConstantI32(static_cast<uint32_t>(LB->getLine()),
                                           I32TypeReg, MAI);
    MCRegister ColReg = emitOpConstantI32(
        static_cast<uint32_t>(LB->getColumn()), I32TypeReg, MAI);
    Ops = {SrcReg, LineReg, ColReg, *ParentRegOpt};
  } else {
    const auto *NS = cast<DINamespace>(S);
    // DINamespace carries no line/column info.
    MCRegister LineReg = emitOpConstantI32(0, I32TypeReg, MAI);
    MCRegister ColReg = emitOpConstantI32(0, I32TypeReg, MAI);
    MCRegister NameReg = getCachedOpStringReg(NS->getName());
    Ops = {SrcReg, LineReg, ColReg, *ParentRegOpt, NameReg};
  }

  return emitExtInst(SPIRV::NonSemanticExtInst::DebugLexicalBlock, VoidTypeReg,
                     ExtInstSetReg, Ops, MAI);
}

std::optional<MCRegister>
SPIRVNonSemanticDebugHandler::emitDebugFunctionDeclaration(
    const DISubprogram *SP, MCRegister VoidTypeReg, MCRegister I32TypeReg,
    MCRegister ExtInstSetReg, SPIRV::ModuleAnalysisInfo &MAI) {
  assert(SP && "SP must not be null in emitDebugFunctionDeclaration");
  assert(!SP->isDefinition() &&
         "SP must not be a definition in emitDebugFunctionDeclaration");

  // The IR verifier already enforces that this cannot be null.
  const DISubroutineType *ST = SP->getType();

  auto FnTyRegOpt = lookupOptReg(DebugScopeRegs, ST);
  if (!FnTyRegOpt)
    return std::nullopt;
  MCRegister FnTyReg = *FnTyRegOpt;

  auto ParentRegOpt = resolveScope(SP->getScope(), SP->getUnit());
  if (!ParentRegOpt)
    return std::nullopt;

  MCRegister ParentReg = *ParentRegOpt;

  MCRegister FileStrReg = getCachedScopePathOpStringReg(SP);

  MCRegister NameReg = getCachedOpStringReg(SP->getName());
  MCRegister LinkageReg = getCachedOpStringReg(SP->getLinkageName());
  MCRegister SrcReg = getOrEmitDebugSourceForFileStrReg(FileStrReg, VoidTypeReg,
                                                        ExtInstSetReg, MAI);

  MCRegister LineReg =
      emitOpConstantI32(static_cast<uint32_t>(SP->getLine()), I32TypeReg, MAI);
  MCRegister ColReg = emitOpConstantI32(0, I32TypeReg, MAI);

  uint32_t FlagsVal = transDebugFlags(SP);
  // TODO: When composite scopes are DebugFunctionDeclaration parents (available
  // in DebugScopeRegs), sync declaration Flags with SPIRV-LLVM-Translator.
  FlagsVal &= ~NSDIFlagIsDefinition;
  MCRegister FlagsReg = emitOpConstantI32(FlagsVal, I32TypeReg, MAI);

  return emitExtInst(SPIRV::NonSemanticExtInst::DebugFunctionDeclaration,
                     VoidTypeReg, ExtInstSetReg,
                     {NameReg, FnTyReg, SrcReg, LineReg, ColReg, ParentReg,
                      LinkageReg, FlagsReg},
                     MAI);
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDebugFunction(
    const DISubprogram *SP, MCRegister VoidTypeReg, MCRegister I32TypeReg,
    MCRegister ExtInstSetReg, SPIRV::ModuleAnalysisInfo &MAI) {
  assert(SP && "SP must not be null in emitDebugFunction");
  assert(SP->isDefinition() && "SP must be a definition in emitDebugFunction");

  const DISubroutineType *ST = SP->getType();
  auto FnTyRegOpt = lookupOptReg(DebugScopeRegs, ST);
  if (!FnTyRegOpt)
    return std::nullopt;

  auto ParentRegOpt = resolveScope(SP->getScope(), SP->getUnit());
  if (!ParentRegOpt)
    return std::nullopt;

  MCRegister NameReg = getCachedOpStringReg(SP->getName());
  MCRegister LinkageReg = getCachedOpStringReg(SP->getLinkageName());
  MCRegister FileStrReg = getCachedScopePathOpStringReg(SP);
  MCRegister SrcReg = getOrEmitDebugSourceForFileStrReg(FileStrReg, VoidTypeReg,
                                                        ExtInstSetReg, MAI);

  MCRegister LineReg =
      emitOpConstantI32(static_cast<uint32_t>(SP->getLine()), I32TypeReg, MAI);
  // LLVM's DISubprogram has no column field but SPIR-V expects one in
  // DebugFunction.
  MCRegister ColReg = emitOpConstantI32(0, I32TypeReg, MAI);
  MCRegister FlagsReg = emitOpConstantI32(transDebugFlags(SP), I32TypeReg, MAI);
  MCRegister ScopeLineReg = emitOpConstantI32(
      static_cast<uint32_t>(SP->getScopeLine()), I32TypeReg, MAI);

  SmallVector<MCRegister, 10> Ops = {NameReg,    *FnTyRegOpt, SrcReg,
                                     LineReg,    ColReg,      *ParentRegOpt,
                                     LinkageReg, FlagsReg,    ScopeLineReg};

  if (const DISubprogram *Decl = SP->getDeclaration()) {
    if (auto DeclRegOpt = lookupOptReg(DebugScopeRegs, Decl))
      Ops.push_back(*DeclRegOpt);
  }

  return emitExtInst(SPIRV::NonSemanticExtInst::DebugFunction, VoidTypeReg,
                     ExtInstSetReg, Ops, MAI);
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
  return lookupOptReg(DebugScopeRegs, Ty);
}

// Unimplemented no-op; see emitDebugExpression declaration.
std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDebugExpression(
    const DIExpression *, MCRegister, MCRegister, SPIRV::ModuleAnalysisInfo &) {
  return std::nullopt;
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDebugGlobalVariable(
    const DIGlobalVariable *GV, const GlobalVariableDebugInfo &Info,
    MCRegister VoidTypeReg, MCRegister I32TypeReg, MCRegister ExtInstSetReg,
    SPIRV::ModuleAnalysisInfo &MAI) {
  assert(GV && "GV must not be null in emitDebugGlobalVariable");

  auto ParentRegOpt = resolveScope(GV->getScope());
  if (!ParentRegOpt)
    return std::nullopt;

  MCRegister ParentReg = *ParentRegOpt;

  // TyReg: DebugInfoNone when GV has no DI type (as done in
  // SPIRV-LLVM-Translator). Declarations (isDefinition: false) can have null
  // getType() while definitions must have a non-null one (enforced by the IR
  // verifier).
  MCRegister TyReg = CachedDebugInfoNoneReg;
  if (const DIType *Ty = GV->getType()) {
    auto TyRegOpt = lookupOptReg(DebugScopeRegs, Ty);
    if (!TyRegOpt)
      return std::nullopt;
    TyReg = *TyRegOpt;
  }

  std::optional<MCRegister> StaticMemberRegOpt;
  if (const DIDerivedType *SM = GV->getStaticDataMemberDeclaration()) {
    StaticMemberRegOpt = lookupOptReg(DebugScopeRegs, SM);
    if (!StaticMemberRegOpt)
      return std::nullopt;
  }

  MCRegister NameReg = getCachedOpStringReg(GV->getName());
  MCRegister LinkageReg = getCachedOpStringReg(GV->getLinkageName());
  MCRegister FileStrReg = getCachedScopePathOpStringReg(
      GV->getFile(), /*UseEmptyPathIfNullScope=*/true);
  MCRegister SrcReg = getOrEmitDebugSourceForFileStrReg(FileStrReg, VoidTypeReg,
                                                        ExtInstSetReg, MAI);

  MCRegister LineReg =
      emitOpConstantI32(static_cast<uint32_t>(GV->getLine()), I32TypeReg, MAI);
  // DIGlobalVariable or DIGlobalVariableExpression metadata carry no column
  // field. Column is hardcoded to 0 (because it can't be determined), matching
  // SPIRV-LLVM-Translator.
  MCRegister ColReg = emitOpConstantI32(0, I32TypeReg, MAI);

  // Variable: @g OpVariable id when !dbg matches; else a DebugExpression for
  // the GVE init value when no @g exists; else DebugInfoNone.
  MCRegister VariableReg = CachedDebugInfoNoneReg;
  if (const GlobalVariable *LLVMGV = Info.LLVMGV) {
    MCRegister GVReg = MAI.getGlobalObjReg(LLVMGV);
    if (GVReg.isValid())
      VariableReg = GVReg;
  } else if (Info.Expr) {
    if (auto ExprReg =
            emitDebugExpression(Info.Expr, VoidTypeReg, ExtInstSetReg, MAI))
      VariableReg = *ExprReg;
  }

  MCRegister FlagsReg = emitOpConstantI32(transDebugFlags(GV), I32TypeReg, MAI);

  SmallVector<MCRegister, 10> Ops = {NameReg,    TyReg,       SrcReg,
                                     LineReg,    ColReg,      ParentReg,
                                     LinkageReg, VariableReg, FlagsReg};

  if (StaticMemberRegOpt)
    Ops.push_back(*StaticMemberRegOpt);

  return emitExtInst(SPIRV::NonSemanticExtInst::DebugGlobalVariable,
                     VoidTypeReg, ExtInstSetReg, Ops, MAI);
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDebugTypeVector(
    const DICompositeType *VT, MCRegister ExtInstSetReg,
    SPIRV::ModuleAnalysisInfo &MAI) {
  const auto *BaseTy = dyn_cast_or_null<DIBasicType>(VT->getBaseType());
  if (!BaseTy)
    return std::nullopt;
  auto BTIt = DebugScopeRegs.find(BaseTy);
  if (BTIt == DebugScopeRegs.end())
    return std::nullopt;

  // DebugTypeVector models only 1D vectors (multi-subrange types cannot be
  // encoded).
  DINodeArray Elements = VT->getElements();
  if (Elements.size() != 1)
    return std::nullopt;
  const auto *SR = cast<DISubrange>(Elements[0]);
  const auto *CI = dyn_cast_if_present<ConstantInt *>(SR->getCount());
  if (!CI)
    return std::nullopt;

  MCRegister VoidTypeReg = getOrEmitOpTypeVoidReg(MAI);
  MCRegister I32TypeReg = getOrEmitOpTypeInt32Reg(MAI);
  MCRegister CountReg = emitOpConstantI32(
      static_cast<uint32_t>(CI->getZExtValue()), I32TypeReg, MAI);
  return emitExtInst(SPIRV::NonSemanticExtInst::DebugTypeVector, VoidTypeReg,
                     ExtInstSetReg, {BTIt->second, CountReg}, MAI);
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDebugTypeArray(
    const DICompositeType *AT, MCRegister ExtInstSetReg,
    SPIRV::ModuleAnalysisInfo &MAI) {
  // The element (base) type must already be in DebugScopeRegs. Unlike
  // DebugTypeVector, the element may be any debug type, not only a basic type.
  auto BaseRegOpt = lookupOptReg(DebugScopeRegs, AT->getBaseType());
  if (!BaseRegOpt)
    return std::nullopt;

  MCRegister VoidTypeReg = getOrEmitOpTypeVoidReg(MAI);
  MCRegister I32TypeReg = getOrEmitOpTypeInt32Reg(MAI);

  SmallVector<MCRegister> Ops;
  Ops.push_back(*BaseRegOpt);

  // One component count per DISubrange, in DWARF subrange order. Emit 0 for
  // counts that are not a compile-time constant (dynamic arrays). This matches
  // OpTypeRuntimeArray.
  for (const DINode *Element : AT->getElements()) {
    const auto *SR = dyn_cast<DISubrange>(Element);
    if (!SR)
      continue;
    // A DIVariable count (a variable-length array) is not a ConstantInt, so it
    // maps to 0 here. DebugTypeArray also allows a DebugLocalVariable or
    // DebugGlobalVariable id for it, but no frontend we target emits one. A
    // constant wider than 32 bits maps to 0 too, since the count operand is a
    // 32-bit OpConstant and such an array cannot occur in a shader.
    uint32_t Count = 0;
    if (const auto *CI = dyn_cast_if_present<ConstantInt *>(SR->getCount())) {
      const APInt &Value = CI->getValue();
      if (Value.getActiveBits() <= 32)
        Count = static_cast<uint32_t>(Value.getZExtValue());
    }
    Ops.push_back(emitOpConstantI32(Count, I32TypeReg, MAI));
  }

  return emitExtInst(SPIRV::NonSemanticExtInst::DebugTypeArray, VoidTypeReg,
                     ExtInstSetReg, Ops, MAI);
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDebugTypeMember(
    const DIDerivedType *M, MCRegister VoidTypeReg, MCRegister I32TypeReg,
    MCRegister ExtInstSetReg, SPIRV::ModuleAnalysisInfo &MAI) {
  // The member type must already be in DebugScopeRegs.
  auto TyRegOpt = lookupOptReg(DebugScopeRegs, M->getBaseType());
  if (!TyRegOpt)
    return std::nullopt;

  MCRegister NameReg = getCachedOpStringReg(M->getName());
  MCRegister FileStrReg = getCachedScopePathOpStringReg(
      M->getFile(), /*UseEmptyPathIfNullScope=*/true);
  MCRegister SrcReg = getOrEmitDebugSourceForFileStrReg(FileStrReg, VoidTypeReg,
                                                        ExtInstSetReg, MAI);
  MCRegister LineReg =
      emitOpConstantI32(static_cast<uint32_t>(M->getLine()), I32TypeReg, MAI);

  // DIDerivedType members carry no column, so emit 0.
  MCRegister ColReg = emitOpConstantI32(0, I32TypeReg, MAI);
  MCRegister OffsetReg = emitOpConstantI32(
      static_cast<uint32_t>(M->getOffsetInBits()), I32TypeReg, MAI);
  MCRegister SizeReg = emitOpConstantI32(
      static_cast<uint32_t>(M->getSizeInBits()), I32TypeReg, MAI);
  MCRegister FlagsReg = emitOpConstantI32(transDebugFlags(M), I32TypeReg, MAI);

  // In NonSemantic.Shader.DebugInfo a DebugTypeMember has no Parent operand:
  // only the composite references its members. This is by design, it drops the
  // Parent that OpenCL.DebugInfo.100 had, and it avoids a composite/member
  // reference cycle.
  //
  // FIXME: Static members are not handled yet: their constant initializer is
  // available but is not emitted as the optional Value operand, and under DWARF
  // 5 a static member is tagged DW_TAG_variable, which the caller's member loop
  // skips.
  return emitExtInst(SPIRV::NonSemanticExtInst::DebugTypeMember, VoidTypeReg,
                     ExtInstSetReg,
                     {NameReg, *TyRegOpt, SrcReg, LineReg, ColReg, OffsetReg,
                      SizeReg, FlagsReg},
                     MAI);
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDebugTypeComposite(
    const DICompositeType *CT, ArrayRef<MCRegister> MemberRegs,
    MCRegister VoidTypeReg, MCRegister I32TypeReg, MCRegister ExtInstSetReg,
    SPIRV::ModuleAnalysisInfo &MAI) {
  auto ParentRegOpt = resolveScope(CT->getScope());
  if (!ParentRegOpt)
    return std::nullopt;

  MCRegister NameReg = getCachedOpStringReg(CT->getName());
  MCRegister LinkageReg = getCachedOpStringReg(CT->getIdentifier());
  MCRegister FileStrReg = getCachedScopePathOpStringReg(
      CT->getFile(), /*UseEmptyPathIfNullScope=*/true);
  MCRegister SrcReg = getOrEmitDebugSourceForFileStrReg(FileStrReg, VoidTypeReg,
                                                        ExtInstSetReg, MAI);

  MCRegister TagReg =
      emitOpConstantI32(mapCompositeTypeTag(CT->getTag()), I32TypeReg, MAI);
  MCRegister LineReg =
      emitOpConstantI32(static_cast<uint32_t>(CT->getLine()), I32TypeReg, MAI);
  MCRegister ColReg = emitOpConstantI32(0, I32TypeReg, MAI);

  // A forward declaration has no known size or members: Size is DebugInfoNone.
  MCRegister SizeReg = CachedDebugInfoNoneReg;
  if (!CT->isForwardDecl())
    SizeReg = emitOpConstantI32(static_cast<uint32_t>(CT->getSizeInBits()),
                                I32TypeReg, MAI);

  MCRegister FlagsReg = emitOpConstantI32(transDebugFlags(CT), I32TypeReg, MAI);

  SmallVector<MCRegister> Ops = {NameReg,    TagReg,  SrcReg,
                                 LineReg,    ColReg,  *ParentRegOpt,
                                 LinkageReg, SizeReg, FlagsReg};
  Ops.append(MemberRegs.begin(), MemberRegs.end());
  return emitExtInst(SPIRV::NonSemanticExtInst::DebugTypeComposite, VoidTypeReg,
                     ExtInstSetReg, Ops, MAI);
}

std::optional<MCRegister> SPIRVNonSemanticDebugHandler::emitDebugTypedef(
    const DIDerivedType *TD, MCRegister VoidTypeReg, MCRegister I32TypeReg,
    MCRegister ExtInstSetReg, SPIRV::ModuleAnalysisInfo &MAI) {
  // The underlying (base) type must already be in DebugScopeRegs.
  auto BaseRegOpt = lookupOptReg(DebugScopeRegs, TD->getBaseType());
  if (!BaseRegOpt)
    return std::nullopt;

  MCRegister NameReg = getCachedOpStringReg(TD->getName());
  MCRegister FileStrReg = getCachedScopePathOpStringReg(
      TD->getFile(), /*UseEmptyPathIfNullScope=*/true);
  MCRegister SrcReg = getOrEmitDebugSourceForFileStrReg(FileStrReg, VoidTypeReg,
                                                        ExtInstSetReg, MAI);
  MCRegister LineReg =
      emitOpConstantI32(static_cast<uint32_t>(TD->getLine()), I32TypeReg, MAI);
  // DIDerivedType typedefs carry no column, so emit 0.
  MCRegister ColReg = emitOpConstantI32(0, I32TypeReg, MAI);

  // Parent must be a lexical scope. Valid NSDI lexical scopes are
  // DebugCompilationUnit, DebugFunction, DebugLexicalBlock, or
  // DebugTypeComposite.
  auto ParentRegOpt = resolveScope(TD->getScope());
  if (!ParentRegOpt)
    return std::nullopt;
  MCRegister ParentReg = *ParentRegOpt;

  return emitExtInst(
      SPIRV::NonSemanticExtInst::DebugTypedef, VoidTypeReg, ExtInstSetReg,
      {NameReg, *BaseRegOpt, SrcReg, LineReg, ColReg, ParentReg}, MAI);
}

void SPIRVNonSemanticDebugHandler::emitNonSemanticDebugStrings(
    SPIRV::ModuleAnalysisInfo &MAI) {
  if (CompileUnits.empty())
    return;
  // Check that prepareModuleOutput() registered the extended instruction set.
  // If the subtarget does not support the extension, neither strings nor ext
  // insts are emitted.
  if (!MAI.getExtInstSetReg(NSSet).isValid())
    return;

  for (const CompileUnitInfo &Info : CompileUnits) {
    if (Info.TheCU) {
      MCRegister PathReg = emitOpStringIfNew(Info.FilePath, MAI);
      ScopeToPathOpStringReg[Info.TheCU] = PathReg;
      if (const DIFile *F = Info.TheCU->getFile())
        ScopeToPathOpStringReg[F] = PathReg;
    }
  }

  for (const DIBasicType *BT : BasicTypes)
    emitOpStringIfNew(BT->getName(), MAI);

  for (const DISubprogram *SP : concat<const DISubprogram *>(
           SubprogramDeclarations, SubprogramDefinitions)) {
    emitOpStringIfNew(SP->getName(), MAI);
    emitOpStringIfNew(SP->getLinkageName(), MAI);
    emitAndCacheScopePathOpStringReg(SP, MAI);
  }

  // Cache the OpStrings each DebugTypeComposite and its DebugTypeMembers use:
  // the composite name, identifier (linkage name), and path, plus each member
  // name and path.
  for (const DICompositeType *CT : CompositeTypes) {
    emitOpStringIfNew(CT->getName(), MAI);
    emitOpStringIfNew(CT->getIdentifier(), MAI);
    emitAndCacheScopePathOpStringReg(CT->getFile(), MAI);
    for (const DINode *Element : CT->getElements()) {
      const auto *M = dyn_cast<DIDerivedType>(Element);
      if (!M || M->getTag() != dwarf::DW_TAG_member)
        continue;
      emitOpStringIfNew(M->getName(), MAI);
      emitAndCacheScopePathOpStringReg(M->getFile(), MAI);
    }
  }

  // Cache the name and path OpStrings each DebugTypedef uses.
  for (const DIDerivedType *TD : TypedefTypes) {
    emitOpStringIfNew(TD->getName(), MAI);
    emitAndCacheScopePathOpStringReg(TD->getFile(), MAI);
  }

  for (const auto &[GV, _] : GlobalVariableDebugInfoMap) {
    emitOpStringIfNew(GV->getName(), MAI);
    emitOpStringIfNew(GV->getLinkageName(), MAI);
    emitAndCacheScopePathOpStringReg(GV->getFile(), MAI);
  }

  // Cache the path OpString each DebugLexicalBlock uses (source file), plus
  // the Name OpString for the DINamespace case.
  for (const DIScope *S : LexicalBlocks) {
    emitAndCacheScopePathOpStringReg(S->getFile(), MAI);
    if (const auto *NS = dyn_cast<DINamespace>(S))
      emitOpStringIfNew(NS->getName(), MAI);
  }

  for (const DILocation *DL : UniqueDebugLocations)
    emitAndCacheScopePathOpStringReg(DL->getScope(), MAI);

  CachedEmptyStringReg = emitOpStringIfNew("", MAI);

#ifndef NDEBUG
  NonSemanticOpStringsSectionEmitted = true;
#endif
}

void SPIRVNonSemanticDebugHandler::emitDebugFunctionDefinition(
    MCRegister DebugFunctionReg, MCRegister OpFunctionReg,
    SPIRV::ModuleAnalysisInfo &MAI) {
  assert(DebugFunctionReg.isValid() && OpFunctionReg.isValid() &&
         "DebugFunctionDefinition operands must be valid");
  MCRegister VoidTypeReg = getOrEmitOpTypeVoidReg(MAI);
  MCRegister ExtInstSetReg = MAI.getExtInstSetReg(NSSet);
  emitExtInst(SPIRV::NonSemanticExtInst::DebugFunctionDefinition, VoidTypeReg,
              ExtInstSetReg, {DebugFunctionReg, OpFunctionReg}, MAI);
}

void SPIRVNonSemanticDebugHandler::resetPerFunctionDebugState() {
  CurrentMF = nullptr;
  LastFunctionOpVariable = nullptr;
  DebugFunctionDefinitionEmitted = false;
  LastLineMI = nullptr;
}

void SPIRVNonSemanticDebugHandler::preparePerFunctionDebug(
    const MachineFunction *MF) {
  resetPerFunctionDebugState();
  if (!GlobalNSDIEnabled || !CurrentMAI)
    return;

  CurrentMF = MF;

  if (MF->getFunction()
          .getFnAttribute(SPIRV_BACKEND_SERVICE_FUN_NAME)
          .isValid())
    return;

  const DISubprogram *SP = MF->getFunction().getSubprogram();
  if (!SP || !SP->isDefinition())
    return;

  // DebugFunctionDefinition is emitted after the last function-level
  // OpVariable. If there are none, it is emitted after the entry OpLabel.
  LastFunctionOpVariable =
      findLastFunctionOpVariableDeclaration(*MF, *CurrentMAI);
}

void SPIRVNonSemanticDebugHandler::tryEmitDebugFunctionDefinition(
    SPIRV::ModuleAnalysisInfo &MAI) {
  if (DebugFunctionDefinitionEmitted || !GlobalNSDIEnabled)
    return;

  assert(CurrentMF && "no current MachineFunction");
  const Function &F = CurrentMF->getFunction();
  const DISubprogram *SP = F.getSubprogram();
  if (!SP || !SP->isDefinition())
    return;

  auto DFIt = DebugScopeRegs.find(SP);
  if (DFIt == DebugScopeRegs.end())
    return;

  MCRegister OpFunctionReg = MAI.getGlobalObjReg(&F);
  if (!OpFunctionReg.isValid())
    return;

  emitDebugFunctionDefinition(DFIt->second, OpFunctionReg, MAI);
  DebugFunctionDefinitionEmitted = true;
}

void SPIRVNonSemanticDebugHandler::beginFunctionImpl(
    const MachineFunction *MF) {
  preparePerFunctionDebug(MF);
}

void SPIRVNonSemanticDebugHandler::endFunctionImpl(const MachineFunction *MF) {
  (void)MF;
  resetPerFunctionDebugState();
}

void SPIRVNonSemanticDebugHandler::beginInstruction(const MachineInstr *MI) {
  assert(CurMI == nullptr && "CurMI must be null");
  CurMI = MI;

  if (!DebugFunctionDefinitionEmitted)
    return;
  emitDebugLineForInstruction(MI);
}

static bool isMergeInstruction(unsigned Opcode) {
  return Opcode == SPIRV::OpSelectionMerge || Opcode == SPIRV::OpLoopMerge ||
         Opcode == SPIRV::OpLoopControlINTEL;
}

static bool isDebugLineTarget(const MachineInstr *MI,
                              SPIRV::ModuleAnalysisInfo &MAI) {
  if (MAI.getSkipEmission(MI))
    return false;
  switch (MI->getOpcode()) {
  case SPIRV::OpFunction:
  case SPIRV::OpFunctionParameter:
  case SPIRV::OpFunctionEnd:
  case SPIRV::OpLabel:
  case SPIRV::OpPhi:
    return false;
  default:
    return true;
  }
}

static const MachineInstr *
findAdjacentEmittedInstruction(const MachineInstr *MI,
                               SPIRV::ModuleAnalysisInfo &MAI, bool Forward) {
  for (const MachineInstr *Adj = Forward ? MI->getNextNode()
                                         : MI->getPrevNode();
       Adj; Adj = Forward ? Adj->getNextNode() : Adj->getPrevNode()) {
    if (MAI.getSkipEmission(Adj))
      continue;
    return Adj;
  }
  return nullptr;
}

void SPIRVNonSemanticDebugHandler::emitDebugLineForInstruction(
    const MachineInstr *MI) {
  assert(DebugFunctionDefinitionEmitted &&
         "DebugFunctionDefinition must be emitted");
  assert(CurrentMAI && "CurrentMAI must be set");

  SPIRV::ModuleAnalysisInfo &MAI = *CurrentMAI;

  // Structural opcodes don't require a DebugLine, other opcodes might have
  // already been emitted in the module scope.
  if (!isDebugLineTarget(MI, MAI))
    return;

  // DebugLine can be emitted before a merge instruction, but not after it
  // (nothing may sit between the merge and its terminator). We can use either
  // the merge's or the terminator's debug info; we emit the terminator's one.
  const MachineInstr *Prev = findAdjacentEmittedInstruction(MI, MAI, false);
  if (Prev && isMergeInstruction(Prev->getOpcode()))
    return;

  if (isMergeInstruction(MI->getOpcode())) {
    // Use the terminator's debug info; when we reach it later, the check
    // above skips it.
    MI = findAdjacentEmittedInstruction(MI, MAI, true);
    assert(MI && "Merge instruction must be followed by a terminator");
  }

  // The range of DebugLine must be reset at each basic block boundary.
  if (LastLineMI && MI->getParent() != LastLineMI->getParent())
    LastLineMI = nullptr;

  MCRegister VoidTypeReg = getOrEmitOpTypeVoidReg(MAI);
  MCRegister ExtInstSetReg = MAI.getExtInstSetReg(NSSet);

  const DILocation *DL = MI->getDebugLoc().get();
  if (!DL) {
    // No location for the current instruction
    if (LastLineMI) {
      // Close the current DebugLine region.
      emitExtInst(SPIRV::NonSemanticExtInst::DebugNoLine, VoidTypeReg,
                  ExtInstSetReg, {}, MAI);
      LastLineMI = nullptr;
    }
    // No DebugLine region to close.
    return;
  }

  // At this point, there is a location for the current instruction.
  // If it matches the last emitted DebugLine, no new DebugLine region is
  // needed. Otherwise, emit a new DebugLine region and update LastLineMI.

  MCRegister FileStrReg = getCachedScopePathOpStringReg(
      DL->getScope(), /*UseEmptyPathIfNullScope=*/true);
  unsigned Line = DL->getLine();
  unsigned Col = DL->getColumn();

  MCRegister SrcReg = DebugSourceRegByFileStr.lookup(FileStrReg.id());
  MCRegister LineReg = I32ConstantCache.lookup(Line);
  MCRegister ColStartReg = I32ConstantCache.lookup(Col);
  MCRegister ColEndReg = I32ConstantCache.lookup(Col + 1);

  // The elements of each collected DILocation (DebugSource, line/column
  // constants) are pre-emitted from LLVM-IR instruction !dbg attachments and
  // debug-program records; MIR is expected to reuse those same locations (or
  // carry none). A lookup miss means codegen attached a source position whose
  // elements were never pre-emitted, and debug-line emission is skipped.
  if (!SrcReg.isValid() || !LineReg.isValid() || !ColStartReg.isValid() ||
      !ColEndReg.isValid())
    return;

  // Current location matches the last emitted DebugLine region.
  if (LastLineMI && MI->getDebugLoc() == LastLineMI->getDebugLoc())
    return;

  // A new DebugLine region is needed. Emit it and update LastLineMI.
  emitExtInst(SPIRV::NonSemanticExtInst::DebugLine, VoidTypeReg, ExtInstSetReg,
              {SrcReg, LineReg, LineReg, ColStartReg, ColEndReg}, MAI);

  LastLineMI = MI;
}

void SPIRVNonSemanticDebugHandler::endInstruction() {
  const MachineInstr *MI = CurMI;
  CurMI = nullptr;

  if (!MI || !GlobalNSDIEnabled || DebugFunctionDefinitionEmitted || !CurrentMF)
    return;

  if (MI != LastFunctionOpVariable)
    return;

  // If this is the last function-level OpVariable, emit the
  // DebugFunctionDefinition. Otherwise, we had already done it before right
  // after the OpLabel (see notifyEntryLabelEmitted).
  assert(CurrentMAI && "CurrentMAI must be set");
  tryEmitDebugFunctionDefinition(*CurrentMAI);
}

void SPIRVNonSemanticDebugHandler::notifyEntryLabelEmitted(
    const MachineFunction &MF) {
  if (!GlobalNSDIEnabled || DebugFunctionDefinitionEmitted || !CurrentMF)
    return;

  assert(CurrentMF == &MF &&
         "notification does not match the current MachineFunction");

  if (LastFunctionOpVariable)
    return;

  // If there are no function-level OpVariables, emit the
  // DebugFunctionDefinition. Otherwise, DebugFunctionDefinition is emitted
  // after the last OpVariable (see endInstruction).
  tryEmitDebugFunctionDefinition(*CurrentMAI);
}

void SPIRVNonSemanticDebugHandler::emitNonSemanticGlobalDebugInfo(
    SPIRV::ModuleAnalysisInfo &MAI) {
  if (GlobalDIEmitted)
    return;

  GlobalDIEmitted = true;

  if (CompileUnits.empty()) {
    GlobalNSDIEnabled = false;
    return;
  }

  // Retrieve the ext inst set register allocated by prepareModuleOutput().
  MCRegister ExtInstSetReg = MAI.getExtInstSetReg(NSSet);
  if (!ExtInstSetReg.isValid()) {
    GlobalNSDIEnabled = false;
    return;
  }

#ifndef NDEBUG
  assert(NonSemanticOpStringsSectionEmitted &&
         "emitNonSemanticDebugStrings() must run before "
         "emitNonSemanticGlobalDebugInfo()");
#endif

  CurrentMAI = &MAI;

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
    MCRegister FileStrReg = ScopeToPathOpStringReg.lookup(Info.TheCU);
    assert(FileStrReg.isValid() &&
           "CU path OpString must be emitted in emitNonSemanticDebugStrings");
    MCRegister DebugSourceReg = getOrEmitDebugSourceForFileStrReg(
        FileStrReg, VoidTypeReg, ExtInstSetReg, MAI);
    MCRegister CUDbgReg = emitExtInst(
        SPIRV::NonSemanticExtInst::DebugCompilationUnit, VoidTypeReg,
        ExtInstSetReg,
        {DebugInfoVersionReg, DwarfVersionReg, DebugSourceReg, SrcLangReg},
        MAI);
    if (Info.TheCU)
      DebugScopeRegs[Info.TheCU] = CUDbgReg;
  }

  // Zero constant used as the Flags operand in DebugTypeBasic and
  // DebugTypePointer. Cached with other i32 constants.
  MCRegister I32ZeroReg = emitOpConstantI32(0, I32TypeReg, MAI);

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
    DebugScopeRegs[BT] = BTReg;
  }

  // Emit DebugTypeVector for each collected vector type.
  for (const DICompositeType *VT : VectorTypes) {
    if (auto VecReg = emitDebugTypeVector(VT, ExtInstSetReg, MAI))
      DebugScopeRegs[VT] = *VecReg;
  }

  // Emit DebugTypePointer for each referenced pointer type.
  for (const DIDerivedType *PT : PointerTypes) {
    if (auto PtrReg = emitDebugTypePointer(PT, ExtInstSetReg, MAI))
      DebugScopeRegs[PT] = *PtrReg;
  }

  // Emit DebugTypeArray for each collected array type. Placed after the basic,
  // vector, and pointer types so an array over any of them can resolve its
  // element id. An array whose element type was not emitted is skipped.
  for (const DICompositeType *AT : ArrayTypes) {
    if (auto ArrReg = emitDebugTypeArray(AT, ExtInstSetReg, MAI))
      DebugScopeRegs[AT] = *ArrReg;
  }

  // Emit DebugTypeFunction for each distinct DISubroutineType.
  for (const DISubroutineType *ST : SubroutineTypes) {
    if (auto FnTyReg =
            emitDebugTypeFunctionForSubroutineType(ST, ExtInstSetReg, MAI))
      DebugScopeRegs[ST] = *FnTyReg;
  }

  // Emit DebugLexicalBlock for each collected DINamespace, in parent-before-
  // child order. Placed before any DINamespace-scoped entity (typedefs,
  // function declarations, composite types, functions, global variables) so
  // their Parent operand can reference an already-emitted DebugLexicalBlock.
  // DINamespace never chains through a DISubprogram (DINamespace::getScope()
  // returns DIScope, not DILocalScope), so this never depends on
  // DebugScopeRegs.
  for (const DIScope *S :
       make_filter_range(LexicalBlocks, IsaPred<DINamespace>)) {
    if (auto LBReg = emitDebugLexicalBlock(S, VoidTypeReg, I32TypeReg,
                                           ExtInstSetReg, MAI))
      DebugScopeRegs[S] = *LBReg;
  }

  // Emit DebugTypedef for each typedef. Placed after the other type loops so a
  // typedef can resolve its underlying type. A typedef whose base type is not
  // emitted is skipped. A typedef whose base is another typedef emitted later
  // in this same pass is also skipped, the emission-order gap tracked in
  // https://github.com/llvm/llvm-project/issues/211850.
  for (const DIDerivedType *TD : TypedefTypes) {
    if (auto TDReg =
            emitDebugTypedef(TD, VoidTypeReg, I32TypeReg, ExtInstSetReg, MAI))
      DebugScopeRegs[TD] = *TDReg;
  }

  // Emit DebugFunctionDeclaration for DISubprogram declarations.
  for (const DISubprogram *SP : SubprogramDeclarations) {
    if (auto DeclReg = emitDebugFunctionDeclaration(SP, VoidTypeReg, I32TypeReg,
                                                    ExtInstSetReg, MAI))
      DebugScopeRegs[SP] = *DeclReg;
  }

  // Emit DebugTypeMember and DebugTypeComposite for each struct, class, or
  // union. Each member is emitted before the composite that lists it, so the
  // Members operand references already-defined ids. A member whose type is not
  // in DebugScopeRegs is skipped.
  for (const DICompositeType *CT : CompositeTypes) {
    SmallVector<MCRegister> MemberRegs;
    for (const DINode *Element : CT->getElements()) {
      const auto *M = dyn_cast<DIDerivedType>(Element);
      if (!M || M->getTag() != dwarf::DW_TAG_member)
        continue;
      if (auto MemberReg = emitDebugTypeMember(M, VoidTypeReg, I32TypeReg,
                                               ExtInstSetReg, MAI))
        MemberRegs.push_back(*MemberReg);
    }
    if (auto CompReg = emitDebugTypeComposite(CT, MemberRegs, VoidTypeReg,
                                              I32TypeReg, ExtInstSetReg, MAI))
      DebugScopeRegs[CT] = *CompReg;
  }

  // Emit DebugFunction for DISubprogram definitions.
  for (const DISubprogram *SP : SubprogramDefinitions) {
    if (auto FnReg =
            emitDebugFunction(SP, VoidTypeReg, I32TypeReg, ExtInstSetReg, MAI))
      DebugScopeRegs[SP] = *FnReg;
  }

  // Emit DebugLexicalBlock for each collected DILexicalBlock, in parent-
  // before-child order. Placed after DebugFunction so a block directly
  // enclosed by a function (the common case) can resolve its Parent operand;
  // DINamespace entries were already emitted above.
  for (const DIScope *S :
       make_filter_range(LexicalBlocks, IsaPred<DILexicalBlock>)) {
    if (auto LBReg = emitDebugLexicalBlock(S, VoidTypeReg, I32TypeReg,
                                           ExtInstSetReg, MAI))
      DebugScopeRegs[S] = *LBReg;
  }

  // Emit DebugGlobalVariable for each collected DIGlobalVariable.
  for (const auto &[GV, Info] : GlobalVariableDebugInfoMap)
    emitDebugGlobalVariable(GV, Info, VoidTypeReg, I32TypeReg, ExtInstSetReg,
                            MAI);

  for (const DILocation *DL : UniqueDebugLocations) {
    emitOpConstantI32(DL->getLine(), I32TypeReg, MAI);
    emitOpConstantI32(DL->getColumn(), I32TypeReg, MAI);
    emitOpConstantI32(DL->getColumn() + 1, I32TypeReg, MAI);
    MCRegister FileStrReg =
        getCachedScopePathOpStringReg(DL->getScope(),
                                      /*UseEmptyPathIfNullScope=*/true);
    getOrEmitDebugSourceForFileStrReg(FileStrReg, VoidTypeReg, ExtInstSetReg,
                                      MAI);
  }

  GlobalNSDIEnabled = true;
}

SmallString<128>
SPIRVNonSemanticDebugHandler::getDebugFullPath(const DIScope *Scope) const {
  SmallString<128> Out;
  if (!Scope)
    return Out;
  StringRef Filename = Scope->getFilename();
  const auto Style = sys::path::Style::native;
  if (sys::path::is_absolute(Filename, Style))
    Out.assign(Filename.begin(), Filename.end());
  else {
    StringRef Dir = Scope->getDirectory();
    Out.assign(Dir.begin(), Dir.end());
    sys::path::append(Out, Style, Filename);
  }
  return Out;
}

MCRegister SPIRVNonSemanticDebugHandler::getOrEmitDebugSourceForFileStrReg(
    MCRegister FileStrReg, MCRegister VoidTypeReg, MCRegister ExtInstSetReg,
    SPIRV::ModuleAnalysisInfo &MAI) {
  const unsigned Key = FileStrReg.id();
  auto It = DebugSourceRegByFileStr.find(Key);
  if (It != DebugSourceRegByFileStr.end())
    return It->second;

  MCRegister DS = emitExtInst(SPIRV::NonSemanticExtInst::DebugSource,
                              VoidTypeReg, ExtInstSetReg, {FileStrReg}, MAI);
  DebugSourceRegByFileStr[Key] = DS;
  return DS;
}
