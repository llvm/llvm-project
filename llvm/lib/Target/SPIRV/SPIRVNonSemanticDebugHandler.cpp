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
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Path.h"

using namespace llvm;

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

  // Collect basic and pointer types referenced by debug variable records.
  for (const auto &F : *M) {
    for (const auto &BB : F) {
      for (const auto &I : BB) {
        for (const DbgVariableRecord &DVR :
             filterDbgVars(I.getDbgRecordRange())) {
          const DIType *Ty = DVR.getVariable()->getType();
          if (const auto *BT = dyn_cast<DIBasicType>(Ty)) {
            BasicTypes.insert(BT);
          } else if (const auto *DT = dyn_cast<DIDerivedType>(Ty)) {
            if (DT->getTag() == dwarf::DW_TAG_pointer_type) {
              PointerTypes.insert(DT);
              if (const auto *BT =
                      dyn_cast_or_null<DIBasicType>(DT->getBaseType()))
                BasicTypes.insert(BT);
            }
          }
        }
      }
    }
  }
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

void SPIRVNonSemanticDebugHandler::emitDebugTypePointer(
    const DIDerivedType *PT, MCRegister VoidTypeReg, MCRegister I32TypeReg,
    MCRegister ExtInstSetReg, MCRegister I32ZeroReg,
    const DenseMap<const DIBasicType *, MCRegister> &BasicTypeRegs,
    SPIRV::ModuleAnalysisInfo &MAI) {
  // A DWARF address space is required to determine the SPIR-V storage class.
  // Skip pointer types that do not carry one.
  if (!PT->getDWARFAddressSpace().has_value())
    return;

  // For SPIR-V targets, Clang sets DwarfAddressSpace to the LLVM IR address
  // space, which addressSpaceToStorageClass expects.
  const auto &ST = static_cast<const SPIRVSubtarget &>(Asm->getSubtargetInfo());
  MCRegister StorageClassReg = emitOpConstantI32(
      addressSpaceToStorageClass(PT->getDWARFAddressSpace().value(), ST),
      I32TypeReg, MAI);

  if (const auto *BaseType = dyn_cast_or_null<DIBasicType>(PT->getBaseType())) {
    auto BTIt = BasicTypeRegs.find(BaseType);
    if (BTIt != BasicTypeRegs.end())
      emitExtInst(SPIRV::NonSemanticExtInst::DebugTypePointer, VoidTypeReg,
                  ExtInstSetReg, {BTIt->second, StorageClassReg, I32ZeroReg},
                  MAI);
  } else {
    // Void pointer: use DebugInfoNone for the base type. Note that
    // spirv-val currently rejects DebugInfoNone as the base type of
    // DebugTypePointer; see issue #109287 and the DISABLED spirv-val run
    // in debug-type-pointer.ll.
    MCRegister NoneReg = emitExtInst(SPIRV::NonSemanticExtInst::DebugInfoNone,
                                     VoidTypeReg, ExtInstSetReg, {}, MAI);
    emitExtInst(SPIRV::NonSemanticExtInst::DebugTypePointer, VoidTypeReg,
                ExtInstSetReg, {NoneReg, StorageClassReg, I32ZeroReg}, MAI);
  }
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
    FileStringRegs.push_back(emitOpString(Info.FilePath, MAI));

  for (const DIBasicType *BT : BasicTypes)
    BasicTypeNameRegs.push_back(emitOpString(BT->getName(), MAI));
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

  MCRegister VoidTypeReg = findOrEmitOpTypeVoid(MAI);
  MCRegister I32TypeReg = findOrEmitOpTypeInt32(MAI);

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
  // FileStringRegs was populated by emitNonSemanticDebugStrings() in section 7.
  assert(FileStringRegs.size() == CompileUnits.size() &&
         "FileStringRegs must be populated by emitNonSemanticDebugStrings()");
  for (auto [Info, FileStrReg, SrcLangReg] :
       llvm::zip(CompileUnits, FileStringRegs, SrcLangRegs)) {
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

  // Maps each DIBasicType to its DebugTypeBasic result register for use as
  // operands in DebugTypePointer instructions.
  DenseMap<const DIBasicType *, MCRegister> BasicTypeRegs;

  // BasicTypeNameRegs was populated by emitNonSemanticDebugStrings() in
  // section 7.
  assert(
      BasicTypeNameRegs.size() == BasicTypes.size() &&
      "BasicTypeNameRegs must be populated by emitNonSemanticDebugStrings()");
  unsigned BTIdx = 0;
  for (const DIBasicType *BT : BasicTypes) {
    MCRegister NameReg = BasicTypeNameRegs[BTIdx++];
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
    BasicTypeRegs[BT] = BTReg;
  }

  // Emit DebugTypePointer for each referenced pointer type.
  for (const DIDerivedType *PT : PointerTypes)
    emitDebugTypePointer(PT, VoidTypeReg, I32TypeReg, ExtInstSetReg, I32ZeroReg,
                         BasicTypeRegs, MAI);
}
