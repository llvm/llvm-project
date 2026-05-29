//===-- SPIRVAuxDataHandler.cpp - NonSemantic.AuxData emitter -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVAuxDataHandler.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<bool> SPVPreserveAuxData(
    "spirv-preserve-auxdata",
    cl::desc("Preserve LLVM attributes and metadata as "
             "NonSemantic.AuxData ExtInst annotations (requires "
             "SPV_KHR_non_semantic_info)"),
    cl::Optional, cl::Hidden, cl::init(false));

namespace {
// Khronos NonSemantic.AuxData opcodes.
enum AuxDataOpcode : uint32_t {
  FunctionMetadata = 0,
  FunctionAttribute = 1,
  GlobalVariableMetadata = 2,
  GlobalVariableAttribute = 3,
  Linkage = 4,
};

enum AuxDataLinkageType : uint32_t {
  AvailableExternally = 0,
};

constexpr unsigned NonSemanticAuxDataSet =
    static_cast<unsigned>(SPIRV::InstructionSet::NonSemantic_AuxData);

AttributeSet getGOAttrs(const GlobalObject *GO) {
  if (const auto *F = dyn_cast<Function>(GO))
    return F->getAttributes().getFnAttrs();
  return cast<GlobalVariable>(GO)->getAttributes();
}
} // namespace

SPIRVAuxDataHandler::SPIRVAuxDataHandler(AsmPrinter &AP, const Module &M)
    : Asm(AP), Mod(M) {
  for (const Function &F : M)
    if (F.hasFnAttribute(SPIRV_WAS_AVAILABLE_EXTERNALLY_ATTR))
      LinkagePreservedFns.push_back(&F);
}

bool SPIRVAuxDataHandler::hasWork() const {
  return !LinkagePreservedFns.empty() || SPVPreserveAuxData;
}

void SPIRVAuxDataHandler::prepareModuleOutput(const SPIRVSubtarget &ST,
                                              SPIRV::ModuleAnalysisInfo &MAI) {
  if (!hasWork())
    return;
  if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_non_semantic_info))
    return;
  MAI.Reqs.addExtension(SPIRV::Extension::SPV_KHR_non_semantic_info);
  if (!MAI.ExtInstSetMap.count(NonSemanticAuxDataSet))
    MAI.ExtInstSetMap[NonSemanticAuxDataSet] = MAI.getNextIDRegister();
}

MCRegister
SPIRVAuxDataHandler::getOrEmitString(StringRef S,
                                     SPIRV::ModuleAnalysisInfo &MAI) {
  auto [It, Inserted] = StringRegs.try_emplace(S);
  if (!Inserted)
    return It->second;
  MCRegister Reg = MAI.getNextIDRegister();
  It->second = Reg;
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpString);
  Inst.addOperand(MCOperand::createReg(Reg));
  addStringImm(S, Inst);
  emitMCInst(Inst);
  return Reg;
}

void SPIRVAuxDataHandler::collectAttributesFor(
    const GlobalObject *GO, function_ref<MCRegister()> GetNameReg,
    SPIRV::ModuleAnalysisInfo &MAI) {
  uint32_t Opcode =
      isa<Function>(GO) ? FunctionAttribute : GlobalVariableAttribute;
  for (const Attribute &A : getGOAttrs(GO)) {
    if (A.isStringAttribute() &&
        A.getKindAsString() == SPIRV_WAS_AVAILABLE_EXTERNALLY_ATTR)
      continue;
    ExtInstRecord Rec;
    Rec.Opcode = Opcode;
    Rec.Operands.push_back(GetNameReg());
    if (A.isStringAttribute()) {
      Rec.Operands.push_back(getOrEmitString(A.getKindAsString(), MAI));
      StringRef Val = A.getValueAsString();
      if (!Val.empty())
        Rec.Operands.push_back(getOrEmitString(Val, MAI));
    } else {
      Rec.Operands.push_back(
          getOrEmitString(StringPool.save(A.getAsString()), MAI));
    }
    PendingRecords.push_back(std::move(Rec));
  }
}

void SPIRVAuxDataHandler::collectMetadataFor(
    const GlobalObject *GO, function_ref<MCRegister()> GetNameReg,
    ArrayRef<StringRef> MDNames, SPIRV::ModuleAnalysisInfo &MAI) {
  SmallVector<std::pair<unsigned, MDNode *>> AllMD;
  GO->getAllMetadata(AllMD);
  if (AllMD.empty())
    return;
  uint32_t Opcode =
      isa<Function>(GO) ? FunctionMetadata : GlobalVariableMetadata;
  // Skip non-MDString operands: emitting them would require a full value
  // translation we can't safely drive from here.
  auto CollectStrings = [&](MDNode *MD) -> std::optional<SmallVector<MCRegister, 4>> {
    SmallVector<MCRegister, 4> Out;
    for (const MDOperand &MdOp : MD->operands()) {
      auto *MDStr = dyn_cast_or_null<MDString>(MdOp.get());
      if (!MDStr)
        return std::nullopt;
      Out.push_back(getOrEmitString(MDStr->getString(), MAI));
    }
    return Out;
  };
  for (const auto &MD : AllMD) {
    if (MD.first == LLVMContext::MD_dbg)
      continue;
    StringRef MDName = MDNames[MD.first];
    if (MDName == "spirv.Decorations" ||
        MDName == "spirv.ParameterDecorations")
      continue;
    auto Operands = CollectStrings(MD.second);
    if (!Operands)
      continue;
    ExtInstRecord Rec;
    Rec.Opcode = Opcode;
    Rec.Operands.push_back(GetNameReg());
    Rec.Operands.push_back(getOrEmitString(MDName, MAI));
    Rec.Operands.append(Operands->begin(), Operands->end());
    PendingRecords.push_back(std::move(Rec));
  }
}

void SPIRVAuxDataHandler::emitAuxDataStrings(SPIRV::ModuleAnalysisInfo &MAI) {
  if (!SPVPreserveAuxData)
    return;
  if (!MAI.getExtInstSetReg(NonSemanticAuxDataSet).isValid())
    return;
  SmallVector<StringRef> MDNames;
  Mod.getContext().getMDKindNames(MDNames);
  for (const GlobalObject &GO : Mod.global_objects()) {
    if (GO.isDeclaration())
      continue;
    // Defer the name OpString until the first record actually fires.
    MCRegister NameReg;
    auto GetNameReg = [&]() {
      if (!NameReg.isValid())
        NameReg = getOrEmitString(GO.getName(), MAI);
      return NameReg;
    };
    collectAttributesFor(&GO, GetNameReg, MAI);
    collectMetadataFor(&GO, GetNameReg, MDNames, MAI);
  }
}

void SPIRVAuxDataHandler::emitAuxData(SPIRV::ModuleAnalysisInfo &MAI) {
  MCRegister ExtSetReg = MAI.getExtInstSetReg(NonSemanticAuxDataSet);
  if (!ExtSetReg.isValid())
    return;

  MCRegister VoidTypeReg = findOrEmitOpTypeVoid(MAI);

  for (const ExtInstRecord &Rec : PendingRecords)
    emitAuxDataExtInst(Rec.Opcode, VoidTypeReg, ExtSetReg, Rec.Operands, MAI);

  if (LinkagePreservedFns.empty())
    return;

  MCRegister I32TypeReg = findOrEmitOpTypeInt32(MAI);
  MCRegister AEConstReg;
  for (const Function *F : LinkagePreservedFns) {
    MCRegister FnReg = MAI.getGlobalObjReg(F);
    if (!FnReg.isValid())
      continue;
    if (!AEConstReg.isValid())
      AEConstReg =
          emitOpConstantI32(AvailableExternally, I32TypeReg, MAI);
    emitAuxDataExtInst(Linkage, VoidTypeReg, ExtSetReg, {FnReg, AEConstReg},
                       MAI);
  }
}

void SPIRVAuxDataHandler::emitAuxDataExtInst(
    uint32_t Opcode, MCRegister VoidTypeReg, MCRegister ExtSetReg,
    ArrayRef<MCRegister> Operands, SPIRV::ModuleAnalysisInfo &MAI) {
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpExtInst);
  Inst.addOperand(MCOperand::createReg(MAI.getNextIDRegister()));
  Inst.addOperand(MCOperand::createReg(VoidTypeReg));
  Inst.addOperand(MCOperand::createReg(ExtSetReg));
  Inst.addOperand(MCOperand::createImm(static_cast<int64_t>(Opcode)));
  for (MCRegister R : Operands)
    Inst.addOperand(MCOperand::createReg(R));
  emitMCInst(Inst);
}

void SPIRVAuxDataHandler::emitMCInst(MCInst &Inst) {
  Asm.OutStreamer->emitInstruction(Inst, Asm.getSubtargetInfo());
}

MCRegister
SPIRVAuxDataHandler::findOrEmitOpTypeVoid(SPIRV::ModuleAnalysisInfo &MAI) {
  for (const MachineInstr *MI : MAI.getMSInstrs(SPIRV::MB_TypeConstVars))
    if (MI->getOpcode() == SPIRV::OpTypeVoid)
      return MAI.getRegisterAlias(MI->getMF(), MI->getOperand(0).getReg());
  MCRegister Reg = MAI.getNextIDRegister();
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpTypeVoid);
  Inst.addOperand(MCOperand::createReg(Reg));
  emitMCInst(Inst);
  return Reg;
}

MCRegister
SPIRVAuxDataHandler::findOrEmitOpTypeInt32(SPIRV::ModuleAnalysisInfo &MAI) {
  constexpr int64_t Int32BitWidth = 32;
  constexpr int64_t UnsignedSignedness = 0;
  for (const MachineInstr *MI : MAI.getMSInstrs(SPIRV::MB_TypeConstVars))
    if (MI->getOpcode() == SPIRV::OpTypeInt &&
        MI->getOperand(1).getImm() == Int32BitWidth &&
        MI->getOperand(2).getImm() == UnsignedSignedness)
      return MAI.getRegisterAlias(MI->getMF(), MI->getOperand(0).getReg());
  MCRegister Reg = MAI.getNextIDRegister();
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpTypeInt);
  Inst.addOperand(MCOperand::createReg(Reg));
  Inst.addOperand(MCOperand::createImm(Int32BitWidth));
  Inst.addOperand(MCOperand::createImm(UnsignedSignedness));
  emitMCInst(Inst);
  return Reg;
}

MCRegister
SPIRVAuxDataHandler::emitOpConstantI32(uint32_t Value, MCRegister I32TypeReg,
                                       SPIRV::ModuleAnalysisInfo &MAI) {
  MCRegister Reg = MAI.getNextIDRegister();
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpConstantI);
  Inst.addOperand(MCOperand::createReg(Reg));
  Inst.addOperand(MCOperand::createReg(I32TypeReg));
  Inst.addOperand(MCOperand::createImm(static_cast<int64_t>(Value)));
  emitMCInst(Inst);
  return Reg;
}
