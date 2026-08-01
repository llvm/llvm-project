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
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

static cl::opt<bool> SPVPreserveAuxData(
    "spirv-preserve-auxdata",
    cl::desc("Preserve LLVM attributes and metadata as "
             "NonSemantic.AuxData ExtInst annotations (requires "
             "SPV_KHR_non_semantic_info)"),
    cl::Optional, cl::Hidden, cl::init(false));

namespace {
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

static bool wasAvailableExternally(const GlobalObject *GO) {
  if (const auto *F = dyn_cast<Function>(GO))
    return F->hasFnAttribute(SPIRV_WAS_AVAILABLE_EXTERNALLY_ATTR);
  return cast<GlobalVariable>(GO)->getAttributes().hasAttribute(
      SPIRV_WAS_AVAILABLE_EXTERNALLY_ATTR);
}

SPIRVAuxDataHandler::SPIRVAuxDataHandler(AsmPrinter &AP, const Module &M)
    : Asm(AP), Mod(M) {
  for (const GlobalObject &GO : M.global_objects())
    if (wasAvailableExternally(&GO))
      LinkagePreservedGOs.push_back(&GO);
}

bool SPIRVAuxDataHandler::hasWork() const { return SPVPreserveAuxData; }

void SPIRVAuxDataHandler::prepareModuleOutput(const SPIRVSubtarget &ST,
                                              SPIRV::ModuleAnalysisInfo &MAI) {
  if (!hasWork())
    return;
  if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_non_semantic_info)) {
    if (SPVPreserveAuxData)
      report_fatal_error("-spirv-preserve-auxdata requires the "
                         "SPV_KHR_non_semantic_info extension to be enabled.");
    return;
  }
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

void SPIRVAuxDataHandler::collectAttributesFor(const GlobalObject *GO,
                                               SPIRV::ModuleAnalysisInfo &MAI) {
  AuxDataOpcode Opcode = isa<Function>(GO) ? FunctionAttributeOpcode
                                           : GlobalVariableAttributeOpcode;
  for (const Attribute &A : getGOAttrs(GO)) {
    if (A.isStringAttribute() &&
        A.getKindAsString() == SPIRV_WAS_AVAILABLE_EXTERNALLY_ATTR)
      continue;
    ExtInstRecord Rec;
    Rec.Opcode = Opcode;
    Rec.Target = GO;
    if (A.isStringAttribute()) {
      Rec.Operands.push_back({getOrEmitString(A.getKindAsString(), MAI)});
      StringRef Val = A.getValueAsString();
      if (!Val.empty())
        Rec.Operands.push_back({getOrEmitString(Val, MAI)});
    } else {
      Rec.Operands.push_back(
          {getOrEmitString(StringPool.save(A.getAsString()), MAI)});
    }
    PendingRecords.push_back(std::move(Rec));
  }
}

void SPIRVAuxDataHandler::collectMetadataFor(const GlobalObject *GO,
                                             ArrayRef<StringRef> MDNames,
                                             SPIRV::ModuleAnalysisInfo &MAI) {
  SmallVector<std::pair<unsigned, MDNode *>> AllMD;
  GO->getAllMetadata(AllMD);
  if (AllMD.empty())
    return;
  AuxDataOpcode Opcode =
      isa<Function>(GO) ? FunctionMetadataOpcode : GlobalVariableMetadataOpcode;
  // MDString operands become OpStrings; ValueAsMetadata constants (e.g.
  // !{i32 5}) become OpConstants emitted at section 10. Any other operand
  // kind would need full value translation, so skip the whole node.
  auto CollectOperands =
      [&](MDNode *MD) -> std::optional<SmallVector<Operand, 4>> {
    SmallVector<Operand, 4> Out;
    for (const MDOperand &MdOp : MD->operands()) {
      Metadata *Md = MdOp.get();
      if (auto *MDStr = dyn_cast_or_null<MDString>(Md)) {
        Out.push_back({getOrEmitString(MDStr->getString(), MAI)});
      } else if (auto *VAM = dyn_cast_or_null<ValueAsMetadata>(Md)) {
        auto *C = dyn_cast<Constant>(VAM->getValue());
        if (!C || !(isa<ConstantInt>(C) || isa<ConstantFP>(C)))
          return std::nullopt;
        Out.push_back({MCRegister(), C});
      } else {
        return std::nullopt;
      }
    }
    return Out;
  };
  for (const auto &MD : AllMD) {
    if (MD.first == LLVMContext::MD_dbg)
      continue;
    StringRef MDName = MDNames[MD.first];
    if (MDName == "spirv.Decorations" || MDName == "spirv.ParameterDecorations")
      continue;
    auto Operands = CollectOperands(MD.second);
    if (!Operands)
      continue;
    ExtInstRecord Rec;
    Rec.Opcode = Opcode;
    Rec.Target = GO;
    Rec.Operands.push_back({getOrEmitString(MDName, MAI)});
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
    collectAttributesFor(&GO, MAI);
    collectMetadataFor(&GO, MDNames, MAI);
  }
}

void SPIRVAuxDataHandler::emitAuxData(SPIRV::ModuleAnalysisInfo &MAI) {
  MCRegister ExtSetReg = MAI.getExtInstSetReg(NonSemanticAuxDataSet);
  if (!ExtSetReg.isValid())
    return;

  MCRegister VoidTypeReg = findOrEmitOpTypeVoid(MAI);

  for (const ExtInstRecord &Rec : PendingRecords) {
    MCRegister TargetReg = MAI.getGlobalObjReg(Rec.Target);
    if (!TargetReg.isValid())
      continue;
    SmallVector<MCRegister, 5> Operands;
    Operands.push_back(TargetReg);
    for (const Operand &Op : Rec.Operands)
      Operands.push_back(Op.Const ? emitConstant(Op.Const, MAI) : Op.Reg);
    emitAuxDataExtInst(Rec.Opcode, VoidTypeReg, ExtSetReg, Operands, MAI);
  }

  if (LinkagePreservedGOs.empty())
    return;

  MCRegister UInt32TypeReg = findOrEmitOpTypeUInt32(MAI);
  MCRegister AEConstReg;
  for (const GlobalObject *GO : LinkagePreservedGOs) {
    MCRegister TargetReg = MAI.getGlobalObjReg(GO);
    if (!TargetReg.isValid())
      continue;
    if (!AEConstReg.isValid())
      AEConstReg =
          emitOpConstantUInt32(AvailableExternally, UInt32TypeReg, MAI);
    emitAuxDataExtInst(LinkageOpcode, VoidTypeReg, ExtSetReg,
                       {TargetReg, AEConstReg}, MAI);
  }
}

void SPIRVAuxDataHandler::emitAuxDataExtInst(AuxDataOpcode Opcode,
                                             MCRegister VoidTypeReg,
                                             MCRegister ExtSetReg,
                                             ArrayRef<MCRegister> Operands,
                                             SPIRV::ModuleAnalysisInfo &MAI) {
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpExtInst);
  Inst.addOperand(MCOperand::createReg(MAI.getNextIDRegister()));
  Inst.addOperand(MCOperand::createReg(VoidTypeReg));
  Inst.addOperand(MCOperand::createReg(ExtSetReg));
  Inst.addOperand(MCOperand::createImm(Opcode));
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
SPIRVAuxDataHandler::findOrEmitOpTypeInt(unsigned BitWidth,
                                         SPIRV::ModuleAnalysisInfo &MAI) {
  // SPIR-V OpTypeInt: <width>, <signedness>. Signedness 0 = unsigned, 1 =
  // signed; we always emit unsigned.
  constexpr int64_t UnsignedSignedness = 0;
  for (const MachineInstr *MI : MAI.getMSInstrs(SPIRV::MB_TypeConstVars))
    if (MI->getOpcode() == SPIRV::OpTypeInt &&
        MI->getOperand(1).getImm() == static_cast<int64_t>(BitWidth) &&
        MI->getOperand(2).getImm() == UnsignedSignedness)
      return MAI.getRegisterAlias(MI->getMF(), MI->getOperand(0).getReg());
  MCRegister Reg = MAI.getNextIDRegister();
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpTypeInt);
  Inst.addOperand(MCOperand::createReg(Reg));
  Inst.addOperand(MCOperand::createImm(BitWidth));
  Inst.addOperand(MCOperand::createImm(UnsignedSignedness));
  emitMCInst(Inst);
  return Reg;
}

MCRegister
SPIRVAuxDataHandler::findOrEmitOpTypeUInt32(SPIRV::ModuleAnalysisInfo &MAI) {
  return findOrEmitOpTypeInt(32, MAI);
}

MCRegister
SPIRVAuxDataHandler::findOrEmitOpTypeFloat(unsigned BitWidth,
                                           SPIRV::ModuleAnalysisInfo &MAI) {
  for (const MachineInstr *MI : MAI.getMSInstrs(SPIRV::MB_TypeConstVars))
    if (MI->getOpcode() == SPIRV::OpTypeFloat &&
        MI->getOperand(1).getImm() == static_cast<int64_t>(BitWidth))
      return MAI.getRegisterAlias(MI->getMF(), MI->getOperand(0).getReg());
  MCRegister Reg = MAI.getNextIDRegister();
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpTypeFloat);
  Inst.addOperand(MCOperand::createReg(Reg));
  Inst.addOperand(MCOperand::createImm(BitWidth));
  emitMCInst(Inst);
  return Reg;
}

MCRegister SPIRVAuxDataHandler::emitConstant(const Constant *C,
                                             SPIRV::ModuleAnalysisInfo &MAI) {
  auto [It, Inserted] = ConstantRegs.try_emplace(C);
  if (!Inserted)
    return It->second;

  APInt Bits;
  unsigned Opcode;
  MCRegister TypeReg;
  if (const auto *CI = dyn_cast<ConstantInt>(C)) {
    Bits = CI->getValue();
    Opcode = SPIRV::OpConstantI;
    TypeReg = findOrEmitOpTypeInt(Bits.getBitWidth(), MAI);
  } else {
    const auto *CF = cast<ConstantFP>(C);
    Bits = CF->getValueAPF().bitcastToAPInt();
    Opcode = SPIRV::OpConstantF;
    TypeReg = findOrEmitOpTypeFloat(Bits.getBitWidth(), MAI);
  }

  MCRegister Reg = MAI.getNextIDRegister();
  It->second = Reg;
  MCInst Inst;
  Inst.setOpcode(Opcode);
  Inst.addOperand(MCOperand::createReg(Reg));
  Inst.addOperand(MCOperand::createReg(TypeReg));
  // SPIR-V encodes the literal as ceil(width/32) little-endian 32-bit words.
  unsigned NumWords = divideCeil(Bits.getBitWidth(), 32);
  for (unsigned I = 0; I < NumWords; ++I)
    Inst.addOperand(MCOperand::createImm(Bits.extractBitsAsZExtValue(
        std::min(32u, Bits.getBitWidth() - I * 32), I * 32)));
  // The asm printer needs this hint to render an f16 literal correctly.
  if (Opcode == SPIRV::OpConstantF && Bits.getBitWidth() == 16)
    Inst.setFlags(SPIRV::INST_PRINTER_WIDTH16);
  emitMCInst(Inst);
  return Reg;
}

MCRegister SPIRVAuxDataHandler::emitOpConstantUInt32(
    uint32_t Value, MCRegister UInt32TypeReg, SPIRV::ModuleAnalysisInfo &MAI) {
  MCRegister Reg = MAI.getNextIDRegister();
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpConstantI);
  Inst.addOperand(MCOperand::createReg(Reg));
  Inst.addOperand(MCOperand::createReg(UInt32TypeReg));
  Inst.addOperand(MCOperand::createImm(static_cast<int64_t>(Value)));
  emitMCInst(Inst);
  return Reg;
}
