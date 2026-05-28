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
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

namespace {
constexpr int64_t AuxDataLinkage = 4;
constexpr uint32_t LinkageAvailableExternally = 0;
} // namespace

SPIRVAuxDataHandler::SPIRVAuxDataHandler(AsmPrinter &AP, const Module &M)
    : Asm(AP) {
  for (const Function &F : M)
    if (F.hasFnAttribute(SPIRV_WAS_AVAILABLE_EXTERNALLY_ATTR))
      LinkagePreservedFns.push_back(&F);
}

void SPIRVAuxDataHandler::prepareModuleOutput(const SPIRVSubtarget &ST,
                                              SPIRV::ModuleAnalysisInfo &MAI) {
  if (LinkagePreservedFns.empty())
    return;
  if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_non_semantic_info))
    return;
  MAI.Reqs.addExtension(SPIRV::Extension::SPV_KHR_non_semantic_info);
  constexpr unsigned AuxSet =
      static_cast<unsigned>(SPIRV::InstructionSet::NonSemantic_AuxData);
  if (!MAI.ExtInstSetMap.count(AuxSet))
    MAI.ExtInstSetMap[AuxSet] = MAI.getNextIDRegister();
}

void SPIRVAuxDataHandler::emitAuxData(SPIRV::ModuleAnalysisInfo &MAI) {
  if (LinkagePreservedFns.empty())
    return;
  constexpr unsigned AuxSet =
      static_cast<unsigned>(SPIRV::InstructionSet::NonSemantic_AuxData);
  MCRegister ExtSetReg = MAI.getExtInstSetReg(AuxSet);
  if (!ExtSetReg.isValid())
    return;

  MCRegister VoidTypeReg = findOrEmitOpTypeVoid(MAI);
  MCRegister I32TypeReg = findOrEmitOpTypeInt32(MAI);
  // Share one OpConstant across all AE functions.
  MCRegister ZeroReg;

  for (const Function *F : LinkagePreservedFns) {
    MCRegister FnReg = MAI.getGlobalObjReg(F);
    if (!FnReg.isValid())
      continue;
    if (!ZeroReg.isValid())
      ZeroReg = emitOpConstantI32(LinkageAvailableExternally, I32TypeReg, MAI);
    MCRegister ValReg = ZeroReg;
    MCInst Inst;
    Inst.setOpcode(SPIRV::OpExtInst);
    Inst.addOperand(MCOperand::createReg(MAI.getNextIDRegister()));
    Inst.addOperand(MCOperand::createReg(VoidTypeReg));
    Inst.addOperand(MCOperand::createReg(ExtSetReg));
    Inst.addOperand(MCOperand::createImm(AuxDataLinkage));
    Inst.addOperand(MCOperand::createReg(FnReg));
    Inst.addOperand(MCOperand::createReg(ValReg));
    emitMCInst(Inst);
  }
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
  for (const MachineInstr *MI : MAI.getMSInstrs(SPIRV::MB_TypeConstVars))
    if (MI->getOpcode() == SPIRV::OpTypeInt &&
        MI->getOperand(1).getImm() == 32 && MI->getOperand(2).getImm() == 0)
      return MAI.getRegisterAlias(MI->getMF(), MI->getOperand(0).getReg());
  MCRegister Reg = MAI.getNextIDRegister();
  MCInst Inst;
  Inst.setOpcode(SPIRV::OpTypeInt);
  Inst.addOperand(MCOperand::createReg(Reg));
  Inst.addOperand(MCOperand::createImm(32));
  Inst.addOperand(MCOperand::createImm(0));
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
