//===-- ARMInstrInfo.cpp - ARM Instruction Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "ARMConstantPoolValue.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMTargetMachine.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
using namespace llvm;

ARMInstrInfo::ARMInstrInfo(const ARMSubtarget &STI)
  : ARMBaseInstrInfo(STI), RI(*this, STI) {
}

/// getNoopForMachoTarget - Return the noop instruction to use for a noop.
void ARMInstrInfo::getNoopForMachoTarget(MCInst &NopInst) const {
  if (hasNOP()) {
    NopInst.setOpcode(ARM::HINT);
    NopInst.addOperand(MCOperand::CreateImm(0));
    NopInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    NopInst.addOperand(MCOperand::CreateReg(0));
  } else {
    NopInst.setOpcode(ARM::MOVr);
    NopInst.addOperand(MCOperand::CreateReg(ARM::R0));
    NopInst.addOperand(MCOperand::CreateReg(ARM::R0));
    NopInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    NopInst.addOperand(MCOperand::CreateReg(0));
    NopInst.addOperand(MCOperand::CreateReg(0));
  }
}

unsigned ARMInstrInfo::getUnindexedOpcode(unsigned Opc) const {
  switch (Opc) {
  default: break;
  case ARM::LDR_PRE_IMM:
  case ARM::LDR_PRE_REG:
  case ARM::LDR_POST_IMM:
  case ARM::LDR_POST_REG:
    return ARM::LDRi12;
  case ARM::LDRH_PRE:
  case ARM::LDRH_POST:
    return ARM::LDRH;
  case ARM::LDRB_PRE_IMM:
  case ARM::LDRB_PRE_REG:
  case ARM::LDRB_POST_IMM:
  case ARM::LDRB_POST_REG:
    return ARM::LDRBi12;
  case ARM::LDRSH_PRE:
  case ARM::LDRSH_POST:
    return ARM::LDRSH;
  case ARM::LDRSB_PRE:
  case ARM::LDRSB_POST:
    return ARM::LDRSB;
  case ARM::STR_PRE_IMM:
  case ARM::STR_PRE_REG:
  case ARM::STR_POST_IMM:
  case ARM::STR_POST_REG:
    return ARM::STRi12;
  case ARM::STRH_PRE:
  case ARM::STRH_POST:
    return ARM::STRH;
  case ARM::STRB_PRE_IMM:
  case ARM::STRB_PRE_REG:
  case ARM::STRB_POST_IMM:
  case ARM::STRB_POST_REG:
    return ARM::STRBi12;
  }

  return 0;
}

namespace {
  /// ARMCGBR - Create Global Base Reg pass. This initializes the PIC
  /// global base register for ARM ELF.
  struct ARMCGBR : public MachineFunctionPass {
    static char ID;
    ARMCGBR() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
      if (AFI->getGlobalBaseReg() == 0)
        return false;

      const ARMTargetMachine *TM =
        static_cast<const ARMTargetMachine *>(&MF.getTarget());
      if (TM->getRelocationModel() != Reloc::PIC_)
        return false;

      LLVMContext* Context = &MF.getFunction()->getContext();
      GlobalValue *GV = new GlobalVariable(Type::getInt32Ty(*Context), false,
                                           GlobalValue::ExternalLinkage, 0,
                                           "_GLOBAL_OFFSET_TABLE_");
      unsigned Id = AFI->createPICLabelUId();
      ARMConstantPoolValue *CPV = ARMConstantPoolConstant::Create(GV, Id);
      unsigned Align = TM->getDataLayout()->getPrefTypeAlignment(GV->getType());
      unsigned Idx = MF.getConstantPool()->getConstantPoolIndex(CPV, Align);

      MachineBasicBlock &FirstMBB = MF.front();
      MachineBasicBlock::iterator MBBI = FirstMBB.begin();
      DebugLoc DL = FirstMBB.findDebugLoc(MBBI);
      unsigned GlobalBaseReg = AFI->getGlobalBaseReg();
      unsigned Opc = TM->getSubtarget<ARMSubtarget>().isThumb2() ?
                     ARM::t2LDRpci : ARM::LDRcp;
      const TargetInstrInfo &TII = *TM->getInstrInfo();
      MachineInstrBuilder MIB = BuildMI(FirstMBB, MBBI, DL,
                                        TII.get(Opc), GlobalBaseReg)
                                .addConstantPoolIndex(Idx);
      if (Opc == ARM::LDRcp)
        MIB.addImm(0);
      AddDefaultPred(MIB);

      return true;
    }

    virtual const char *getPassName() const {
      return "ARM PIC Global Base Reg Initialization";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

char ARMCGBR::ID = 0;
FunctionPass*
llvm::createARMGlobalBaseRegPass() { return new ARMCGBR(); }
