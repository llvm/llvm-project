//===-- DPUResolveMacroInstrPass.cpp --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DPU.h"
#include "DPUInstrInfo.h"
#include "DPUSubtarget.h"
#include "MCTargetDesc/DPUAsmCondition.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <llvm/CodeGen/MachineInstrBuilder.h>

#define GET_REGINFO_ENUM
#include "DPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#include "DPUGenInstrInfo.inc"

#define DEBUG_TYPE "dpu-resolve-macro-instr"

using namespace llvm;

namespace {
class DPUResolveMacroInstrPass : public MachineFunctionPass {
public:
  static char ID;

  explicit DPUResolveMacroInstrPass(DPUTargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  llvm::StringRef getPassName() const override {
    return "DPU Resolve Macro Instructions";
  }

private:
  const DPUTargetMachine &TM;
};

char DPUResolveMacroInstrPass::ID = 0;
} // namespace

FunctionPass *llvm::createDPUResolveMacroInstrPass(DPUTargetMachine &tm) {
  return new DPUResolveMacroInstrPass(tm);
}

static DPUAsmCondition::Condition
findSelect64SetConditionFor(DPUAsmCondition::Condition cond) {
  switch (cond) {
  default:
    llvm_unreachable("invalid condition");
  case DPUAsmCondition::Condition::SetZero:
  case DPUAsmCondition::Condition::SetEqual:
    return DPUAsmCondition::Condition::SetExtendedZero;
  case DPUAsmCondition::Condition::SetNotZero:
  case DPUAsmCondition::Condition::SetNotEqual:
    return DPUAsmCondition::Condition::SetExtendedNotZero;
  case DPUAsmCondition::Condition::SetGreaterThanSigned:
    return DPUAsmCondition::Condition::SetExtendedGreaterThanSigned;
  case DPUAsmCondition::Condition::SetGreaterOrEqualSigned:
    return DPUAsmCondition::Condition::SetGreaterOrEqualSigned;
  case DPUAsmCondition::Condition::SetLessThanSigned:
    return DPUAsmCondition::Condition::SetLessThanSigned;
  case DPUAsmCondition::Condition::SetLessOrEqualSigned:
    return DPUAsmCondition::Condition::SetExtendedLessOrEqualSigned;
  case DPUAsmCondition::Condition::SetGreaterThanUnsigned:
    return DPUAsmCondition::Condition::SetExtendedGreaterThanUnsigned;
  case DPUAsmCondition::Condition::SetGreaterOrEqualUnsigned:
    return DPUAsmCondition::Condition::SetGreaterOrEqualUnsigned;
  case DPUAsmCondition::Condition::SetLessThanUnsigned:
    return DPUAsmCondition::Condition::SetLessThanUnsigned;
  case DPUAsmCondition::Condition::SetLessOrEqualUnsigned:
    return DPUAsmCondition::Condition::SetExtendedLessOrEqualUnsigned;
  }
}

static unsigned int findJumpOpcodeForCondition(int64_t cond,
                                               bool hasImmediateOperand) {
  switch (cond) {
  default:
    llvm_unreachable("invalid condition");
  case ISD::SETOEQ:
  case ISD::SETUEQ:
  case ISD::SETEQ:
    return hasImmediateOperand ? DPU::JEQrii : DPU::JEQrri;
  case ISD::SETONE:
  case ISD::SETUNE:
  case ISD::SETNE:
    return hasImmediateOperand ? DPU::JNEQrii : DPU::JNEQrri;
  case ISD::SETOGT:
  case ISD::SETGT:
    return hasImmediateOperand ? DPU::JGTSrii : DPU::JGTSrri;
  case ISD::SETOGE:
  case ISD::SETGE:
    return hasImmediateOperand ? DPU::JGESrii : DPU::JGESrri;
  case ISD::SETOLT:
  case ISD::SETLT:
    return hasImmediateOperand ? DPU::JLTSrii : DPU::JLTSrri;
  case ISD::SETOLE:
  case ISD::SETLE:
    return hasImmediateOperand ? DPU::JLESrii : DPU::JLESrri;
  case ISD::SETUGT:
    return hasImmediateOperand ? DPU::JGTUrii : DPU::JGTUrri;
  case ISD::SETUGE:
    return hasImmediateOperand ? DPU::JGEUrii : DPU::JGEUrri;
  case ISD::SETULT:
    return hasImmediateOperand ? DPU::JLTUrii : DPU::JLTUrri;
  case ISD::SETULE:
    return hasImmediateOperand ? DPU::JLEUrii : DPU::JLEUrri;
  }
}

static void resolve64BitImmediateAluInstruction(
    MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBIter,
    const DPUInstrInfo &InstrInfo, unsigned int LsbOpcode,
    unsigned int MsbOpcode) {
  MachineFunction *MF = MBB->getParent();
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();

  unsigned int DestReg = MBBIter->getOperand(0).getReg();
  unsigned int Op1Reg = MBBIter->getOperand(1).getReg();
  int64_t Op2Imm = MBBIter->getOperand(2).getImm();

  unsigned int LSBDestReg = TRI->getSubReg(DestReg, DPU::sub_32bit);
  unsigned int MSBDestReg = TRI->getSubReg(DestReg, DPU::sub_32bit_hi);

  unsigned int LSBDOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit);
  unsigned int MSBDOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit_hi);

  int64_t LSBOp2Imm = Op2Imm & 0xFFFFFFFFl;
  int64_t MSBOp2Imm = (Op2Imm >> 32) & 0xFFFFFFFFl;

  BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(), InstrInfo.get(LsbOpcode),
          LSBDestReg)
      .addReg(LSBDOp1Reg)
      .addImm(LSBOp2Imm);
  BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(), InstrInfo.get(MsbOpcode),
          MSBDestReg)
      .addReg(MSBDOp1Reg)
      .addImm(MSBOp2Imm);
}

static void resolve64BitRegisterAluInstruction(
    MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBIter,
    const DPUInstrInfo &InstrInfo, unsigned int LsbOpcode,
    unsigned int MsbOpcode) {
  MachineFunction *MF = MBB->getParent();
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();

  unsigned int DestReg = MBBIter->getOperand(0).getReg();
  unsigned int Op1Reg = MBBIter->getOperand(1).getReg();
  unsigned int Op2Reg = MBBIter->getOperand(2).getReg();

  unsigned int LSBDestReg = TRI->getSubReg(DestReg, DPU::sub_32bit);
  unsigned int MSBDestReg = TRI->getSubReg(DestReg, DPU::sub_32bit_hi);

  unsigned int LSBDOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit);
  unsigned int MSBDOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit_hi);

  unsigned int LSBOp2Reg = TRI->getSubReg(Op2Reg, DPU::sub_32bit);
  unsigned int MSBOp2Reg = TRI->getSubReg(Op2Reg, DPU::sub_32bit_hi);

  BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(), InstrInfo.get(LsbOpcode),
          LSBDestReg)
      .addReg(LSBDOp1Reg)
      .addReg(LSBOp2Reg);
  BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(), InstrInfo.get(MsbOpcode),
          MSBDestReg)
      .addReg(MSBDOp1Reg)
      .addReg(MSBOp2Reg);
}

static void resolveJeq64(MachineBasicBlock *MBB,
                         MachineBasicBlock::iterator MBBIter,
                         const DPUInstrInfo &InstrInfo) {
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineFunction::iterator I = ++MBB->getIterator();
  MachineFunction *F = MBB->getParent();
  MachineBasicBlock *trueMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *endMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(I, trueMBB);
  F->insert(I, endMBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  endMBB->splice(endMBB->begin(), MBB, std::next(MBBIter), MBB->end());
  endMBB->transferSuccessorsAndUpdatePHIs(MBB);
  // Next, add the true and fallthrough blocks as its successors.
  auto JumpMBB = MBBIter->getOperand(3).getMBB();
  MBB->addSuccessor(trueMBB);
  MBB->addSuccessor(endMBB);
  trueMBB->addSuccessor(JumpMBB);
  trueMBB->addSuccessor(endMBB);

  unsigned int Op1Reg = MBBIter->getOperand(1).getReg();
  unsigned int Op2Reg = MBBIter->getOperand(2).getReg();

  MachineFunction *MF = MBB->getParent();
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  unsigned int LsbOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit);
  unsigned int MsbOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit_hi);
  unsigned int LsbOp2Reg = TRI->getSubReg(Op2Reg, DPU::sub_32bit);
  unsigned int MsbOp2Reg = TRI->getSubReg(Op2Reg, DPU::sub_32bit_hi);

  BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(), InstrInfo.get(DPU::JNEQrri))
      .addReg(LsbOp1Reg)
      .addReg(LsbOp2Reg)
      .addMBB(endMBB);

  BuildMI(trueMBB, MBBIter->getDebugLoc(), InstrInfo.get(DPU::JEQrri))
      .addReg(MsbOp1Reg)
      .addReg(MsbOp2Reg)
      .addMBB(JumpMBB);
}

static void resolveJneq64(MachineBasicBlock *MBB,
                          MachineBasicBlock::iterator MBBIter,
                          const DPUInstrInfo &InstrInfo) {
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineFunction::iterator I = ++MBB->getIterator();
  MachineFunction *F = MBB->getParent();
  MachineBasicBlock *trueMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *endMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(I, trueMBB);
  F->insert(I, endMBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  endMBB->splice(endMBB->begin(), MBB, std::next(MBBIter), MBB->end());
  endMBB->transferSuccessorsAndUpdatePHIs(MBB);
  // Next, add the true and fallthrough blocks as its successors.
  auto JumpMBB = MBBIter->getOperand(3).getMBB();
  MBB->addSuccessor(trueMBB);
  MBB->addSuccessor(JumpMBB);
  trueMBB->addSuccessor(JumpMBB);
  trueMBB->addSuccessor(endMBB);

  unsigned int Op1Reg = MBBIter->getOperand(1).getReg();
  unsigned int Op2Reg = MBBIter->getOperand(2).getReg();

  MachineFunction *MF = MBB->getParent();
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  unsigned int LsbOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit);
  unsigned int MsbOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit_hi);
  unsigned int LsbOp2Reg = TRI->getSubReg(Op2Reg, DPU::sub_32bit);
  unsigned int MsbOp2Reg = TRI->getSubReg(Op2Reg, DPU::sub_32bit_hi);

  BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(), InstrInfo.get(DPU::JNEQrri))
      .addReg(LsbOp1Reg)
      .addReg(LsbOp2Reg)
      .addMBB(JumpMBB);

  BuildMI(trueMBB, MBBIter->getDebugLoc(), InstrInfo.get(DPU::JNEQrri))
      .addReg(MsbOp1Reg)
      .addReg(MsbOp2Reg)
      .addMBB(JumpMBB);
}

static void resolveJcc64AsSub64(MachineBasicBlock *MBB,
                                MachineBasicBlock::iterator MBBIter,
                                const DPUInstrInfo &InstrInfo,
                                DPUAsmCondition::Condition Cond) {
  unsigned int Op1Reg = MBBIter->getOperand(1).getReg();
  unsigned int Op2Reg = MBBIter->getOperand(2).getReg();
  auto JumpMBB = MBBIter->getOperand(3).getMBB();

  MachineFunction *MF = MBB->getParent();
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  unsigned int LsbOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit);
  unsigned int MsbOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit_hi);
  unsigned int LsbOp2Reg = TRI->getSubReg(Op2Reg, DPU::sub_32bit);
  unsigned int MsbOp2Reg = TRI->getSubReg(Op2Reg, DPU::sub_32bit_hi);

  BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(), InstrInfo.get(DPU::SUBzrr))
      .addReg(LsbOp1Reg)
      .addReg(LsbOp2Reg);
  BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(), InstrInfo.get(DPU::SUBCzrrci))
      .addReg(MsbOp1Reg)
      .addReg(MsbOp2Reg)
      .addImm(Cond)
      .addMBB(JumpMBB);
}

static void resolveJcc64(MachineBasicBlock *MBB,
                         MachineBasicBlock::iterator MBBIter,
                         const DPUInstrInfo &InstrInfo) {
  switch (MBBIter->getOperand(0).getImm()) {
  default:
    llvm_unreachable("invalid condition");
  case ISD::SETOEQ:
  case ISD::SETUEQ:
  case ISD::SETEQ:
    resolveJeq64(MBB, MBBIter, InstrInfo);
    break;
  case ISD::SETONE:
  case ISD::SETUNE:
  case ISD::SETNE:
    resolveJneq64(MBB, MBBIter, InstrInfo);
    break;
  case ISD::SETOGT:
  case ISD::SETGT:
    resolveJcc64AsSub64(MBB, MBBIter, InstrInfo,
                        DPUAsmCondition::Condition::ExtendedGreaterThanSigned);
    break;
  case ISD::SETOGE:
  case ISD::SETGE:
    resolveJcc64AsSub64(MBB, MBBIter, InstrInfo,
                        DPUAsmCondition::Condition::GreaterOrEqualSigned);
    break;
  case ISD::SETOLT:
  case ISD::SETLT:
    resolveJcc64AsSub64(MBB, MBBIter, InstrInfo,
                        DPUAsmCondition::Condition::LessThanSigned);
    break;
  case ISD::SETOLE:
  case ISD::SETLE:
    resolveJcc64AsSub64(MBB, MBBIter, InstrInfo,
                        DPUAsmCondition::Condition::ExtendedLessOrEqualSigned);
    break;
  case ISD::SETUGT:
    resolveJcc64AsSub64(
        MBB, MBBIter, InstrInfo,
        DPUAsmCondition::Condition::ExtendedGreaterThanUnsigned);
    break;
  case ISD::SETUGE:
    resolveJcc64AsSub64(MBB, MBBIter, InstrInfo,
                        DPUAsmCondition::Condition::GreaterOrEqualUnsigned);
    break;
  case ISD::SETULT:
    resolveJcc64AsSub64(MBB, MBBIter, InstrInfo,
                        DPUAsmCondition::Condition::LessThanUnsigned);
    break;
  case ISD::SETULE:
    resolveJcc64AsSub64(
        MBB, MBBIter, InstrInfo,
        DPUAsmCondition::Condition::ExtendedLessOrEqualUnsigned);
    break;
  }
}

static bool resolveMacroInstructionsInMBB(MachineBasicBlock *MBB,
                                          const DPUInstrInfo &InstrInfo) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBIter = MBB->begin(), End = MBB->end();

  while (MBBIter != End) {
    bool InstrModified = true;
    switch (MBBIter->getOpcode()) {
    default:
      InstrModified = false;
      break;
    case DPU::Jcc: {
      unsigned int OpCode =
          findJumpOpcodeForCondition(MBBIter->getOperand(0).getImm(), false);
      BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(), InstrInfo.get(OpCode))
          .add(MBBIter->getOperand(1))
          .add(MBBIter->getOperand(2))
          .add(MBBIter->getOperand(3));
      break;
    }
    case DPU::TmpJcci:
    case DPU::Jcci: {
      unsigned int OpCode =
          findJumpOpcodeForCondition(MBBIter->getOperand(0).getImm(), true);
      BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(), InstrInfo.get(OpCode))
          .add(MBBIter->getOperand(1))
          .add(MBBIter->getOperand(2))
          .add(MBBIter->getOperand(MBBIter->getNumOperands() - 1));
      break;
    }
    case DPU::Jcc64:
      resolveJcc64(MBB, MBBIter, InstrInfo);
      break;
    case DPU::SET64cc: {
      MachineFunction *MF = MBB->getParent();
      const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();

      unsigned int DestReg = MBBIter->getOperand(0).getReg();
      auto ImmCond = static_cast<DPUAsmCondition::Condition>(
          MBBIter->getOperand(1).getImm());
      unsigned int Op1Reg = MBBIter->getOperand(2).getReg();
      unsigned int Op2Reg = MBBIter->getOperand(3).getReg();

      DPUAsmCondition::Condition SetCondition =
          findSelect64SetConditionFor(ImmCond);

      unsigned int LSBDOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit);
      unsigned int MSBDOp1Reg = TRI->getSubReg(Op1Reg, DPU::sub_32bit_hi);

      unsigned int LSBOp2Reg = TRI->getSubReg(Op2Reg, DPU::sub_32bit);
      unsigned int MSBOp2Reg = TRI->getSubReg(Op2Reg, DPU::sub_32bit_hi);

      BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(), InstrInfo.get(DPU::SUBzrr))
          .addReg(LSBDOp1Reg)
          .addReg(LSBOp2Reg);
      BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(),
              InstrInfo.get(DPU::SUBCrrrc), DestReg)
          .addReg(MSBDOp1Reg)
          .addReg(MSBOp2Reg)
          .addImm(SetCondition);

      break;
    }
    case DPU::MOVE64ri: {
      MachineFunction *MF = MBB->getParent();
      const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();

      unsigned int DestReg = MBBIter->getOperand(0).getReg();
      int64_t Op1Imm = MBBIter->getOperand(1).getImm();

      int64_t LSBOp1Imm = Op1Imm & 0xFFFFFFFFl;
      int64_t MSBOp1Imm = (Op1Imm >> 32) & 0xFFFFFFFFl;
      if (isInt<32>(Op1Imm)) {
        if (Op1Imm < 0) {
          BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(),
                  InstrInfo.get(DPU::MOVE_Sri), DestReg)
              .addImm(LSBOp1Imm);
        } else {
          BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(),
                  InstrInfo.get(DPU::MOVE_Uri), DestReg)
              .addImm(LSBOp1Imm);
        }
      } else {
        unsigned int LSBDestReg = TRI->getSubReg(DestReg, DPU::sub_32bit);
        unsigned int MSBDestReg = TRI->getSubReg(DestReg, DPU::sub_32bit_hi);

        BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(),
                InstrInfo.get(DPU::MOVEri), LSBDestReg)
            .addImm(LSBOp1Imm);
        BuildMI(*MBB, MBBIter, MBBIter->getDebugLoc(),
                InstrInfo.get(DPU::MOVEri), MSBDestReg)
            .addImm(MSBOp1Imm);
      }
      break;
    }
    case DPU::ADD64rr:
      resolve64BitRegisterAluInstruction(MBB, MBBIter, InstrInfo, DPU::ADDrrr,
                                         DPU::ADDCrrr);
      break;
    case DPU::ADD64ri:
      resolve64BitImmediateAluInstruction(MBB, MBBIter, InstrInfo, DPU::ADDrri,
                                          DPU::ADDCrri);
      break;
    case DPU::SUB64rr:
      resolve64BitRegisterAluInstruction(MBB, MBBIter, InstrInfo, DPU::SUBrrr,
                                         DPU::SUBCrrr);
      break;
    case DPU::OR64rr:
      resolve64BitRegisterAluInstruction(MBB, MBBIter, InstrInfo, DPU::ORrrr,
                                         DPU::ORrrr);
      break;
    case DPU::OR64ri:
      resolve64BitImmediateAluInstruction(MBB, MBBIter, InstrInfo, DPU::ORrri,
                                          DPU::ORrri);
      break;
    case DPU::AND64rr:
      resolve64BitRegisterAluInstruction(MBB, MBBIter, InstrInfo, DPU::ANDrrr,
                                         DPU::ANDrrr);
      break;
    case DPU::AND64ri:
      resolve64BitImmediateAluInstruction(MBB, MBBIter, InstrInfo, DPU::ANDrri,
                                          DPU::ANDrri);
      break;
    case DPU::XOR64rr:
      resolve64BitRegisterAluInstruction(MBB, MBBIter, InstrInfo, DPU::XORrrr,
                                         DPU::XORrrr);
      break;
    case DPU::XOR64ri:
      resolve64BitImmediateAluInstruction(MBB, MBBIter, InstrInfo, DPU::XORrri,
                                          DPU::XORrri);
      break;
    }

    if (InstrModified) {
      MBB->erase(MBBIter++);
      Modified = true;
    } else {
      ++MBBIter;
    }
  }

  return Modified;
}

bool DPUResolveMacroInstrPass::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** DPU/ResolveMacroInstrPass: " << MF.getName()
                    << " **********\n\n");

  auto &SubTarget = static_cast<const DPUSubtarget &>(MF.getSubtarget());
  auto &InstrInfo = *SubTarget.getInstrInfo();
  bool changeMade = false;

  for (auto &MFI : MF) {
    MachineBasicBlock *MBB = &MFI;
    changeMade |= resolveMacroInstructionsInMBB(MBB, InstrInfo);
  }

  return changeMade;
}
