//===-- AArch64CondBrTuning.cpp --- Conditional branch tuning for AArch64 -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains a pass that transforms CBZ/CBNZ/TBZ/TBNZ instructions
/// into a conditional branch (B.cond), when the NZCV flags can be set for
/// "free".  This is preferred on targets that have more flexibility when
/// scheduling B.cond instructions as compared to CBZ/CBNZ/TBZ/TBNZ (assuming
/// all other variables are equal).  This can also reduce register pressure.
///
/// A few examples:
///
/// 1) add w8, w0, w1  -> cmn w0, w1             ; CMN is an alias of ADDS.
///    cbz w8, .LBB_2  -> b.eq .LBB0_2
///
/// 2) add w8, w0, w1  -> adds w8, w0, w1        ; w8 has multiple uses.
///    cbz w8, .LBB1_2 -> b.eq .LBB1_2
///
/// 3) sub w8, w0, w1       -> subs w8, w0, w1   ; w8 has multiple uses.
///    tbz w8, #31, .LBB6_2 -> b.pl .LBB6_2
///
/// 4) lsr x8, x0, 32    ->  mov	w9, #4294967296   ; will be hoisted
///    cbz x8, .LBB0_0   ->  cmp	x8, x9            ; fused with bcc
///                      ->  b.lo .LBB0_0
///
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-cond-br-tuning"
#define AARCH64_CONDBR_TUNING_NAME "AArch64 Conditional Branch Tuning"

namespace {
class AArch64CondBrTuning : public MachineFunctionPass {
  const AArch64Subtarget *STI;
  const AArch64InstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const MachineLoopInfo *MLI;

  MachineRegisterInfo *MRI;

public:
  static char ID;
  AArch64CondBrTuning() : MachineFunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return AARCH64_CONDBR_TUNING_NAME; }

private:
  MachineInstr *getOperandDef(const MachineOperand &MO);
  MachineInstr *tryConvertToFlagSetting(MachineInstr &MI, bool IsFlagSetting,
                                        bool Is64Bit);
  MachineInstr *convertToCondBr(MachineInstr &MI);
  bool tryToTuneBranch(MachineInstr &MI, MachineInstr &DefMI);
  bool tryTransformShiftBranch(MachineBasicBlock &MBB, MachineInstr &MI) const;
};
} // end anonymous namespace

char AArch64CondBrTuning::ID = 0;

INITIALIZE_PASS(AArch64CondBrTuning, "aarch64-cond-br-tuning",
                AARCH64_CONDBR_TUNING_NAME, false, false)

void AArch64CondBrTuning::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static bool isNZCVLiveout(const MachineBasicBlock &MBB) {
  return any_of(MBB.successors(), [](const MachineBasicBlock *SuccMBB) {
    return SuccMBB->isLiveIn(AArch64::NZCV);
  });
}

MachineInstr *AArch64CondBrTuning::getOperandDef(const MachineOperand &MO) {
  if (!MO.getReg().isVirtual())
    return nullptr;
  return MRI->getUniqueVRegDef(MO.getReg());
}

MachineInstr *AArch64CondBrTuning::tryConvertToFlagSetting(MachineInstr &MI,
                                                           bool IsFlagSetting,
                                                           bool Is64Bit) {
  // If the instruction has a frame index operand, we can't safely convert it
  // to a flag-setting form, because it can be expanded later into multiple
  // instructions, which don't all have flag-setting forms (e.g. ADDVL).
  if (any_of(MI.operands(), [](const MachineOperand &Op) { return Op.isFI(); }))
    return nullptr;

  // If this is already the flag setting version of the instruction (e.g., SUBS)
  // just make sure the implicit-def of NZCV isn't marked dead.
  if (IsFlagSetting) {
    for (MachineOperand &MO : MI.implicit_operands())
      if (MO.isReg() && MO.isDead() && MO.getReg() == AArch64::NZCV)
        MO.setIsDead(false);
    return &MI;
  }
  unsigned NewOpc = TII->convertToFlagSettingOpc(MI.getOpcode());
  Register NewDestReg = MI.getOperand(0).getReg();
  if (MRI->hasOneNonDBGUse(MI.getOperand(0).getReg()))
    NewDestReg = Is64Bit ? AArch64::XZR : AArch64::WZR;

  MachineInstrBuilder MIB = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(),
                                    TII->get(NewOpc), NewDestReg);

  // If the MI has a debug instruction number, preserve that in the new Machine
  // Instruction that is created.
  if (MI.peekDebugInstrNum() != 0)
    MIB->setDebugInstrNum(MI.peekDebugInstrNum());

  for (const MachineOperand &MO : llvm::drop_begin(MI.operands()))
    MIB.add(MO);

  return MIB;
}

MachineInstr *AArch64CondBrTuning::convertToCondBr(MachineInstr &MI) {
  AArch64CC::CondCode CC;
  MachineBasicBlock *TargetMBB = TII->getBranchDestBlock(MI);
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected opcode!");

  case AArch64::CBZW:
  case AArch64::CBZX:
    CC = AArch64CC::EQ;
    break;
  case AArch64::CBNZW:
  case AArch64::CBNZX:
    CC = AArch64CC::NE;
    break;
  case AArch64::TBZW:
  case AArch64::TBZX:
    CC = AArch64CC::PL;
    break;
  case AArch64::TBNZW:
  case AArch64::TBNZX:
    CC = AArch64CC::MI;
    break;
  }
  return BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), TII->get(AArch64::Bcc))
      .addImm(CC)
      .addMBB(TargetMBB);
}

bool AArch64CondBrTuning::tryToTuneBranch(MachineInstr &MI,
                                          MachineInstr &DefMI) {
  // We don't want NZCV bits live across blocks.
  if (MI.getParent() != DefMI.getParent())
    return false;

  bool IsFlagSetting = true;
  unsigned MIOpc = MI.getOpcode();
  MachineInstr *NewCmp = nullptr, *NewBr = nullptr;
  switch (DefMI.getOpcode()) {
  default:
    return false;
  case AArch64::ADDWri:
  case AArch64::ADDWrr:
  case AArch64::ADDWrs:
  case AArch64::ADDWrx:
  case AArch64::ANDWri:
  case AArch64::ANDWrr:
  case AArch64::ANDWrs:
  case AArch64::BICWrr:
  case AArch64::BICWrs:
  case AArch64::SUBWri:
  case AArch64::SUBWrr:
  case AArch64::SUBWrs:
  case AArch64::SUBWrx:
    IsFlagSetting = false;
    [[fallthrough]];
  case AArch64::ADDSWri:
  case AArch64::ADDSWrr:
  case AArch64::ADDSWrs:
  case AArch64::ADDSWrx:
  case AArch64::ANDSWri:
  case AArch64::ANDSWrr:
  case AArch64::ANDSWrs:
  case AArch64::BICSWrr:
  case AArch64::BICSWrs:
  case AArch64::SUBSWri:
  case AArch64::SUBSWrr:
  case AArch64::SUBSWrs:
  case AArch64::SUBSWrx:
    switch (MIOpc) {
    default:
      llvm_unreachable("Unexpected opcode!");

    case AArch64::CBZW:
    case AArch64::CBNZW:
    case AArch64::TBZW:
    case AArch64::TBNZW:
      // Check to see if the TBZ/TBNZ is checking the sign bit.
      if ((MIOpc == AArch64::TBZW || MIOpc == AArch64::TBNZW) &&
          MI.getOperand(1).getImm() != 31)
        return false;

      // There must not be any instruction between DefMI and MI that clobbers or
      // reads NZCV.
      if (isNZCVTouchedInInstructionRange(*DefMI.getNextNode(), MI, TRI))
        return false;

      NewCmp = tryConvertToFlagSetting(DefMI, IsFlagSetting, /*Is64Bit=*/false);
      if (!NewCmp)
        return false;

      LLVM_DEBUG(dbgs() << "  Replacing instructions:\n    ");
      LLVM_DEBUG(DefMI.print(dbgs()));
      LLVM_DEBUG(dbgs() << "    ");
      LLVM_DEBUG(MI.print(dbgs()));

      NewBr = convertToCondBr(MI);
      break;
    }
    break;

  case AArch64::ADDXri:
  case AArch64::ADDXrr:
  case AArch64::ADDXrs:
  case AArch64::ADDXrx:
  case AArch64::ANDXri:
  case AArch64::ANDXrr:
  case AArch64::ANDXrs:
  case AArch64::BICXrr:
  case AArch64::BICXrs:
  case AArch64::SUBXri:
  case AArch64::SUBXrr:
  case AArch64::SUBXrs:
  case AArch64::SUBXrx:
    IsFlagSetting = false;
    [[fallthrough]];
  case AArch64::ADDSXri:
  case AArch64::ADDSXrr:
  case AArch64::ADDSXrs:
  case AArch64::ADDSXrx:
  case AArch64::ANDSXri:
  case AArch64::ANDSXrr:
  case AArch64::ANDSXrs:
  case AArch64::BICSXrr:
  case AArch64::BICSXrs:
  case AArch64::SUBSXri:
  case AArch64::SUBSXrr:
  case AArch64::SUBSXrs:
  case AArch64::SUBSXrx:
    switch (MIOpc) {
    default:
      llvm_unreachable("Unexpected opcode!");

    case AArch64::CBZX:
    case AArch64::CBNZX:
    case AArch64::TBZX:
    case AArch64::TBNZX: {
      // Check to see if the TBZ/TBNZ is checking the sign bit.
      if ((MIOpc == AArch64::TBZX || MIOpc == AArch64::TBNZX) &&
          MI.getOperand(1).getImm() != 63)
        return false;
      // There must not be any instruction between DefMI and MI that clobbers or
      // reads NZCV.
      if (isNZCVTouchedInInstructionRange(*DefMI.getNextNode(), MI, TRI))
        return false;

      NewCmp = tryConvertToFlagSetting(DefMI, IsFlagSetting, /*Is64Bit=*/true);
      if (!NewCmp)
        return false;

      LLVM_DEBUG(dbgs() << "  Replacing instructions:\n    ");
      LLVM_DEBUG(DefMI.print(dbgs()));
      LLVM_DEBUG(dbgs() << "    ");
      LLVM_DEBUG(MI.print(dbgs()));

      NewBr = convertToCondBr(MI);
      break;
    }
    }
    break;
  }
  (void)NewCmp; (void)NewBr;
  assert(NewCmp && NewBr && "Expected new instructions.");

  LLVM_DEBUG(dbgs() << "  with instruction:\n    ");
  LLVM_DEBUG(NewCmp->print(dbgs()));
  LLVM_DEBUG(dbgs() << "    ");
  LLVM_DEBUG(NewBr->print(dbgs()));

  // If this was a flag setting version of the instruction, we use the original
  // instruction by just clearing the dead marked on the implicit-def of NCZV.
  // Therefore, we should not erase this instruction.
  if (!IsFlagSetting)
    DefMI.eraseFromParent();
  MI.eraseFromParent();
  return true;
}

/// Try to transform lsr + cbz/cbnz instruction pairs into a cmp + bcc pair.
/// This will require an additional instruction to materialize the cmp
/// immediate, but it saves an instruction when the imm mat instruction
/// can be hoisted out of the block and the cmp + bcc instructions are fused.
bool AArch64CondBrTuning::tryTransformShiftBranch(MachineBasicBlock &MBB,
                                                  MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    break;
  case AArch64::CBNZW:
  case AArch64::CBZW: {
    const Register CondReg = MI.getOperand(0).getReg();
    if (!CondReg.isVirtual() || !MRI->hasOneNonDBGUse(CondReg))
      return false;

    MachineInstr &CondDef = *MRI->getUniqueVRegDef(CondReg);
    if (CondDef.getOpcode() != AArch64::UBFMWri ||
        CondDef.getOperand(3).getImm() != 31)
      return false;

    // NZCV can't be live after the last non-terminator in the block for the
    // transformation to work. At this point we know it's not live after MI,
    // so we only have to check the rest of the range here.
    if (isNZCVTouchedInInstructionRange(*MBB.getFirstTerminator(), MI, TRI))
      return false;

    const uint64_t ShiftAmt = CondDef.getOperand(2).getImm();
    const uint64_t CmpImm = 1ULL << ShiftAmt;

    const AArch64CC::CondCode CC =
        (MI.getOpcode() == AArch64::CBZW) ? AArch64CC::LO : AArch64CC::HS;

    const Register ImmReg = MRI->createVirtualRegister(&AArch64::GPR32RegClass);

    BuildMI(MBB, MBB.getFirstTerminator(), CondDef.getDebugLoc(),
            TII->get(AArch64::MOVi32imm))
        .addDef(ImmReg)
        .addImm(CmpImm);

    BuildMI(MBB, MBB.getFirstTerminator(), CondDef.getDebugLoc(),
            TII->get(AArch64::SUBSWrr))
        .addDef(AArch64::WZR)
        .addReg(CondDef.getOperand(1).getReg())
        .addReg(ImmReg);

    BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(AArch64::Bcc))
        .addImm(CC)
        .addMBB(MI.getOperand(1).getMBB());

    CondDef.eraseFromParent();
    MI.eraseFromParent();
    return true;
  }
  case AArch64::CBNZX:
  case AArch64::CBZX: {
    const Register CondReg = MI.getOperand(0).getReg();
    if (!CondReg.isVirtual() || !MRI->hasOneNonDBGUse(CondReg))
      return false;

    MachineInstr &CondDef = *MRI->getUniqueVRegDef(CondReg);
    if (CondDef.getOpcode() != AArch64::UBFMXri ||
        CondDef.getOperand(3).getImm() != 63)
      return false;

    if (isNZCVTouchedInInstructionRange(*MBB.getFirstTerminator(), MI, TRI))
      return false;

    const uint64_t ShiftAmt = CondDef.getOperand(2).getImm();
    const uint64_t CmpImm = 1ULL << ShiftAmt;

    const AArch64CC::CondCode CC =
        (MI.getOpcode() == AArch64::CBZX) ? AArch64CC::LO : AArch64CC::HS;

    const Register ImmReg = MRI->createVirtualRegister(&AArch64::GPR64RegClass);

    BuildMI(MBB, MBB.getFirstTerminator(), CondDef.getDebugLoc(),
            TII->get(AArch64::MOVi64imm))
        .addDef(ImmReg)
        .addImm(CmpImm);

    BuildMI(MBB, MBB.getFirstTerminator(), CondDef.getDebugLoc(),
            TII->get(AArch64::SUBSXrr))
        .addDef(AArch64::XZR)
        .addReg(CondDef.getOperand(1).getReg())
        .addReg(ImmReg);

    BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(AArch64::Bcc))
        .addImm(CC)
        .addMBB(MI.getOperand(1).getMBB());

    CondDef.eraseFromParent();
    MI.eraseFromParent();
    return true;
  }
  }

  return false;
}

bool AArch64CondBrTuning::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  LLVM_DEBUG(
      dbgs() << "********** AArch64 Conditional Branch Tuning  **********\n"
             << "********** Function: " << MF.getName() << '\n');

  STI = &MF.getSubtarget<AArch64Subtarget>();
  TII = static_cast<const AArch64InstrInfo *>(STI->getInstrInfo());
  TRI = STI->getRegisterInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MRI = &MF.getRegInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    bool LocalChange = false;
    for (MachineInstr &MI : MBB.terminators()) {
      switch (MI.getOpcode()) {
      default:
        break;
      case AArch64::CBZW:
      case AArch64::CBZX:
      case AArch64::CBNZW:
      case AArch64::CBNZX:
      case AArch64::TBZW:
      case AArch64::TBZX:
      case AArch64::TBNZW:
      case AArch64::TBNZX:
        MachineInstr *DefMI = getOperandDef(MI.getOperand(0));
        LocalChange = (DefMI && tryToTuneBranch(MI, *DefMI));
        break;
      }
      // If the optimization was successful, we can't optimize any other
      // branches because doing so would clobber the NZCV flags.
      if (LocalChange) {
        Changed = true;
        break;
      }
    }
  }

  if (!STI->hasCmpBccFusion())
    return Changed;

  for (const MachineLoop *ML : *MLI) {
    for (MachineBasicBlock *MBB : ML->blocks()) {
      if (isNZCVLiveout(*MBB))
        continue;

      for (MachineInstr &MI : reverse(MBB->terminators())) {
        if (MI.readsRegister(AArch64::NZCV, TRI))
          break;

        if (tryTransformShiftBranch(*MBB, MI)) {
          Changed = true;
          break;
        }
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createAArch64CondBrTuning() {
  return new AArch64CondBrTuning();
}
