//===-- MipsExpandPseudoInsts.cpp - Expand pseudo instructions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Expand atomic pseudos into LL/SC loops after register allocation.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsInstrInfo.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

#define DEBUG_TYPE "mips-pseudo"

namespace {
class MipsExpandPseudo : public MachineFunctionPass {
public:
  static char ID;
  MipsExpandPseudo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  StringRef getPassName() const override {
    return "Mips pseudo instruction expansion pass";
  }

private:
  const MipsInstrInfo *TII;
  const MipsSubtarget *STI;

  struct AtomicOpcodes {
    unsigned LL, SC, BEQ, BNE, OR, Zero;
  };
  AtomicOpcodes getAtomicOpcodes(unsigned Width) const;

  bool expandAtomicCmpSwap(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, bool IsMasked,
                           unsigned Width,
                           MachineBasicBlock::iterator &NextMBBI);
  bool expandAtomicRMW(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                       AtomicRMWInst::BinOp BinOp, bool IsMasked,
                       unsigned Width, MachineBasicBlock::iterator &NextMBBI);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandMBB(MachineBasicBlock &MBB);
};
char MipsExpandPseudo::ID = 0;
} // end anonymous namespace

MipsExpandPseudo::AtomicOpcodes
MipsExpandPseudo::getAtomicOpcodes(unsigned Width) const {
  if (Width == 64)
    return {STI->hasMips64r6() ? Mips::LLD_R6 : Mips::LLD,
            STI->hasMips64r6() ? Mips::SCD_R6 : Mips::SCD,
            Mips::BEQ64,
            Mips::BNE64,
            Mips::OR64,
            Mips::ZERO_64};

  assert(Width == 32 && "Unexpected atomic width");
  if (STI->inMicroMipsMode())
    return {STI->hasMips32r6() ? Mips::LL_MMR6 : Mips::LL_MM,
            STI->hasMips32r6() ? Mips::SC_MMR6 : Mips::SC_MM,
            STI->hasMips32r6() ? Mips::BEQC_MMR6 : Mips::BEQ_MM,
            STI->hasMips32r6() ? Mips::BNEC_MMR6 : Mips::BNE_MM,
            STI->hasMips32r6() ? Mips::OR_MMR6 : Mips::OR_MM,
            Mips::ZERO};

  bool ArePtrs64bit = STI->getABI().ArePtrs64bit();
  return {STI->hasMips32r6() ? (ArePtrs64bit ? Mips::LL64_R6 : Mips::LL_R6)
                             : (ArePtrs64bit ? Mips::LL64 : Mips::LL),
          STI->hasMips32r6() ? (ArePtrs64bit ? Mips::SC64_R6 : Mips::SC_R6)
                             : (ArePtrs64bit ? Mips::SC64 : Mips::SC),
          Mips::BEQ,
          Mips::BNE,
          Mips::OR,
          Mips::ZERO};
}

// Merge selected bits as old ^ ((old ^ new) & mask).
static void insertMaskedMerge(const MipsInstrInfo *TII, MachineBasicBlock *MBB,
                              DebugLoc DL, Register Dest, Register OldVal,
                              Register NewVal, Register Mask) {
  assert(Dest != OldVal && "The old value must survive the masked merge");
  BuildMI(MBB, DL, TII->get(Mips::XOR), Dest).addReg(OldVal).addReg(NewVal);
  BuildMI(MBB, DL, TII->get(Mips::AND), Dest).addReg(Dest).addReg(Mask);
  BuildMI(MBB, DL, TII->get(Mips::XOR), Dest).addReg(OldVal).addReg(Dest);
}

bool MipsExpandPseudo::expandAtomicCmpSwap(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, bool IsMasked,
    unsigned Width, MachineBasicBlock::iterator &NextMBBI) {
  assert((!IsMasked || Width == 32) && "Masked atomics use a word LL/SC");
  MachineInstr &MI = *MBBI;
  MachineFunction *MF = MBB.getParent();
  DebugLoc DL = MI.getDebugLoc();
  AtomicOpcodes Opc = getAtomicOpcodes(Width);

  Register Dest = MI.getOperand(0).getReg();
  Register Scratch = MI.getOperand(1).getReg();
  Register Ptr = MI.getOperand(2).getReg();
  Register CmpVal = MI.getOperand(3).getReg();
  Register NewVal = MI.getOperand(4).getReg();
  Register Mask = IsMasked ? MI.getOperand(5).getReg() : Register();

  auto *LoopMBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto *StoreMBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto *DoneMBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto It = std::next(MBB.getIterator());
  MF->insert(It, LoopMBB);
  MF->insert(It, StoreMBB);
  MF->insert(It, DoneMBB);

  DoneMBB->splice(DoneMBB->end(), &MBB, MI, MBB.end());
  DoneMBB->transferSuccessorsAndUpdatePHIs(&MBB);
  MBB.addSuccessor(LoopMBB, BranchProbability::getOne());
  LoopMBB->addSuccessor(DoneMBB);
  LoopMBB->addSuccessor(StoreMBB);
  LoopMBB->normalizeSuccProbs();
  StoreMBB->addSuccessor(LoopMBB);
  StoreMBB->addSuccessor(DoneMBB);
  StoreMBB->normalizeSuccProbs();

  BuildMI(LoopMBB, DL, TII->get(Opc.LL), Dest).addReg(Ptr).addImm(0);
  Register Loaded = Dest;
  if (IsMasked) {
    BuildMI(LoopMBB, DL, TII->get(Mips::AND), Scratch)
        .addReg(Dest)
        .addReg(Mask);
    Loaded = Scratch;
  }
  BuildMI(LoopMBB, DL, TII->get(Opc.BNE))
      .addReg(Loaded)
      .addReg(CmpVal)
      .addMBB(DoneMBB);

  if (IsMasked) {
    // Scratch contains the masked old value; XOR clears those bits.
    BuildMI(StoreMBB, DL, TII->get(Mips::XOR), Scratch)
        .addReg(Dest)
        .addReg(Scratch);
    BuildMI(StoreMBB, DL, TII->get(Mips::OR), Scratch)
        .addReg(Scratch)
        .addReg(NewVal);
  } else {
    TII->copyPhysReg(*StoreMBB, StoreMBB->end(), DL, Scratch, NewVal, false);
  }
  BuildMI(StoreMBB, DL, TII->get(Opc.SC), Scratch)
      .addReg(Scratch)
      .addReg(Ptr)
      .addImm(0);
  BuildMI(StoreMBB, DL, TII->get(Opc.BEQ))
      .addReg(Scratch)
      .addReg(Opc.Zero)
      .addMBB(LoopMBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();

  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *LoopMBB);
  computeAndAddLiveIns(LiveRegs, *StoreMBB);
  computeAndAddLiveIns(LiveRegs, *DoneMBB);
  return true;
}

static unsigned getAtomicBinOp(AtomicRMWInst::BinOp BinOp, unsigned Width) {
  switch (BinOp) {
  default:
    llvm_unreachable("Unexpected atomicrmw operation");
  case AtomicRMWInst::Add:
    return Width == 64 ? Mips::DADDu : Mips::ADDu;
  case AtomicRMWInst::Sub:
    return Width == 64 ? Mips::DSUBu : Mips::SUBu;
  case AtomicRMWInst::And:
  case AtomicRMWInst::Nand:
    return Width == 64 ? Mips::AND64 : Mips::AND;
  case AtomicRMWInst::Or:
    return Width == 64 ? Mips::OR64 : Mips::OR;
  case AtomicRMWInst::Xor:
    return Width == 64 ? Mips::XOR64 : Mips::XOR;
  }
}

bool MipsExpandPseudo::expandAtomicRMW(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MBBI,
                                       AtomicRMWInst::BinOp BinOp,
                                       bool IsMasked, unsigned Width,
                                       MachineBasicBlock::iterator &NextMBBI) {
  assert((!IsMasked || Width == 32) && "Masked atomics use a word LL/SC");
  MachineInstr &MI = *MBBI;
  MachineFunction *MF = MBB.getParent();
  DebugLoc DL = MI.getDebugLoc();
  AtomicOpcodes Opc = getAtomicOpcodes(Width);
  bool IsMin = BinOp == AtomicRMWInst::Min || BinOp == AtomicRMWInst::UMin;
  bool IsMax = BinOp == AtomicRMWInst::Max || BinOp == AtomicRMWInst::UMax;
  bool IsMinMax = IsMin || IsMax;
  bool IsSigned = BinOp == AtomicRMWInst::Min || BinOp == AtomicRMWInst::Max;
  bool NeedsBranch = IsMinMax && !STI->hasMips4() && !STI->hasMips32();

  Register Dest = MI.getOperand(0).getReg();
  Register Scratch = MI.getOperand(1).getReg();
  unsigned FirstInput = MI.getNumExplicitDefs();
  Register Ptr = MI.getOperand(FirstInput).getReg();
  Register Incr = MI.getOperand(FirstInput + 1).getReg();
  Register Mask =
      IsMasked ? MI.getOperand(FirstInput + 2).getReg() : Register();

  auto *LoopMBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  MachineBasicBlock *UpdateMBB = nullptr;
  MachineBasicBlock *StoreMBB = LoopMBB;
  auto *DoneMBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto It = std::next(MBB.getIterator());
  MF->insert(It, LoopMBB);
  if (NeedsBranch) {
    UpdateMBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
    StoreMBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
    MF->insert(It, UpdateMBB);
    MF->insert(It, StoreMBB);
    LoopMBB->addSuccessor(UpdateMBB);
    LoopMBB->addSuccessor(StoreMBB);
    LoopMBB->normalizeSuccProbs();
    UpdateMBB->addSuccessor(StoreMBB, BranchProbability::getOne());
  }
  MF->insert(It, DoneMBB);
  DoneMBB->splice(DoneMBB->end(), &MBB, MI, MBB.end());
  DoneMBB->transferSuccessorsAndUpdatePHIs(&MBB);
  MBB.addSuccessor(LoopMBB, BranchProbability::getOne());
  StoreMBB->addSuccessor(LoopMBB);
  StoreMBB->addSuccessor(DoneMBB);
  StoreMBB->normalizeSuccProbs();

  BuildMI(LoopMBB, DL, TII->get(Opc.LL), Dest).addReg(Ptr).addImm(0);
  Register NewVal = Scratch;
  if (IsMinMax) {
    Register Cmp = MI.getOperand(2).getReg();
    // SLT64 defines a GPR32 result, but the r6 selects reuse the full scratch
    // register for a 64-bit value. Its explicit output reserves both aliases.
    Register Cmp32 = Width == 64
                         ? STI->getRegisterInfo()->getSubReg(Cmp, Mips::sub_32)
                         : Cmp.asMCReg();
    Register Loaded = Dest;
    if (IsMasked) {
      BuildMI(LoopMBB, DL, TII->get(Mips::AND), Scratch)
          .addReg(Dest)
          .addReg(Mask);
      if (IsSigned) {
        Register SextShift = MI.getOperand(FirstInput + 3).getReg();
        BuildMI(LoopMBB, DL, TII->get(Mips::SLLV), Scratch)
            .addReg(Scratch)
            .addReg(SextShift);
        BuildMI(LoopMBB, DL, TII->get(Mips::SRAV), Scratch)
            .addReg(Scratch)
            .addReg(SextShift);
      }
      Loaded = Scratch;
    }

    unsigned SLT = IsSigned ? Mips::SLT : Mips::SLTu;
    if (Width == 64)
      SLT = IsSigned ? Mips::SLT64 : Mips::SLTu64;
    else if (STI->inMicroMipsMode())
      SLT = IsSigned ? Mips::SLT_MM : Mips::SLTu_MM;
    BuildMI(LoopMBB, DL, TII->get(SLT), Cmp32)
        .addReg(IsMin ? Incr : Loaded)
        .addReg(IsMin ? Loaded : Incr);
    if (STI->hasMips32r6()) {
      unsigned SELEQZ = Width == 64 ? Mips::SELEQZ64 : Mips::SELEQZ;
      unsigned SELNEZ = Width == 64 ? Mips::SELNEZ64 : Mips::SELNEZ;
      if (STI->inMicroMipsMode()) {
        SELEQZ = Mips::SELEQZ_MMR6;
        SELNEZ = Mips::SELNEZ_MMR6;
      }
      BuildMI(LoopMBB, DL, TII->get(SELEQZ), Scratch)
          .addReg(Loaded)
          .addReg(Cmp);
      BuildMI(LoopMBB, DL, TII->get(SELNEZ), Cmp).addReg(Incr).addReg(Cmp);
      BuildMI(LoopMBB, DL, TII->get(Opc.OR), Scratch)
          .addReg(Scratch)
          .addReg(Cmp);
    } else {
      if (!IsMasked)
        BuildMI(LoopMBB, DL, TII->get(Opc.OR), Scratch)
            .addReg(Dest)
            .addReg(Opc.Zero);
      if (!NeedsBranch) {
        unsigned MOVN = Width == 64 ? Mips::MOVN_I64_I64 : Mips::MOVN_I_I;
        if (STI->inMicroMipsMode())
          MOVN = Mips::MOVN_I_MM;
        BuildMI(LoopMBB, DL, TII->get(MOVN), Scratch)
            .addReg(Incr)
            .addReg(Cmp)
            .addReg(Scratch);
      } else {
        BuildMI(LoopMBB, DL, TII->get(Mips::BEQ))
            .addReg(Cmp32)
            .addReg(Mips::ZERO)
            .addMBB(StoreMBB);
        BuildMI(UpdateMBB, DL, TII->get(Opc.OR), Scratch)
            .addReg(Incr)
            .addReg(Opc.Zero);
      }
    }
  } else if (BinOp == AtomicRMWInst::Xchg) {
    if (IsMasked)
      NewVal = Incr;
    else
      BuildMI(LoopMBB, DL, TII->get(Opc.OR), Scratch)
          .addReg(Incr)
          .addReg(Opc.Zero);
  } else {
    BuildMI(LoopMBB, DL, TII->get(getAtomicBinOp(BinOp, Width)), Scratch)
        .addReg(Dest)
        .addReg(Incr);
    if (BinOp == AtomicRMWInst::Nand)
      BuildMI(LoopMBB, DL, TII->get(Width == 64 ? Mips::NOR64 : Mips::NOR),
              Scratch)
          .addReg(Scratch)
          .addReg(Opc.Zero);
  }

  if (IsMasked)
    insertMaskedMerge(TII, StoreMBB, DL, Scratch, Dest, NewVal, Mask);
  BuildMI(StoreMBB, DL, TII->get(Opc.SC), Scratch)
      .addReg(Scratch)
      .addReg(Ptr)
      .addImm(0);
  BuildMI(StoreMBB, DL, TII->get(Opc.BEQ))
      .addReg(Scratch)
      .addReg(Opc.Zero)
      .addMBB(LoopMBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();

  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *LoopMBB);
  if (NeedsBranch) {
    computeAndAddLiveIns(LiveRegs, *UpdateMBB);
    computeAndAddLiveIns(LiveRegs, *StoreMBB);
  }
  computeAndAddLiveIns(LiveRegs, *DoneMBB);
  return true;
}

bool MipsExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                MachineBasicBlock::iterator &NextMBBI) {
  switch (MBBI->getOpcode()) {
  default:
    return false;
  case Mips::ATOMIC_CMP_SWAP_I32:
    return expandAtomicCmpSwap(MBB, MBBI, false, 32, NextMBBI);
  case Mips::ATOMIC_SWAP_I32:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Xchg, false, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_ADD_I32:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Add, false, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_SUB_I32:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Sub, false, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_AND_I32:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::And, false, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_OR_I32:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Or, false, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_XOR_I32:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Xor, false, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_NAND_I32:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Nand, false, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_MIN_I32:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Min, false, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_MAX_I32:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Max, false, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_UMIN_I32:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::UMin, false, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_UMAX_I32:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::UMax, false, 32, NextMBBI);
  case Mips::ATOMIC_CMP_SWAP_I64:
    return expandAtomicCmpSwap(MBB, MBBI, false, 64, NextMBBI);
  case Mips::ATOMIC_SWAP_I64:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Xchg, false, 64, NextMBBI);
  case Mips::ATOMIC_LOAD_ADD_I64:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Add, false, 64, NextMBBI);
  case Mips::ATOMIC_LOAD_SUB_I64:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Sub, false, 64, NextMBBI);
  case Mips::ATOMIC_LOAD_AND_I64:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::And, false, 64, NextMBBI);
  case Mips::ATOMIC_LOAD_OR_I64:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Or, false, 64, NextMBBI);
  case Mips::ATOMIC_LOAD_XOR_I64:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Xor, false, 64, NextMBBI);
  case Mips::ATOMIC_LOAD_NAND_I64:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Nand, false, 64, NextMBBI);
  case Mips::ATOMIC_LOAD_MIN_I64:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Min, false, 64, NextMBBI);
  case Mips::ATOMIC_LOAD_MAX_I64:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Max, false, 64, NextMBBI);
  case Mips::ATOMIC_LOAD_UMIN_I64:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::UMin, false, 64, NextMBBI);
  case Mips::ATOMIC_LOAD_UMAX_I64:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::UMax, false, 64, NextMBBI);
  case Mips::ATOMIC_CMP_SWAP_MASKED:
    return expandAtomicCmpSwap(MBB, MBBI, true, 32, NextMBBI);
  case Mips::ATOMIC_SWAP_MASKED:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Xchg, true, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_ADD_MASKED:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Add, true, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_SUB_MASKED:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Sub, true, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_NAND_MASKED:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Nand, true, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_MIN_MASKED:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Min, true, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_MAX_MASKED:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::Max, true, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_UMIN_MASKED:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::UMin, true, 32, NextMBBI);
  case Mips::ATOMIC_LOAD_UMAX_MASKED:
    return expandAtomicRMW(MBB, MBBI, AtomicRMWInst::UMax, true, 32, NextMBBI);
  }
}

bool MipsExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool MipsExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<MipsSubtarget>();
  TII = STI->getInstrInfo();

  bool Modified = false;
  for (MachineBasicBlock &MBB : MF)
    Modified |= expandMBB(MBB);

  if (Modified)
    MF.RenumberBlocks();

  return Modified;
}

/// createMipsExpandPseudoPass - returns an instance of the pseudo instruction
/// expansion pass.
FunctionPass *llvm::createMipsExpandPseudoPass() {
  return new MipsExpandPseudo();
}
