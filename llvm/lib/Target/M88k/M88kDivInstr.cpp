//===-- M88kDelaySlotFiller.cpp - Delay Slot Filler for M88k --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Special pass to handle division instructions on MC88100:
//
// - If TM.useDivInstr() returns false then the signed division instruction is
//   replaced with an inline implementation using unsigned division.
// - If TM.noZeroDivCheck() returns false then additional code is inserted to
//   check for zero division after signed and unsigned divisions.
//
// These changes are necessary due to some hardware limitations. The MC88100
// CPU does not reliable detect division by zero, so an additional check is
// required. The signed division instruction traps into the OS if any of the
// operands are negative. The OS handles this situation transparently but
// trapping into kernel mode is expensive. Therefore the instruction is replaced
// with an inline version using the unsigned division instruction.
//
// Both issues are fixed on the MC88110 CPU, and no code is changed if code for
// it is generated.
//
//===----------------------------------------------------------------------===//

#include "M88kInstrInfo.h"
#include "M88kTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "m88k-div-instr"

using namespace llvm;

STATISTIC(ReplacedDiv, "Number of replaced divs instructions");
STATISTIC(InsertedChecks, "Number of inserted checks for division by zero");

// TODO Move into header file.
enum class CC0 : unsigned {
  EQ0 = 0x2,
  NE0 = 0xd,
  GT0 = 0x1,
  LT0 = 0xc,
  GE0 = 0x3,
  LE0 = 0xe
};

namespace {
class M88kDivInstr : public MachineFunctionPass {
  friend class M88kBuilder;

  const M88kTargetMachine *TM;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const RegisterBankInfo *RBI;
  MachineRegisterInfo *MRI;

  bool ReplaceSignedDiv;
  bool AddZeroDivCheck;

public:
  static char ID;

  M88kDivInstr(const M88kTargetMachine *TM = nullptr);

  MachineFunctionProperties getRequiredProperties() const override;

  bool runOnMachineFunction(MachineFunction &MF) override;

  bool runOnMachineBasicBlock(MachineBasicBlock &MBB);

private:
  MachineInstr *replaceSignedDiv(MachineBasicBlock &MBB, MachineInstr *DivInst);
  void addZeroDivCheck(MachineBasicBlock &MBB, MachineInstr *DivInst);

  MachineBasicBlock *createMBB(MachineBasicBlock *ParentMBB,
                               bool IsSucc = true);
  void addUDivBB(MachineBasicBlock *MBB, MachineInstrBuilder &PHI,
                 const DebugLoc &DL, Register Dividend, Register Divisor,
                 unsigned Variant);
};

// Specialiced builder for m88k instructions.
class M88kBuilder {
  MachineBasicBlock *MBB;
  MachineBasicBlock::iterator I;
  const DebugLoc &DL;

  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  const RegisterBankInfo &RBI;

public:
  M88kBuilder(M88kDivInstr &Pass, MachineBasicBlock *MBB, const DebugLoc &DL)
      : MBB(MBB), I(MBB->end()), DL(DL), TII(*Pass.TII), TRI(*Pass.TRI),
        RBI(*Pass.RBI) {}

  void setMBB(MachineBasicBlock *NewMBB) {
    MBB = NewMBB;
    I = MBB->end();
  }

  void constrainInst(MachineInstr *MI) {
    if (!constrainSelectedInstRegOperands(*MI, TII, TRI, RBI))
      llvm_unreachable("Could not constrain register operands");
  }

  MachineInstr *bcnd(CC0 Cc, Register Reg, MachineBasicBlock *TargetMBB) {
    MachineInstr *MI = BuildMI(*MBB, I, DL, TII.get(M88k::BCND))
                           .addImm(static_cast<int64_t>(Cc))
                           .addReg(Reg)
                           .addMBB(TargetMBB);
    constrainInst(MI);
    return MI;
  }

  MachineInstr *br(MachineBasicBlock *TargetMBB) {
    MachineInstr *MI = BuildMI(*MBB, I, DL, TII.get(M88k::BR)).addMBB(TargetMBB);
    constrainInst(MI);
    return MI;
  }

  MachineInstr *divu(Register DstReg, Register SrcAReg, Register SrcBReg) {
    MachineInstr *MI = BuildMI(*MBB, I, DL, TII.get(M88k::DIVUrr), DstReg)
                           .addReg(SrcAReg)
                           .addReg(SrcBReg);
    constrainInst(MI);
    return MI;
  }

  MachineInstr *neg(Register DstReg, Register SrcReg) {
    MachineInstr *MI = BuildMI(*MBB, I, DL, TII.get(M88k::SUBrr), DstReg)
                           .addReg(M88k::R0)
                           .addReg(SrcReg);
    constrainInst(MI);
    return MI;
  }

  MachineInstr *trap503(Register Reg) {
    MachineInstr *MI = BuildMI(*MBB, I, DL, TII.get(M88k::TRAP503)).addReg(Reg);
    constrainInst(MI);
    return MI;
  }
};

} // end anonymous namespace

M88kDivInstr::M88kDivInstr(const M88kTargetMachine *TM)
    : MachineFunctionPass(ID), TM(TM) {
  initializeM88kDivInstrPass(*PassRegistry::getPassRegistry());
}

MachineFunctionProperties M88kDivInstr::getRequiredProperties() const {
  return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA);
}

bool M88kDivInstr::runOnMachineFunction(MachineFunction &MF) {
  const M88kSubtarget &Subtarget = MF.getSubtarget<M88kSubtarget>();

  // Do nothing if the target cpu is the MC88110.
  if (Subtarget.isMC88110())
    return false;

  TII = Subtarget.getInstrInfo();
  TRI = Subtarget.getRegisterInfo();
  RBI = Subtarget.getRegBankInfo();
  MRI = &MF.getRegInfo();

  ReplaceSignedDiv = !TM->useDivInstr();
  AddZeroDivCheck = !TM->noZeroDivCheck();

  bool Changed = false;
  // Iterating in reverse order avoids newly inserted MBBs.
  for (MachineBasicBlock &MBB : reverse(MF))
    Changed |= runOnMachineBasicBlock(MBB);

  return Changed;
}

// Fill in delay slots for the given basic block.
bool M88kDivInstr::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (MachineBasicBlock::reverse_instr_iterator I = MBB.instr_rbegin();
       I != MBB.instr_rend(); ++I) {
    unsigned Opc = I->getOpcode();
    if ((Opc == M88k::DIVSrr || Opc == M88k::DIVSri) && ReplaceSignedDiv) {
      MachineInstr *MI = replaceSignedDiv(MBB, &*I);
      I = MachineBasicBlock::reverse_instr_iterator(MI);
      Changed = true;
    } else if ((Opc == M88k::DIVUrr || Opc == M88k::DIVSrr) &&
               AddZeroDivCheck) {
      // Add the check only for the 2-register form of the instruction.
      // The immediate of the register-immediate version should never be zero!
      addZeroDivCheck(MBB, &*I);
      Changed = true;
    }
  }
  return Changed;
}

// Creates a new MachineBasicBlock. The new block is inserted after/before the
// basic block MBB, depending on flag IsSucc.
MachineBasicBlock *M88kDivInstr::createMBB(MachineBasicBlock *MBB,
                                           bool IsSucc) {
  // Create the new basic block.
  MachineFunction *MF = MBB->getParent();
  MachineBasicBlock *NewMBB = MF->CreateMachineBasicBlock(MBB->getBasicBlock());

  if (IsSucc) {
    MachineFunction::iterator BBI(MBB);
    MF->insert(++BBI, NewMBB);
    MBB->addSuccessor(NewMBB);
  } else {
    MachineFunction::iterator BBI(MBB);
    MF->insert(--BBI, NewMBB);
    NewMBB->addSuccessor(MBB);
  }

  return NewMBB;
}

enum : unsigned {
  GEGT = 0x0,
  LTGT = 0x2,
  GELE = 0x1,
  LTLE = 0x3,
};

// Adds an unsigned div and a branch to the basic block MBB. The parameter
// Variant indicates the variant of the division which is being added. The
// variants are:
// 0b00: dividend >= 0, divisor > 0    => result >= 0.
// 0b10: dividend < 0, divisor > 0     => result < 0.
// 0b01: dividend >= 0, divisor <= 0   => result <= 0.
// 0b11: dividend < 0, divisor <= 0    => result < 0.
// Handling of the sign is complex:
// - In cases 0b01 and 0b11 (divisor <= 0), the function expects that the
//   divisor is already negated.
// - In cases 0b10 and 0b11 (result < 0), the caller has to negate the result.
// - Negation of the dividend is handled by this function.
void M88kDivInstr::addUDivBB(MachineBasicBlock *MBB, MachineInstrBuilder &PHI,
                             const DebugLoc &DL, Register Dividend,
                             Register Divisor, unsigned Variant) {
  // Add the unsigned divide to the basic block.
  // Negate the dividend if necessary.
  M88kBuilder B(*this, MBB, DL);
  Register ResultReg = MRI->createVirtualRegister(&M88k::GPRRCRegClass);
  bool NegateDividend = Variant & 0x2;
  bool NegatedDivisor = Variant & 0x1;
  if (NegateDividend) {
    Register TmpReg = MRI->createVirtualRegister(&M88k::GPRRCRegClass);
    B.neg(TmpReg, Divisor);
    Divisor = TmpReg;
  }
  B.divu(ResultReg, Dividend, Divisor);
  // missing:
  //         .addReg(Dividend, getKillRegState(NegateDividend))
  //         .addReg(Divisor, getKillRegState(NegatedDivisor));

  // Add branch to basic block containing the PHI instrution.
  MachineBasicBlock *TargetMBB = PHI.getInstr()->getParent();
  if (NegatedDivisor && AddZeroDivCheck) {
    B.bcnd(CC0::NE0, Divisor, TargetMBB);
    B.trap503(Divisor);
  } else {
    B.br(TargetMBB);
  }
  MBB->addSuccessor(TargetMBB);

  // Update the PHI instruction.
  PHI.addReg(ResultReg).addMBB(MBB);
}

MachineInstr *M88kDivInstr::replaceSignedDiv(MachineBasicBlock &MBB,
                                             MachineInstr *DivInst) {
  const DebugLoc &DL = DivInst->getDebugLoc();
  const unsigned NewOpc =
      DivInst->getOpcode() == M88k::DIVSrr ? M88k::DIVUrr : M88k::DIVUri;
  bool IsDivByImm = NewOpc == M88k::DIVUri;
  if (IsDivByImm && DivInst->getOperand(2).getImm() == 0) {
    MachineInstr *MI = BuildMI(MBB, MBB.end(), DL, TII->get(M88k::TB0))
                           .addImm(0)
                           .addReg(M88k::R0)
                           .addImm(503);
    DivInst->eraseFromParent();
    return MI;
  }

  Register DstReg = DivInst->getOperand(0).getReg();
  Register Src1Reg = DivInst->getOperand(1).getReg();
  Register Src2Reg = DivInst->getOperand(2).getReg();

  MachineBasicBlock *TailBB = MBB.splitAt(*DivInst, false);
  MBB.removeSuccessor(TailBB);
  MachineInstrBuilder PHI = BuildMI(*TailBB, TailBB->begin(), DL,
                                    TII->get(TargetOpcode::PHI), DstReg);

  MachineBasicBlock *BBDivisorLE0 = createMBB(&MBB);
  MachineBasicBlock *BBDivisorGT0 = createMBB(&MBB);

  // Branch if divisor is <= 0.
  M88kBuilder B(*this, &MBB, DL);
  B.bcnd(CC0::LE0, Src2Reg, BBDivisorLE0);
  B.br(BBDivisorGT0);

  // Branch if dividend is < 0.
  MachineBasicBlock *BBDividendGE0DivisorGT0 = createMBB(BBDivisorGT0);
  MachineBasicBlock *BBDividendLT0DivisorGT0 = createMBB(BBDivisorGT0);
  B.setMBB(BBDivisorGT0);
  B.bcnd(CC0::LT0, Src2Reg, BBDividendLT0DivisorGT0);
  B.br(BBDividendGE0DivisorGT0);

  // Compute quotient: dividend >= 0, divisor > 0.
  addUDivBB(BBDividendGE0DivisorGT0, PHI, DL, Src1Reg, Src2Reg, GEGT);

  // Compute quotient: dividend < 0, divisor > 0.
  addUDivBB(BBDividendLT0DivisorGT0, PHI, DL, Src1Reg, Src2Reg, LTGT);

  // Negate divisor & branch if dividend is < 0. Divisor is <= 0.
  MachineBasicBlock *BBDividendGE0DivisorLE0 = createMBB(BBDivisorLE0);
  MachineBasicBlock *BBDividendLT0DivisorLE0 = createMBB(BBDivisorLE0);
  B.setMBB(BBDivisorLE0);
  Register NegatedDivisorReg = MRI->createVirtualRegister(&M88k::GPRRCRegClass);
  B.neg(NegatedDivisorReg, Src2Reg);
  B.bcnd(CC0::LT0, Src1Reg, BBDividendLT0DivisorLE0);
  B.br(BBDividendGE0DivisorLE0);

  // Create basic block for negated result.
  MachineBasicBlock *NegTailBB = createMBB(TailBB, false);
  Register NegResultReg = MRI->createVirtualRegister(&M88k::GPRRCRegClass);
  MachineInstrBuilder NegPHI =
      BuildMI(*NegTailBB, NegTailBB->end(), DL, TII->get(TargetOpcode::PHI),
              NegResultReg);
  B.setMBB(NegTailBB);
  Register NegatedResultReg = MRI->createVirtualRegister(&M88k::GPRRCRegClass);
  B.neg(NegatedResultReg, NegPHI.getReg(0) /* RegState::Kill */);
  B.br(TailBB);
  PHI.addReg(NegatedResultReg).addMBB(NegTailBB);

  // Compute quotient: dividend >= 0, divisor <= 0.
  addUDivBB(BBDividendGE0DivisorLE0, NegPHI, DL, Src1Reg, NegatedDivisorReg,
            GELE);

  // Compute quotient: dividend < 0, divisor <= 0.
  addUDivBB(BBDividendLT0DivisorLE0, NegPHI, DL, Src1Reg, NegatedDivisorReg,
            LTLE);

  // Constraint the PHI for the negarted result.
  B.constrainInst(NegPHI);

  // Add phi instruction to tail.
  B.constrainInst(PHI);

  ++ReplacedDiv;
  MachineBasicBlock::iterator I(DivInst);
  ++I;
  DivInst->eraseFromParent();
  return &*I;
}

// Inserts a check for division by zero after the div instruction.
// MI must point to a DIVSrr or DIVUrr instruction.
void M88kDivInstr::addZeroDivCheck(MachineBasicBlock &MBB,
                                   MachineInstr *DivInst) {
  assert(DivInst->getOpcode() == M88k::DIVSrr ||
         DivInst->getOpcode() == M88k::DIVUrr && "Unexpected opcode");
  MachineBasicBlock *TailBB = MBB.splitAt(*DivInst);
  M88kBuilder B(*this, &MBB, DivInst->getDebugLoc());
  B.bcnd(CC0::NE0, DivInst->getOperand(2).getReg(), TailBB);
  B.trap503(DivInst->getOperand(2).getReg());
  ++InsertedChecks;
}

char M88kDivInstr::ID = 0;
INITIALIZE_PASS(M88kDivInstr, DEBUG_TYPE, "Handle div instructions", false,
                false)

namespace llvm {
FunctionPass *createM88kDivInstr(const M88kTargetMachine &TM) {
  return new M88kDivInstr(&TM);
}
} // end namespace llvm
