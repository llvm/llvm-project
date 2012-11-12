//===-- DelaySlotFiller.cpp - MBlaze delay slot filler --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A pass that attempts to fill instructions with delay slots. If no
// instructions can be moved into the delay slot then a NOP is placed there.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "delay-slot-filler"

#include "MBlaze.h"
#include "MBlazeTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

STATISTIC(FilledSlots, "Number of delay slots filled");

static cl::opt<bool> MBDisableDelaySlotFiller(
  "disable-mblaze-delay-filler",
  cl::init(false),
  cl::desc("Disable the MBlaze delay slot filter."),
  cl::Hidden);

namespace {
  struct Filler : public MachineFunctionPass {

    TargetMachine &TM;
    const TargetInstrInfo *TII;

    static char ID;
    Filler(TargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm), TII(tm.getInstrInfo()) { }

    virtual const char *getPassName() const {
      return "MBlaze Delay Slot Filler";
    }

    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
    bool runOnMachineFunction(MachineFunction &F) {
      bool Changed = false;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI)
        Changed |= runOnMachineBasicBlock(*FI);
      return Changed;
    }

  };
  char Filler::ID = 0;
} // end of anonymous namespace

static bool hasImmInstruction(MachineBasicBlock::iterator &candidate) {
    // Any instruction with an immediate mode operand greater than
    // 16-bits requires an implicit IMM instruction.
    unsigned numOper = candidate->getNumOperands();
    for (unsigned op = 0; op < numOper; ++op) {
        MachineOperand &mop = candidate->getOperand(op);

        // The operand requires more than 16-bits to represent.
        if (mop.isImm() && (mop.getImm() < -0x8000 || mop.getImm() > 0x7fff))
          return true;

        // We must assume that unknown immediate values require more than
        // 16-bits to represent.
        if (mop.isGlobal() || mop.isSymbol() || mop.isJTI() || mop.isCPI())
          return true;

        // FIXME: we could probably check to see if the FP value happens
        //        to not need an IMM instruction. For now we just always
        //        assume that FP values do.
        if (mop.isFPImm())
          return true;
    }

    return false;
}

static unsigned getLastRealOperand(MachineBasicBlock::iterator &instr) {
  switch (instr->getOpcode()) {
  default: return instr->getNumOperands();

  // These instructions have a variable number of operands but the first two
  // are the "real" operands that we care about during hazard detection.
  case MBlaze::BRLID:
  case MBlaze::BRALID:
  case MBlaze::BRLD:
  case MBlaze::BRALD:
    return 2;
  }
}

static bool delayHasHazard(MachineBasicBlock::iterator &candidate,
                           MachineBasicBlock::iterator &slot) {
  // Hazard check
  MachineBasicBlock::iterator a = candidate;
  MachineBasicBlock::iterator b = slot;

  // MBB layout:-
  //    candidate := a0 = operation(a1, a2)
  //    ...middle bit...
  //    slot := b0 = operation(b1, b2)

  // Possible hazards:-/
  // 1. a1 or a2 was written during the middle bit
  // 2. a0 was read or written during the middle bit
  // 3. a0 is one or more of {b0, b1, b2}
  // 4. b0 is one or more of {a1, a2}
  // 5. a accesses memory, and the middle bit
  //    contains a store operation.
  bool a_is_memory = candidate->mayLoad() || candidate->mayStore();

  // Determine the number of operands in the slot instruction and in the
  // candidate instruction.
  const unsigned aend = getLastRealOperand(a);
  const unsigned bend = getLastRealOperand(b);

  // Check hazards type 1, 2 and 5 by scanning the middle bit
  MachineBasicBlock::iterator m = a;
  for (++m; m != b; ++m) {
    for (unsigned aop = 0; aop<aend; ++aop) {
      bool aop_is_reg = a->getOperand(aop).isReg();
      if (!aop_is_reg) continue;

      bool aop_is_def = a->getOperand(aop).isDef();
      unsigned aop_reg = a->getOperand(aop).getReg();

      const unsigned mend = getLastRealOperand(m);
      for (unsigned mop = 0; mop<mend; ++mop) {
        bool mop_is_reg = m->getOperand(mop).isReg();
        if (!mop_is_reg) continue;

        bool mop_is_def = m->getOperand(mop).isDef();
        unsigned mop_reg = m->getOperand(mop).getReg();

        if (aop_is_def && (mop_reg == aop_reg))
            return true; // Hazard type 2, because aop = a0
        else if (mop_is_def && (mop_reg == aop_reg))
            return true; // Hazard type 1, because aop in {a1, a2}
      }
    }

    // Check hazard type 5
    if (a_is_memory && m->mayStore())
      return true;
  }

  // Check hazard type 3 & 4
  for (unsigned aop = 0; aop<aend; ++aop) {
    if (a->getOperand(aop).isReg()) {
      unsigned aop_reg = a->getOperand(aop).getReg();

      for (unsigned bop = 0; bop<bend; ++bop) {
        if (b->getOperand(bop).isReg() && !b->getOperand(bop).isImplicit()) {
          unsigned bop_reg = b->getOperand(bop).getReg();
          if (aop_reg == bop_reg)
            return true;
        }
      }
    }
  }

  return false;
}

static bool isDelayFiller(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator candidate) {
  if (candidate == MBB.begin())
    return false;

  --candidate;
  return (candidate->hasDelaySlot());
}

static bool hasUnknownSideEffects(MachineBasicBlock::iterator &I) {
  if (!I->hasUnmodeledSideEffects())
    return false;

  unsigned op = I->getOpcode();
  if (op == MBlaze::ADDK || op == MBlaze::ADDIK ||
      op == MBlaze::ADDC || op == MBlaze::ADDIC ||
      op == MBlaze::ADDKC || op == MBlaze::ADDIKC ||
      op == MBlaze::RSUBK || op == MBlaze::RSUBIK ||
      op == MBlaze::RSUBC || op == MBlaze::RSUBIC ||
      op == MBlaze::RSUBKC || op == MBlaze::RSUBIKC)
    return false;

  return true;
}

static MachineBasicBlock::iterator
findDelayInstr(MachineBasicBlock &MBB,MachineBasicBlock::iterator slot) {
  MachineBasicBlock::iterator I = slot;
  while (true) {
    if (I == MBB.begin())
      break;

    --I;
    if (I->hasDelaySlot() || I->isBranch() || isDelayFiller(MBB,I) ||
        I->isCall() || I->isReturn() || I->isBarrier() ||
        hasUnknownSideEffects(I))
      break;

    if (hasImmInstruction(I) || delayHasHazard(I,slot))
      continue;

    return I;
  }

  return MBB.end();
}

/// runOnMachineBasicBlock - Fill in delay slots for the given basic block.
/// Currently, we fill delay slots with NOPs. We assume there is only one
/// delay slot per delayed instruction.
bool Filler::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I)
    if (I->hasDelaySlot()) {
      MachineBasicBlock::iterator D = MBB.end();
      MachineBasicBlock::iterator J = I;

      if (!MBDisableDelaySlotFiller)
        D = findDelayInstr(MBB,I);

      ++FilledSlots;
      Changed = true;

      if (D == MBB.end())
        BuildMI(MBB, ++J, I->getDebugLoc(), TII->get(MBlaze::NOP));
      else
        MBB.splice(++J, &MBB, D);
    }
  return Changed;
}

/// createMBlazeDelaySlotFillerPass - Returns a pass that fills in delay
/// slots in MBlaze MachineFunctions
FunctionPass *llvm::createMBlazeDelaySlotFillerPass(MBlazeTargetMachine &tm) {
  return new Filler(tm);
}

