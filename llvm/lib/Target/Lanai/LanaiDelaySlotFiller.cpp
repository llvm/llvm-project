//===-- LanaiDelaySlotFiller.cpp - Lanai delay slot filler ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple pass to fill delay slots with useful instructions.
//
//===----------------------------------------------------------------------===//

#include "Lanai.h"
#include "LanaiTargetMachine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "delay-slot-filler"

STATISTIC(FilledSlots, "Number of delay slots filled");

static cl::opt<bool>
    NopDelaySlotFiller("lanai-nop-delay-filler", cl::init(false),
                       cl::desc("Fill Lanai delay slots with NOPs."),
                       cl::Hidden);

namespace {
struct FillerImpl {
  // Target machine description which we query for reg. names, data
  // layout, etc.
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MachineBasicBlock::instr_iterator LastFiller;

  bool runOnMachineBasicBlock(MachineBasicBlock &MBB);

  bool runOnMachineFunction(MachineFunction &MF) {
    const LanaiSubtarget &Subtarget = MF.getSubtarget<LanaiSubtarget>();
    TII = Subtarget.getInstrInfo();
    TRI = Subtarget.getRegisterInfo();

    bool Changed = false;
    for (MachineBasicBlock &MBB : MF)
      Changed |= runOnMachineBasicBlock(MBB);
    return Changed;
  }

  void insertDefsUses(MachineBasicBlock::instr_iterator MI,
                      SmallSet<unsigned, 32> &RegDefs,
                      SmallSet<unsigned, 32> &RegUses);

  bool isRegInSet(SmallSet<unsigned, 32> &RegSet, unsigned Reg);

  bool delayHasHazard(MachineBasicBlock::instr_iterator MI, bool &SawLoad,
                      bool &SawStore, SmallSet<unsigned, 32> &RegDefs,
                      SmallSet<unsigned, 32> &RegUses);

  bool findDelayInstr(MachineBasicBlock &MBB,
                      MachineBasicBlock::instr_iterator Slot,
                      MachineBasicBlock::instr_iterator &Filler);
};

class LanaiDelaySlotFillerLegacy : public MachineFunctionPass {
  static char ID;

public:
  explicit LanaiDelaySlotFillerLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "Lanai Delay Slot Filler"; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};
char LanaiDelaySlotFillerLegacy::ID = 0;
} // end of anonymous namespace

// createLanaiDelaySlotFillerPass - Returns a pass that fills in delay
// slots in Lanai MachineFunctions
FunctionPass *
llvm::createLanaiDelaySlotFillerLegacyPass(const LanaiTargetMachine & /*tm*/) {
  return new LanaiDelaySlotFillerLegacy();
}

// runOnMachineBasicBlock - Fill in delay slots for the given basic block.
// There is one or two delay slot per delayed instruction.
bool FillerImpl::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  LastFiller = MBB.instr_end();

  for (MachineBasicBlock::instr_iterator I = MBB.instr_begin();
       I != MBB.instr_end(); ++I) {
    if (I->getDesc().hasDelaySlot()) {
      MachineBasicBlock::instr_iterator InstrWithSlot = I;
      MachineBasicBlock::instr_iterator J = I;

      // Treat RET specially as it is only instruction with 2 delay slots
      // generated while all others generated have 1 delay slot.
      if (I->getOpcode() == Lanai::RET) {
        // RET is generated as part of epilogue generation and hence we know
        // what the two instructions preceding it are and that it is safe to
        // insert RET above them.
        MachineBasicBlock::reverse_instr_iterator RI = ++I.getReverse();
        assert(RI->getOpcode() == Lanai::LDW_RI && RI->getOperand(0).isReg() &&
               RI->getOperand(0).getReg() == Lanai::FP &&
               RI->getOperand(1).isReg() &&
               RI->getOperand(1).getReg() == Lanai::FP &&
               RI->getOperand(2).isImm() && RI->getOperand(2).getImm() == -8);
        ++RI;
        assert(RI->getOpcode() == Lanai::ADD_I_LO &&
               RI->getOperand(0).isReg() &&
               RI->getOperand(0).getReg() == Lanai::SP &&
               RI->getOperand(1).isReg() &&
               RI->getOperand(1).getReg() == Lanai::FP);
        MachineBasicBlock::instr_iterator FI = RI.getReverse();
        MBB.splice(std::next(I), &MBB, FI, I);
        FilledSlots += 2;
      } else {
        if (!NopDelaySlotFiller && findDelayInstr(MBB, I, J)) {
          MBB.splice(std::next(I), &MBB, J);
        } else {
          BuildMI(MBB, std::next(I), DebugLoc(), TII->get(Lanai::NOP));
        }
        ++FilledSlots;
      }

      Changed = true;
      // Record the filler instruction that filled the delay slot.
      // The instruction after it will be visited in the next iteration.
      LastFiller = ++I;
      // RET has 2 delay slots, so we need to advance past the second filler
      // too.
      if (InstrWithSlot->getOpcode() == Lanai::RET)
        ++LastFiller;

      // Bundle the delay slot filler to InstrWithSlot so that the machine
      // verifier doesn't expect this instruction to be a terminator.
      MIBundleBuilder(MBB, InstrWithSlot, std::next(LastFiller));
    }
  }
  return Changed;
}

bool FillerImpl::findDelayInstr(MachineBasicBlock &MBB,
                                MachineBasicBlock::instr_iterator Slot,
                                MachineBasicBlock::instr_iterator &Filler) {
  SmallSet<unsigned, 32> RegDefs;
  SmallSet<unsigned, 32> RegUses;

  insertDefsUses(Slot, RegDefs, RegUses);

  bool SawLoad = false;
  bool SawStore = false;

  for (MachineBasicBlock::reverse_instr_iterator I = ++Slot.getReverse();
       I != MBB.instr_rend(); ++I) {
    // skip debug value
    if (I->isDebugInstr())
      continue;

    // Convert to forward iterator.
    MachineBasicBlock::instr_iterator FI = I.getReverse();

    if (I->hasUnmodeledSideEffects() || I->isInlineAsm() || I->isLabel() ||
        FI == LastFiller || I->isPseudo())
      break;

    if (delayHasHazard(FI, SawLoad, SawStore, RegDefs, RegUses)) {
      insertDefsUses(FI, RegDefs, RegUses);
      continue;
    }
    Filler = FI;
    return true;
  }
  return false;
}

bool FillerImpl::delayHasHazard(MachineBasicBlock::instr_iterator MI,
                                bool &SawLoad, bool &SawStore,
                                SmallSet<unsigned, 32> &RegDefs,
                                SmallSet<unsigned, 32> &RegUses) {
  if (MI->isImplicitDef() || MI->isKill())
    return true;

  // Loads or stores cannot be moved past a store to the delay slot
  // and stores cannot be moved past a load.
  if (MI->mayLoad()) {
    if (SawStore)
      return true;
    SawLoad = true;
  }

  if (MI->mayStore()) {
    if (SawStore)
      return true;
    SawStore = true;
    if (SawLoad)
      return true;
  }

  assert((!MI->isCall() && !MI->isReturn()) &&
         "Cannot put calls or returns in delay slot.");

  for (const MachineOperand &MO : MI->operands()) {
    unsigned Reg;

    if (!MO.isReg() || !(Reg = MO.getReg()))
      continue; // skip

    if (MO.isDef()) {
      // check whether Reg is defined or used before delay slot.
      if (isRegInSet(RegDefs, Reg) || isRegInSet(RegUses, Reg))
        return true;
    }
    if (MO.isUse()) {
      // check whether Reg is defined before delay slot.
      if (isRegInSet(RegDefs, Reg))
        return true;
    }
  }
  return false;
}

// Insert Defs and Uses of MI into the sets RegDefs and RegUses.
void FillerImpl::insertDefsUses(MachineBasicBlock::instr_iterator MI,
                                SmallSet<unsigned, 32> &RegDefs,
                                SmallSet<unsigned, 32> &RegUses) {
  for (const MachineOperand &MO : MI->operands()) {
    if (MO.isRegMask()) {
      // For calls, the regmask clobbers many registers. However, we don't
      // want to add all of those to RegDefs as it would prevent any useful
      // delay slot filling. The caller-saved registers will be handled
      // separately.
      continue;
    }

    if (!MO.isReg())
      continue;

    Register Reg = MO.getReg();
    if (Reg != MO.getReg())
      continue;

    // For calls/returns, we must track implicit uses (function arguments)
    // to prevent moving their definitions into the delay slot. We previously
    // only looked at explicit operands which missed implicit argument uses.
    if (MO.isDef()) {
      // For implicit defs from calls (like return value $rv), we add them.
      // But we've already filtered out regmasks above.
      RegDefs.insert(Reg);
    } else if (MO.isUse()) {
      RegUses.insert(Reg);
    }
  }

  // Call & return instructions define SP implicitly. Instructions modifying SP
  // cannot be inserted in the delay slot of a call/return as these instructions
  // are expanded to multiple instructions with SP modified before the branch
  // that has the delay slot.
  if (MI->isCall() || MI->isReturn())
    RegDefs.insert(Lanai::SP);
}

// Returns true if the Reg or its alias is in the RegSet.
bool FillerImpl::isRegInSet(SmallSet<unsigned, 32> &RegSet, unsigned Reg) {
  // Check Reg and all aliased Registers.
  for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
    if (RegSet.count(*AI))
      return true;
  return false;
}

bool LanaiDelaySlotFillerLegacy::runOnMachineFunction(MachineFunction &MF) {
  FillerImpl Impl;
  return Impl.runOnMachineFunction(MF);
}

PreservedAnalyses
LanaiDelaySlotFillerPass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &MFAM) {
  FillerImpl Impl;
  return Impl.runOnMachineFunction(MF)
             ? getMachineFunctionPassPreservedAnalyses()
                   .preserveSet<CFGAnalyses>()
             : PreservedAnalyses::all();
}
