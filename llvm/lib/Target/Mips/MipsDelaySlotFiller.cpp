//===-- DelaySlotFiller.cpp - Mips Delay Slot Filler ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simple pass to fills delay slots with useful instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "delay-slot-filler"

#include "Mips.h"
#include "MipsTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"

using namespace llvm;

STATISTIC(FilledSlots, "Number of delay slots filled");
STATISTIC(UsefulSlots, "Number of delay slots filled with instructions that"
                       " are not NOP.");

static cl::opt<bool> DisableDelaySlotFiller(
  "disable-mips-delay-filler",
  cl::init(false),
  cl::desc("Disable the delay slot filler, which attempts to fill the Mips"
           "delay slots with useful instructions."),
  cl::Hidden);

// This option can be used to silence complaints by machine verifier passes.
static cl::opt<bool> SkipDelaySlotFiller(
  "skip-mips-delay-filler",
  cl::init(false),
  cl::desc("Skip MIPS' delay slot filling pass."),
  cl::Hidden);

namespace {
  struct Filler : public MachineFunctionPass {
    typedef MachineBasicBlock::instr_iterator InstrIter;
    typedef MachineBasicBlock::reverse_instr_iterator ReverseInstrIter;

    TargetMachine &TM;
    const TargetInstrInfo *TII;
    InstrIter LastFiller;

    static char ID;
    Filler(TargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm), TII(tm.getInstrInfo()) { }

    virtual const char *getPassName() const {
      return "Mips Delay Slot Filler";
    }

    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
    bool runOnMachineFunction(MachineFunction &F) {
      if (SkipDelaySlotFiller)
        return false;

      bool Changed = false;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI)
        Changed |= runOnMachineBasicBlock(*FI);
      return Changed;
    }

    bool isDelayFiller(MachineBasicBlock &MBB,
                       InstrIter candidate);

    void insertCallUses(InstrIter MI,
                        SmallSet<unsigned, 32> &RegDefs,
                        SmallSet<unsigned, 32> &RegUses);

    void insertDefsUses(InstrIter MI,
                        SmallSet<unsigned, 32> &RegDefs,
                        SmallSet<unsigned, 32> &RegUses);

    bool IsRegInSet(SmallSet<unsigned, 32> &RegSet,
                    unsigned Reg);

    bool delayHasHazard(InstrIter candidate,
                        bool &sawLoad, bool &sawStore,
                        SmallSet<unsigned, 32> &RegDefs,
                        SmallSet<unsigned, 32> &RegUses);

    bool
    findDelayInstr(MachineBasicBlock &MBB, InstrIter slot,
                   InstrIter &Filler);


  };
  char Filler::ID = 0;
} // end of anonymous namespace

/// runOnMachineBasicBlock - Fill in delay slots for the given basic block.
/// We assume there is only one delay slot per delayed instruction.
bool Filler::
runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  LastFiller = MBB.instr_end();

  for (InstrIter I = MBB.instr_begin(); I != MBB.instr_end(); ++I)
    if (I->hasDelaySlot()) {
      ++FilledSlots;
      Changed = true;

      InstrIter D;

      // Delay slot filling is disabled at -O0.
      if (!DisableDelaySlotFiller && (TM.getOptLevel() != CodeGenOpt::None) &&
          findDelayInstr(MBB, I, D)) {
        MBB.splice(llvm::next(I), &MBB, D);
        ++UsefulSlots;
      } else
        BuildMI(MBB, llvm::next(I), I->getDebugLoc(), TII->get(Mips::NOP));

      // Record the filler instruction that filled the delay slot.
      // The instruction after it will be visited in the next iteration.
      LastFiller = ++I;

      // Set InsideBundle bit so that the machine verifier doesn't expect this
      // instruction to be a terminator.
      LastFiller->setIsInsideBundle();
     }
  return Changed;

}

/// createMipsDelaySlotFillerPass - Returns a pass that fills in delay
/// slots in Mips MachineFunctions
FunctionPass *llvm::createMipsDelaySlotFillerPass(MipsTargetMachine &tm) {
  return new Filler(tm);
}

bool Filler::findDelayInstr(MachineBasicBlock &MBB,
                            InstrIter slot,
                            InstrIter &Filler) {
  SmallSet<unsigned, 32> RegDefs;
  SmallSet<unsigned, 32> RegUses;

  insertDefsUses(slot, RegDefs, RegUses);

  bool sawLoad = false;
  bool sawStore = false;

  for (ReverseInstrIter I(slot); I != MBB.instr_rend(); ++I) {
    // skip debug value
    if (I->isDebugValue())
      continue;

    // Convert to forward iterator.
    InstrIter FI(llvm::next(I).base());

    if (I->hasUnmodeledSideEffects()
        || I->isInlineAsm()
        || I->isLabel()
        || FI == LastFiller
        || I->isPseudo()
        //
        // Should not allow:
        // ERET, DERET or WAIT, PAUSE. Need to add these to instruction
        // list. TBD.
        )
      break;

    if (delayHasHazard(FI, sawLoad, sawStore, RegDefs, RegUses)) {
      insertDefsUses(FI, RegDefs, RegUses);
      continue;
    }

    Filler = FI;
    return true;
  }

  return false;
}

bool Filler::delayHasHazard(InstrIter candidate,
                            bool &sawLoad, bool &sawStore,
                            SmallSet<unsigned, 32> &RegDefs,
                            SmallSet<unsigned, 32> &RegUses) {
  if (candidate->isImplicitDef() || candidate->isKill())
    return true;

  // Loads or stores cannot be moved past a store to the delay slot
  // and stores cannot be moved past a load.
  if (candidate->mayLoad()) {
    if (sawStore)
      return true;
    sawLoad = true;
  }

  if (candidate->mayStore()) {
    if (sawStore)
      return true;
    sawStore = true;
    if (sawLoad)
      return true;
  }

  assert((!candidate->isCall() && !candidate->isReturn()) &&
         "Cannot put calls or returns in delay slot.");

  for (unsigned i = 0, e = candidate->getNumOperands(); i!= e; ++i) {
    const MachineOperand &MO = candidate->getOperand(i);
    unsigned Reg;

    if (!MO.isReg() || !(Reg = MO.getReg()))
      continue; // skip

    if (MO.isDef()) {
      // check whether Reg is defined or used before delay slot.
      if (IsRegInSet(RegDefs, Reg) || IsRegInSet(RegUses, Reg))
        return true;
    }
    if (MO.isUse()) {
      // check whether Reg is defined before delay slot.
      if (IsRegInSet(RegDefs, Reg))
        return true;
    }
  }
  return false;
}

// Insert Defs and Uses of MI into the sets RegDefs and RegUses.
void Filler::insertDefsUses(InstrIter MI,
                            SmallSet<unsigned, 32> &RegDefs,
                            SmallSet<unsigned, 32> &RegUses) {
  // If MI is a call or return, just examine the explicit non-variadic operands.
  MCInstrDesc MCID = MI->getDesc();
  unsigned e = MI->isCall() || MI->isReturn() ? MCID.getNumOperands() :
                                                MI->getNumOperands();

  // Add RA to RegDefs to prevent users of RA from going into delay slot.
  if (MI->isCall())
    RegDefs.insert(Mips::RA);

  for (unsigned i = 0; i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    unsigned Reg;

    if (!MO.isReg() || !(Reg = MO.getReg()))
      continue;

    if (MO.isDef())
      RegDefs.insert(Reg);
    else if (MO.isUse())
      RegUses.insert(Reg);
  }
}

//returns true if the Reg or its alias is in the RegSet.
bool Filler::IsRegInSet(SmallSet<unsigned, 32> &RegSet, unsigned Reg) {
  // Check Reg and all aliased Registers.
  for (MCRegAliasIterator AI(Reg, TM.getRegisterInfo(), true);
       AI.isValid(); ++AI)
    if (RegSet.count(*AI))
      return true;
  return false;
}
