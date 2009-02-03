//===-- IA64Bundling.cpp - IA-64 instruction bundling pass. ------------ --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Add stops where required to prevent read-after-write and write-after-write
// dependencies, for both registers and memory addresses. There are exceptions:
//
//    - Compare instructions (cmp*, tbit, tnat, fcmp, frcpa) are OK with
//      WAW dependencies so long as they all target p0, or are of parallel
//      type (.and*/.or*)
//
// FIXME: bundling, for now, is left to the assembler.
// FIXME: this might be an appropriate place to translate between different
//        instructions that do the same thing, if this helps bundling.
// 
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ia64-codegen"
#include "IA64.h"
#include "IA64InstrInfo.h"
#include "IA64TargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include <set>
using namespace llvm;

STATISTIC(StopBitsAdded, "Number of stop bits added");

namespace {
  struct IA64BundlingPass : public MachineFunctionPass {
    static char ID;
    /// Target machine description which we query for reg. names, data
    /// layout, etc.
    ///
    IA64TargetMachine &TM;

    IA64BundlingPass(IA64TargetMachine &tm) 
      : MachineFunctionPass(&ID), TM(tm) { }

    virtual const char *getPassName() const {
      return "IA64 (Itanium) Bundling Pass";
    }

    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
    bool runOnMachineFunction(MachineFunction &F) {
      bool Changed = false;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI)
        Changed |= runOnMachineBasicBlock(*FI);
      return Changed;
    }

    // XXX: ugly global, but pending writes can cross basic blocks. Note that
    // taken branches end instruction groups. So we only need to worry about
    // 'fallthrough' code
    std::set<unsigned> PendingRegWrites;
  };
  char IA64BundlingPass::ID = 0;
} // end of anonymous namespace

/// createIA64BundlingPass - Returns a pass that adds STOP (;;) instructions
/// and arranges the result into bundles.
///
FunctionPass *llvm::createIA64BundlingPass(IA64TargetMachine &tm) {
  return new IA64BundlingPass(tm);
}

/// runOnMachineBasicBlock - add stops and bundle this MBB.
///
bool IA64BundlingPass::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ) {
    MachineInstr *CurrentInsn = I++;
    std::set<unsigned> CurrentReads, CurrentWrites, OrigWrites;

    for(unsigned i=0; i < CurrentInsn->getNumOperands(); i++) {
      MachineOperand &MO=CurrentInsn->getOperand(i);
      if (MO.isReg()) {
        if(MO.isUse()) { // TODO: exclude p0
          CurrentReads.insert(MO.getReg());
        }
        if(MO.isDef()) { // TODO: exclude p0
          CurrentWrites.insert(MO.getReg());
          OrigWrites.insert(MO.getReg()); // FIXME: use a nondestructive
                                          // set_intersect instead?
        }
      }
    }
    
    // CurrentReads/CurrentWrites contain info for the current instruction.
    // Does it read or write any registers that are pending a write?
    // (i.e. not separated by a stop)
    set_intersect(CurrentReads, PendingRegWrites);
    set_intersect(CurrentWrites, PendingRegWrites);
    
    if(! (CurrentReads.empty() && CurrentWrites.empty()) ) {
      // there is a conflict, insert a stop and reset PendingRegWrites
      CurrentInsn = BuildMI(MBB, CurrentInsn,
                            TM.getInstrInfo()->get(IA64::STOP), 0);
      PendingRegWrites=OrigWrites; // carry over current writes to next insn
      Changed=true; StopBitsAdded++; // update stats      
    } else { // otherwise, track additional pending writes
      set_union(PendingRegWrites, OrigWrites);
    }
  } // onto the next insn in the MBB

  return Changed;
}

