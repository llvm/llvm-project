//===-- MachineSink.cpp - Sinking for machine instructions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass 
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "machine-sink"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

STATISTIC(NumSunk, "Number of machine instructions sunk");

namespace {
  class VISIBILITY_HIDDEN MachineSinking : public MachineFunctionPass {
    const TargetMachine   *TM;
    const TargetInstrInfo *TII;
    MachineFunction       *CurMF; // Current MachineFunction
    MachineRegisterInfo  *RegInfo; // Machine register information
    MachineDominatorTree *DT;   // Machine dominator tree for the current Loop

  public:
    static char ID; // Pass identification
    MachineSinking() : MachineFunctionPass(&ID) {}
    
    virtual bool runOnMachineFunction(MachineFunction &MF);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      MachineFunctionPass::getAnalysisUsage(AU);
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
    }
  private:
    bool ProcessBlock(MachineBasicBlock &MBB);
    bool SinkInstruction(MachineInstr *MI, bool &SawStore);
    bool AllUsesDominatedByBlock(unsigned Reg, MachineBasicBlock *MBB) const;
  };
} // end anonymous namespace
  
char MachineSinking::ID = 0;
static RegisterPass<MachineSinking>
X("machine-sink", "Machine code sinking");

FunctionPass *llvm::createMachineSinkingPass() { return new MachineSinking(); }

/// AllUsesDominatedByBlock - Return true if all uses of the specified register
/// occur in blocks dominated by the specified block.
bool MachineSinking::AllUsesDominatedByBlock(unsigned Reg, 
                                             MachineBasicBlock *MBB) const {
  assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
         "Only makes sense for vregs");
  for (MachineRegisterInfo::reg_iterator I = RegInfo->reg_begin(Reg),
       E = RegInfo->reg_end(); I != E; ++I) {
    if (I.getOperand().isDef()) continue;  // ignore def.
    
    // Determine the block of the use.
    MachineInstr *UseInst = &*I;
    MachineBasicBlock *UseBlock = UseInst->getParent();
    if (UseInst->getOpcode() == TargetInstrInfo::PHI) {
      // PHI nodes use the operand in the predecessor block, not the block with
      // the PHI.
      UseBlock = UseInst->getOperand(I.getOperandNo()+1).getMBB();
    }
    // Check that it dominates.
    if (!DT->dominates(MBB, UseBlock))
      return false;
  }
  return true;
}



bool MachineSinking::runOnMachineFunction(MachineFunction &MF) {
  DOUT << "******** Machine Sinking ********\n";
  
  CurMF = &MF;
  TM = &CurMF->getTarget();
  TII = TM->getInstrInfo();
  RegInfo = &CurMF->getRegInfo();
  DT = &getAnalysis<MachineDominatorTree>();

  bool EverMadeChange = false;
  
  while (1) {
    bool MadeChange = false;

    // Process all basic blocks.
    for (MachineFunction::iterator I = CurMF->begin(), E = CurMF->end(); 
         I != E; ++I)
      MadeChange |= ProcessBlock(*I);
    
    // If this iteration over the code changed anything, keep iterating.
    if (!MadeChange) break;
    EverMadeChange = true;
  } 
  return EverMadeChange;
}

bool MachineSinking::ProcessBlock(MachineBasicBlock &MBB) {
  bool MadeChange = false;
  
  // Can't sink anything out of a block that has less than two successors.
  if (MBB.succ_size() <= 1) return false;
  
  // Walk the basic block bottom-up.  Remember if we saw a store.
  bool SawStore = false;
  for (MachineBasicBlock::iterator I = MBB.end(); I != MBB.begin(); ){
    MachineBasicBlock::iterator LastIt = I;
    if (SinkInstruction(--I, SawStore)) {
      I = LastIt;
      ++NumSunk;
    }
  }
  
  return MadeChange;
}

/// SinkInstruction - Determine whether it is safe to sink the specified machine
/// instruction out of its current block into a successor.
bool MachineSinking::SinkInstruction(MachineInstr *MI, bool &SawStore) {
  // Check if it's safe to move the instruction.
  if (!MI->isSafeToMove(TII, SawStore))
    return false;
  
  // FIXME: This should include support for sinking instructions within the
  // block they are currently in to shorten the live ranges.  We often get
  // instructions sunk into the top of a large block, but it would be better to
  // also sink them down before their first use in the block.  This xform has to
  // be careful not to *increase* register pressure though, e.g. sinking
  // "x = y + z" down if it kills y and z would increase the live ranges of y
  // and z only the shrink the live range of x.
  
  // Loop over all the operands of the specified instruction.  If there is
  // anything we can't handle, bail out.
  MachineBasicBlock *ParentBlock = MI->getParent();
  
  // SuccToSinkTo - This is the successor to sink this instruction to, once we
  // decide.
  MachineBasicBlock *SuccToSinkTo = 0;
  
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;  // Ignore non-register operands.
    
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    
    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      // If this is a physical register use, we can't move it.  If it is a def,
      // we can move it, but only if the def is dead.
      if (MO.isUse() || !MO.isDead())
        return false;
    } else {
      // Virtual register uses are always safe to sink.
      if (MO.isUse()) continue;
      
      // FIXME: This picks a successor to sink into based on having one
      // successor that dominates all the uses.  However, there are cases where
      // sinking can happen but where the sink point isn't a successor.  For
      // example:
      //   x = computation
      //   if () {} else {}
      //   use x
      // the instruction could be sunk over the whole diamond for the 
      // if/then/else (or loop, etc), allowing it to be sunk into other blocks
      // after that.
      
      // Virtual register defs can only be sunk if all their uses are in blocks
      // dominated by one of the successors.
      if (SuccToSinkTo) {
        // If a previous operand picked a block to sink to, then this operand
        // must be sinkable to the same block.
        if (!AllUsesDominatedByBlock(Reg, SuccToSinkTo)) 
          return false;
        continue;
      }
      
      // Otherwise, we should look at all the successors and decide which one
      // we should sink to.
      for (MachineBasicBlock::succ_iterator SI = ParentBlock->succ_begin(),
           E = ParentBlock->succ_end(); SI != E; ++SI) {
        if (AllUsesDominatedByBlock(Reg, *SI)) {
          SuccToSinkTo = *SI;
          break;
        }
      }
      
      // If we couldn't find a block to sink to, ignore this instruction.
      if (SuccToSinkTo == 0)
        return false;
    }
  }
  
  // If there are no outputs, it must have side-effects.
  if (SuccToSinkTo == 0)
    return false;

  // It's not safe to sink instructions to EH landing pad. Control flow into
  // landing pad is implicitly defined.
  if (SuccToSinkTo->isLandingPad())
    return false;
  
  DEBUG(cerr << "Sink instr " << *MI);
  DEBUG(cerr << "to block " << *SuccToSinkTo);
  
  // If the block has multiple predecessors, this would introduce computation on
  // a path that it doesn't already exist.  We could split the critical edge,
  // but for now we just punt.
  // FIXME: Split critical edges if not backedges.
  if (SuccToSinkTo->pred_size() > 1) {
    DEBUG(cerr << " *** PUNTING: Critical edge found\n");
    return false;
  }
  
  // Determine where to insert into.  Skip phi nodes.
  MachineBasicBlock::iterator InsertPos = SuccToSinkTo->begin();
  while (InsertPos != SuccToSinkTo->end() && 
         InsertPos->getOpcode() == TargetInstrInfo::PHI)
    ++InsertPos;
  
  // Move the instruction.
  SuccToSinkTo->splice(InsertPos, ParentBlock, MI,
                       ++MachineBasicBlock::iterator(MI));
  return true;
}
