//===-- PhiElimination.cpp - Eliminate PHI nodes by inserting copies ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates machine instruction PHI nodes by inserting copy
// instructions.  This destroys SSA information, but is the desired input for
// some register allocators.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "phielim"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
#include <map>
using namespace llvm;

STATISTIC(NumAtomic, "Number of atomic phis lowered");

namespace {
  class VISIBILITY_HIDDEN PNE : public MachineFunctionPass {
    MachineRegisterInfo  *MRI; // Machine register information

  public:
    static char ID; // Pass identification, replacement for typeid
    PNE() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &Fn);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<LiveVariables>();
      AU.addPreservedID(MachineLoopInfoID);
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    /// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions
    /// in predecessor basic blocks.
    ///
    bool EliminatePHINodes(MachineFunction &MF, MachineBasicBlock &MBB);
    void LowerAtomicPHINode(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator AfterPHIsIt);

    /// analyzePHINodes - Gather information about the PHI nodes in
    /// here. In particular, we want to map the number of uses of a virtual
    /// register which is used in a PHI node. We map that to the BB the
    /// vreg is coming from. This is used later to determine when the vreg
    /// is killed in the BB.
    ///
    void analyzePHINodes(const MachineFunction& Fn);

    typedef std::pair<const MachineBasicBlock*, unsigned> BBVRegPair;
    typedef std::map<BBVRegPair, unsigned> VRegPHIUse;

    VRegPHIUse VRegPHIUseCount;

    // Defs of PHI sources which are implicit_def.
    SmallPtrSet<MachineInstr*, 4> ImpDefs;
  };
}

char PNE::ID = 0;
static RegisterPass<PNE>
X("phi-node-elimination", "Eliminate PHI nodes for register allocation");

const PassInfo *const llvm::PHIEliminationID = &X;

bool PNE::runOnMachineFunction(MachineFunction &Fn) {
  MRI = &Fn.getRegInfo();

  analyzePHINodes(Fn);

  bool Changed = false;

  // Eliminate PHI instructions by inserting copies into predecessor blocks.
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    Changed |= EliminatePHINodes(Fn, *I);

  // Remove dead IMPLICIT_DEF instructions.
  for (SmallPtrSet<MachineInstr*,4>::iterator I = ImpDefs.begin(),
         E = ImpDefs.end(); I != E; ++I) {
    MachineInstr *DefMI = *I;
    unsigned DefReg = DefMI->getOperand(0).getReg();
    if (MRI->use_empty(DefReg))
      DefMI->eraseFromParent();
  }

  ImpDefs.clear();
  VRegPHIUseCount.clear();
  return Changed;
}


/// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions in
/// predecessor basic blocks.
///
bool PNE::EliminatePHINodes(MachineFunction &MF, MachineBasicBlock &MBB) {
  if (MBB.empty() || MBB.front().getOpcode() != TargetInstrInfo::PHI)
    return false;   // Quick exit for basic blocks without PHIs.

  // Get an iterator to the first instruction after the last PHI node (this may
  // also be the end of the basic block).
  MachineBasicBlock::iterator AfterPHIsIt = MBB.begin();
  while (AfterPHIsIt != MBB.end() &&
         AfterPHIsIt->getOpcode() == TargetInstrInfo::PHI)
    ++AfterPHIsIt;    // Skip over all of the PHI nodes...

  while (MBB.front().getOpcode() == TargetInstrInfo::PHI)
    LowerAtomicPHINode(MBB, AfterPHIsIt);

  return true;
}

/// isSourceDefinedByImplicitDef - Return true if all sources of the phi node
/// are implicit_def's.
static bool isSourceDefinedByImplicitDef(const MachineInstr *MPhi,
                                         const MachineRegisterInfo *MRI) {
  for (unsigned i = 1; i != MPhi->getNumOperands(); i += 2) {
    unsigned SrcReg = MPhi->getOperand(i).getReg();
    const MachineInstr *DefMI = MRI->getVRegDef(SrcReg);
    if (!DefMI || DefMI->getOpcode() != TargetInstrInfo::IMPLICIT_DEF)
      return false;
  }
  return true;
}

/// LowerAtomicPHINode - Lower the PHI node at the top of the specified block,
/// under the assuption that it needs to be lowered in a way that supports
/// atomic execution of PHIs.  This lowering method is always correct all of the
/// time.
/// 
void PNE::LowerAtomicPHINode(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator AfterPHIsIt) {
  // Unlink the PHI node from the basic block, but don't delete the PHI yet.
  MachineInstr *MPhi = MBB.remove(MBB.begin());

  unsigned NumSrcs = (MPhi->getNumOperands() - 1) / 2;
  unsigned DestReg = MPhi->getOperand(0).getReg();
  bool isDead = MPhi->getOperand(0).isDead();

  // Create a new register for the incoming PHI arguments.
  MachineFunction &MF = *MBB.getParent();
  const TargetRegisterClass *RC = MF.getRegInfo().getRegClass(DestReg);
  unsigned IncomingReg = 0;

  // Insert a register to register copy at the top of the current block (but
  // after any remaining phi nodes) which copies the new incoming register
  // into the phi node destination.
  const TargetInstrInfo *TII = MF.getTarget().getInstrInfo();
  if (isSourceDefinedByImplicitDef(MPhi, MRI))
    // If all sources of a PHI node are implicit_def, just emit an
    // implicit_def instead of a copy.
    BuildMI(MBB, AfterPHIsIt, MPhi->getDebugLoc(),
            TII->get(TargetInstrInfo::IMPLICIT_DEF), DestReg);
  else {
    IncomingReg = MF.getRegInfo().createVirtualRegister(RC);
    TII->copyRegToReg(MBB, AfterPHIsIt, DestReg, IncomingReg, RC, RC);
  }

  // Update live variable information if there is any.
  LiveVariables *LV = getAnalysisIfAvailable<LiveVariables>();
  if (LV) {
    MachineInstr *PHICopy = prior(AfterPHIsIt);

    if (IncomingReg) {
      // Increment use count of the newly created virtual register.
      LV->getVarInfo(IncomingReg).NumUses++;

      // Add information to LiveVariables to know that the incoming value is
      // killed.  Note that because the value is defined in several places (once
      // each for each incoming block), the "def" block and instruction fields
      // for the VarInfo is not filled in.
      LV->addVirtualRegisterKilled(IncomingReg, PHICopy);

      LV->getVarInfo(IncomingReg).UsedBlocks[MBB.getNumber()] = true;
    }

    // Since we are going to be deleting the PHI node, if it is the last use of
    // any registers, or if the value itself is dead, we need to move this
    // information over to the new copy we just inserted.
    LV->removeVirtualRegistersKilled(MPhi);

    // If the result is dead, update LV.
    if (isDead) {
      LV->addVirtualRegisterDead(DestReg, PHICopy);
      LV->removeVirtualRegisterDead(DestReg, MPhi);
    }
  }

  // Adjust the VRegPHIUseCount map to account for the removal of this PHI node.
  for (unsigned i = 1; i != MPhi->getNumOperands(); i += 2)
    --VRegPHIUseCount[BBVRegPair(MPhi->getOperand(i + 1).getMBB(),
                                 MPhi->getOperand(i).getReg())];

  // Now loop over all of the incoming arguments, changing them to copy into the
  // IncomingReg register in the corresponding predecessor basic block.
  SmallPtrSet<MachineBasicBlock*, 8> MBBsInsertedInto;
  for (int i = NumSrcs - 1; i >= 0; --i) {
    unsigned SrcReg = MPhi->getOperand(i*2+1).getReg();
    assert(TargetRegisterInfo::isVirtualRegister(SrcReg) &&
           "Machine PHI Operands must all be virtual registers!");

    // If source is defined by an implicit def, there is no need to insert a
    // copy.
    MachineInstr *DefMI = MRI->getVRegDef(SrcReg);
    if (DefMI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF) {
      ImpDefs.insert(DefMI);
      continue;
    }

    // Get the MachineBasicBlock equivalent of the BasicBlock that is the source
    // path the PHI.
    MachineBasicBlock &opBlock = *MPhi->getOperand(i*2+2).getMBB();

    // Check to make sure we haven't already emitted the copy for this block.
    // This can happen because PHI nodes may have multiple entries for the same
    // basic block.
    if (!MBBsInsertedInto.insert(&opBlock))
      continue;  // If the copy has already been emitted, we're done.
 
    // Find a safe location to insert the copy, this may be the first terminator
    // in the block (or end()).
    MachineBasicBlock::iterator InsertPos = opBlock.getFirstTerminator();

    // Insert the copy.
    TII->copyRegToReg(opBlock, InsertPos, IncomingReg, SrcReg, RC, RC);

    // Now update live variable information if we have it.  Otherwise we're done
    if (!LV) continue;
    
    // We want to be able to insert a kill of the register if this PHI (aka, the
    // copy we just inserted) is the last use of the source value.  Live
    // variable analysis conservatively handles this by saying that the value is
    // live until the end of the block the PHI entry lives in.  If the value
    // really is dead at the PHI copy, there will be no successor blocks which
    // have the value live-in.
    //
    // Check to see if the copy is the last use, and if so, update the live
    // variables information so that it knows the copy source instruction kills
    // the incoming value.
    LiveVariables::VarInfo &InRegVI = LV->getVarInfo(SrcReg);
    InRegVI.UsedBlocks[opBlock.getNumber()] = true;

    // Loop over all of the successors of the basic block, checking to see if
    // the value is either live in the block, or if it is killed in the block.
    // Also check to see if this register is in use by another PHI node which
    // has not yet been eliminated.  If so, it will be killed at an appropriate
    // point later.

    // Is it used by any PHI instructions in this block?
    bool ValueIsLive = VRegPHIUseCount[BBVRegPair(&opBlock, SrcReg)] != 0;

    std::vector<MachineBasicBlock*> OpSuccBlocks;
    
    // Otherwise, scan successors, including the BB the PHI node lives in.
    for (MachineBasicBlock::succ_iterator SI = opBlock.succ_begin(),
           E = opBlock.succ_end(); SI != E && !ValueIsLive; ++SI) {
      MachineBasicBlock *SuccMBB = *SI;

      // Is it alive in this successor?
      unsigned SuccIdx = SuccMBB->getNumber();
      if (SuccIdx < InRegVI.AliveBlocks.size() &&
          InRegVI.AliveBlocks[SuccIdx]) {
        ValueIsLive = true;
        break;
      }

      OpSuccBlocks.push_back(SuccMBB);
    }

    // Check to see if this value is live because there is a use in a successor
    // that kills it.
    if (!ValueIsLive) {
      switch (OpSuccBlocks.size()) {
      case 1: {
        MachineBasicBlock *MBB = OpSuccBlocks[0];
        for (unsigned i = 0, e = InRegVI.Kills.size(); i != e; ++i)
          if (InRegVI.Kills[i]->getParent() == MBB) {
            ValueIsLive = true;
            break;
          }
        break;
      }
      case 2: {
        MachineBasicBlock *MBB1 = OpSuccBlocks[0], *MBB2 = OpSuccBlocks[1];
        for (unsigned i = 0, e = InRegVI.Kills.size(); i != e; ++i)
          if (InRegVI.Kills[i]->getParent() == MBB1 || 
              InRegVI.Kills[i]->getParent() == MBB2) {
            ValueIsLive = true;
            break;
          }
        break;        
      }
      default:
        std::sort(OpSuccBlocks.begin(), OpSuccBlocks.end());
        for (unsigned i = 0, e = InRegVI.Kills.size(); i != e; ++i)
          if (std::binary_search(OpSuccBlocks.begin(), OpSuccBlocks.end(),
                                 InRegVI.Kills[i]->getParent())) {
            ValueIsLive = true;
            break;
          }
      }
    }        

    // Okay, if we now know that the value is not live out of the block, we can
    // add a kill marker in this block saying that it kills the incoming value!
    if (!ValueIsLive) {
      // In our final twist, we have to decide which instruction kills the
      // register.  In most cases this is the copy, however, the first
      // terminator instruction at the end of the block may also use the value.
      // In this case, we should mark *it* as being the killing block, not the
      // copy.
      MachineBasicBlock::iterator KillInst = prior(InsertPos);
      MachineBasicBlock::iterator Term = opBlock.getFirstTerminator();
      if (Term != opBlock.end()) {
        if (Term->readsRegister(SrcReg))
          KillInst = Term;
      
        // Check that no other terminators use values.
#ifndef NDEBUG
        for (MachineBasicBlock::iterator TI = next(Term); TI != opBlock.end();
             ++TI) {
          assert(!TI->readsRegister(SrcReg) &&
                 "Terminator instructions cannot use virtual registers unless"
                 "they are the first terminator in a block!");
        }
#endif
      }
      
      // Finally, mark it killed.
      LV->addVirtualRegisterKilled(SrcReg, KillInst);

      // This vreg no longer lives all of the way through opBlock.
      unsigned opBlockNum = opBlock.getNumber();
      if (opBlockNum < InRegVI.AliveBlocks.size())
        InRegVI.AliveBlocks[opBlockNum] = false;
    }
  }
    
  // Really delete the PHI instruction now!
  MF.DeleteMachineInstr(MPhi);
  ++NumAtomic;
}

/// analyzePHINodes - Gather information about the PHI nodes in here. In
/// particular, we want to map the number of uses of a virtual register which is
/// used in a PHI node. We map that to the BB the vreg is coming from. This is
/// used later to determine when the vreg is killed in the BB.
///
void PNE::analyzePHINodes(const MachineFunction& Fn) {
  for (MachineFunction::const_iterator I = Fn.begin(), E = Fn.end();
       I != E; ++I)
    for (MachineBasicBlock::const_iterator BBI = I->begin(), BBE = I->end();
         BBI != BBE && BBI->getOpcode() == TargetInstrInfo::PHI; ++BBI)
      for (unsigned i = 1, e = BBI->getNumOperands(); i != e; i += 2)
        ++VRegPHIUseCount[BBVRegPair(BBI->getOperand(i + 1).getMBB(),
                                     BBI->getOperand(i).getReg())];
}
