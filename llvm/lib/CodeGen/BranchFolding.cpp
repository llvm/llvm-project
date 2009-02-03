//===-- BranchFolding.cpp - Fold machine code branch instructions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass forwards branches to unconditional branches to make them branch
// directly to the target block.  This pass often results in dead MBB's, which
// it then removes.
//
// Note that this pass must be run after register allocation, it cannot handle
// SSA form.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "branchfolding"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumDeadBlocks, "Number of dead blocks removed");
STATISTIC(NumBranchOpts, "Number of branches optimized");
STATISTIC(NumTailMerge , "Number of block tails merged");
static cl::opt<cl::boolOrDefault> FlagEnableTailMerge("enable-tail-merge", 
                              cl::init(cl::BOU_UNSET), cl::Hidden);
// Throttle for huge numbers of predecessors (compile speed problems)
static cl::opt<unsigned>
TailMergeThreshold("tail-merge-threshold", 
          cl::desc("Max number of predecessors to consider tail merging"),
          cl::init(150), cl::Hidden);

namespace {
  struct VISIBILITY_HIDDEN BranchFolder : public MachineFunctionPass {
    static char ID;
    explicit BranchFolder(bool defaultEnableTailMerge) : 
      MachineFunctionPass(&ID) {
      switch (FlagEnableTailMerge) {
        case cl::BOU_UNSET: EnableTailMerge = defaultEnableTailMerge; break;
        case cl::BOU_TRUE: EnableTailMerge = true; break;
        case cl::BOU_FALSE: EnableTailMerge = false; break;
      }
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const { return "Control Flow Optimizer"; }
    const TargetInstrInfo *TII;
    MachineModuleInfo *MMI;
    bool MadeChange;
  private:
    // Tail Merging.
    bool EnableTailMerge;
    bool TailMergeBlocks(MachineFunction &MF);
    bool TryMergeBlocks(MachineBasicBlock* SuccBB,
                        MachineBasicBlock* PredBB);
    void ReplaceTailWithBranchTo(MachineBasicBlock::iterator OldInst,
                                 MachineBasicBlock *NewDest);
    MachineBasicBlock *SplitMBBAt(MachineBasicBlock &CurMBB,
                                  MachineBasicBlock::iterator BBI1);
    unsigned ComputeSameTails(unsigned CurHash, unsigned minCommonTailLength);
    void RemoveBlocksWithHash(unsigned CurHash, MachineBasicBlock* SuccBB,
                                                MachineBasicBlock* PredBB);
    unsigned CreateCommonTailOnlyBlock(MachineBasicBlock *&PredBB,
                                       unsigned maxCommonTailLength);

    typedef std::pair<unsigned,MachineBasicBlock*> MergePotentialsElt;
    typedef std::vector<MergePotentialsElt>::iterator MPIterator;
    std::vector<MergePotentialsElt> MergePotentials;

    typedef std::pair<MPIterator, MachineBasicBlock::iterator> SameTailElt;
    std::vector<SameTailElt> SameTails;

    const TargetRegisterInfo *RegInfo;
    RegScavenger *RS;
    // Branch optzn.
    bool OptimizeBranches(MachineFunction &MF);
    void OptimizeBlock(MachineBasicBlock *MBB);
    void RemoveDeadBlock(MachineBasicBlock *MBB);
    bool OptimizeImpDefsBlock(MachineBasicBlock *MBB);
    
    bool CanFallThrough(MachineBasicBlock *CurBB);
    bool CanFallThrough(MachineBasicBlock *CurBB, bool BranchUnAnalyzable,
                        MachineBasicBlock *TBB, MachineBasicBlock *FBB,
                        const SmallVectorImpl<MachineOperand> &Cond);
  };
  char BranchFolder::ID = 0;
}

FunctionPass *llvm::createBranchFoldingPass(bool DefaultEnableTailMerge) { 
      return new BranchFolder(DefaultEnableTailMerge); }

/// RemoveDeadBlock - Remove the specified dead machine basic block from the
/// function, updating the CFG.
void BranchFolder::RemoveDeadBlock(MachineBasicBlock *MBB) {
  assert(MBB->pred_empty() && "MBB must be dead!");
  DOUT << "\nRemoving MBB: " << *MBB;
  
  MachineFunction *MF = MBB->getParent();
  // drop all successors.
  while (!MBB->succ_empty())
    MBB->removeSuccessor(MBB->succ_end()-1);
  
  // If there are any labels in the basic block, unregister them from
  // MachineModuleInfo.
  if (MMI && !MBB->empty()) {
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      if (I->isLabel())
        // The label ID # is always operand #0, an immediate.
        MMI->InvalidateLabel(I->getOperand(0).getImm());
    }
  }
  
  // Remove the block.
  MF->erase(MBB);
}

/// OptimizeImpDefsBlock - If a basic block is just a bunch of implicit_def
/// followed by terminators, and if the implicitly defined registers are not
/// used by the terminators, remove those implicit_def's. e.g.
/// BB1:
///   r0 = implicit_def
///   r1 = implicit_def
///   br
/// This block can be optimized away later if the implicit instructions are
/// removed.
bool BranchFolder::OptimizeImpDefsBlock(MachineBasicBlock *MBB) {
  SmallSet<unsigned, 4> ImpDefRegs;
  MachineBasicBlock::iterator I = MBB->begin();
  while (I != MBB->end()) {
    if (I->getOpcode() != TargetInstrInfo::IMPLICIT_DEF)
      break;
    unsigned Reg = I->getOperand(0).getReg();
    ImpDefRegs.insert(Reg);
    for (const unsigned *SubRegs = RegInfo->getSubRegisters(Reg);
         unsigned SubReg = *SubRegs; ++SubRegs)
      ImpDefRegs.insert(SubReg);
    ++I;
  }
  if (ImpDefRegs.empty())
    return false;

  MachineBasicBlock::iterator FirstTerm = I;
  while (I != MBB->end()) {
    if (!TII->isUnpredicatedTerminator(I))
      return false;
    // See if it uses any of the implicitly defined registers.
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = I->getOperand(i);
      if (!MO.isReg() || !MO.isUse())
        continue;
      unsigned Reg = MO.getReg();
      if (ImpDefRegs.count(Reg))
        return false;
    }
    ++I;
  }

  I = MBB->begin();
  while (I != FirstTerm) {
    MachineInstr *ImpDefMI = &*I;
    ++I;
    MBB->erase(ImpDefMI);
  }

  return true;
}

bool BranchFolder::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  if (!TII) return false;

  RegInfo = MF.getTarget().getRegisterInfo();

  // Fix CFG.  The later algorithms expect it to be right.
  bool EverMadeChange = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; I++) {
    MachineBasicBlock *MBB = I, *TBB = 0, *FBB = 0;
    SmallVector<MachineOperand, 4> Cond;
    if (!TII->AnalyzeBranch(*MBB, TBB, FBB, Cond))
      EverMadeChange |= MBB->CorrectExtraCFGEdges(TBB, FBB, !Cond.empty());
    EverMadeChange |= OptimizeImpDefsBlock(MBB);
  }

  RS = RegInfo->requiresRegisterScavenging(MF) ? new RegScavenger() : NULL;

  MMI = getAnalysisIfAvailable<MachineModuleInfo>();

  bool MadeChangeThisIteration = true;
  while (MadeChangeThisIteration) {
    MadeChangeThisIteration = false;
    MadeChangeThisIteration |= TailMergeBlocks(MF);
    MadeChangeThisIteration |= OptimizeBranches(MF);
    EverMadeChange |= MadeChangeThisIteration;
  }

  // See if any jump tables have become mergable or dead as the code generator
  // did its thing.
  MachineJumpTableInfo *JTI = MF.getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JTs = JTI->getJumpTables();
  if (!JTs.empty()) {
    // Figure out how these jump tables should be merged.
    std::vector<unsigned> JTMapping;
    JTMapping.reserve(JTs.size());
    
    // We always keep the 0th jump table.
    JTMapping.push_back(0);

    // Scan the jump tables, seeing if there are any duplicates.  Note that this
    // is N^2, which should be fixed someday.
    for (unsigned i = 1, e = JTs.size(); i != e; ++i)
      JTMapping.push_back(JTI->getJumpTableIndex(JTs[i].MBBs));
    
    // If a jump table was merge with another one, walk the function rewriting
    // references to jump tables to reference the new JT ID's.  Keep track of
    // whether we see a jump table idx, if not, we can delete the JT.
    BitVector JTIsLive(JTs.size());
    for (MachineFunction::iterator BB = MF.begin(), E = MF.end();
         BB != E; ++BB) {
      for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end();
           I != E; ++I)
        for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op) {
          MachineOperand &Op = I->getOperand(op);
          if (!Op.isJTI()) continue;
          unsigned NewIdx = JTMapping[Op.getIndex()];
          Op.setIndex(NewIdx);

          // Remember that this JT is live.
          JTIsLive.set(NewIdx);
        }
    }
   
    // Finally, remove dead jump tables.  This happens either because the
    // indirect jump was unreachable (and thus deleted) or because the jump
    // table was merged with some other one.
    for (unsigned i = 0, e = JTIsLive.size(); i != e; ++i)
      if (!JTIsLive.test(i)) {
        JTI->RemoveJumpTable(i);
        EverMadeChange = true;
      }
  }
  
  delete RS;
  return EverMadeChange;
}

//===----------------------------------------------------------------------===//
//  Tail Merging of Blocks
//===----------------------------------------------------------------------===//

/// HashMachineInstr - Compute a hash value for MI and its operands.
static unsigned HashMachineInstr(const MachineInstr *MI) {
  unsigned Hash = MI->getOpcode();
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &Op = MI->getOperand(i);
    
    // Merge in bits from the operand if easy.
    unsigned OperandHash = 0;
    switch (Op.getType()) {
    case MachineOperand::MO_Register:          OperandHash = Op.getReg(); break;
    case MachineOperand::MO_Immediate:         OperandHash = Op.getImm(); break;
    case MachineOperand::MO_MachineBasicBlock:
      OperandHash = Op.getMBB()->getNumber();
      break;
    case MachineOperand::MO_FrameIndex:
    case MachineOperand::MO_ConstantPoolIndex:
    case MachineOperand::MO_JumpTableIndex:
      OperandHash = Op.getIndex();
      break;
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_ExternalSymbol:
      // Global address / external symbol are too hard, don't bother, but do
      // pull in the offset.
      OperandHash = Op.getOffset();
      break;
    default: break;
    }
    
    Hash += ((OperandHash << 3) | Op.getType()) << (i&31);
  }
  return Hash;
}

/// HashEndOfMBB - Hash the last few instructions in the MBB.  For blocks
/// with no successors, we hash two instructions, because cross-jumping 
/// only saves code when at least two instructions are removed (since a 
/// branch must be inserted).  For blocks with a successor, one of the
/// two blocks to be tail-merged will end with a branch already, so
/// it gains to cross-jump even for one instruction.

static unsigned HashEndOfMBB(const MachineBasicBlock *MBB,
                             unsigned minCommonTailLength) {
  MachineBasicBlock::const_iterator I = MBB->end();
  if (I == MBB->begin())
    return 0;   // Empty MBB.
  
  --I;
  unsigned Hash = HashMachineInstr(I);
    
  if (I == MBB->begin() || minCommonTailLength == 1)
    return Hash;   // Single instr MBB.
  
  --I;
  // Hash in the second-to-last instruction.
  Hash ^= HashMachineInstr(I) << 2;
  return Hash;
}

/// ComputeCommonTailLength - Given two machine basic blocks, compute the number
/// of instructions they actually have in common together at their end.  Return
/// iterators for the first shared instruction in each block.
static unsigned ComputeCommonTailLength(MachineBasicBlock *MBB1,
                                        MachineBasicBlock *MBB2,
                                        MachineBasicBlock::iterator &I1,
                                        MachineBasicBlock::iterator &I2) {
  I1 = MBB1->end();
  I2 = MBB2->end();
  
  unsigned TailLen = 0;
  while (I1 != MBB1->begin() && I2 != MBB2->begin()) {
    --I1; --I2;
    if (!I1->isIdenticalTo(I2) || 
        // FIXME: This check is dubious. It's used to get around a problem where
        // people incorrectly expect inline asm directives to remain in the same
        // relative order. This is untenable because normal compiler
        // optimizations (like this one) may reorder and/or merge these
        // directives.
        I1->getOpcode() == TargetInstrInfo::INLINEASM) {
      ++I1; ++I2;
      break;
    }
    ++TailLen;
  }
  return TailLen;
}

/// ReplaceTailWithBranchTo - Delete the instruction OldInst and everything
/// after it, replacing it with an unconditional branch to NewDest.  This
/// returns true if OldInst's block is modified, false if NewDest is modified.
void BranchFolder::ReplaceTailWithBranchTo(MachineBasicBlock::iterator OldInst,
                                           MachineBasicBlock *NewDest) {
  MachineBasicBlock *OldBB = OldInst->getParent();
  
  // Remove all the old successors of OldBB from the CFG.
  while (!OldBB->succ_empty())
    OldBB->removeSuccessor(OldBB->succ_begin());
  
  // Remove all the dead instructions from the end of OldBB.
  OldBB->erase(OldInst, OldBB->end());

  // If OldBB isn't immediately before OldBB, insert a branch to it.
  if (++MachineFunction::iterator(OldBB) != MachineFunction::iterator(NewDest))
    TII->InsertBranch(*OldBB, NewDest, 0, SmallVector<MachineOperand, 0>());
  OldBB->addSuccessor(NewDest);
  ++NumTailMerge;
}

/// SplitMBBAt - Given a machine basic block and an iterator into it, split the
/// MBB so that the part before the iterator falls into the part starting at the
/// iterator.  This returns the new MBB.
MachineBasicBlock *BranchFolder::SplitMBBAt(MachineBasicBlock &CurMBB,
                                            MachineBasicBlock::iterator BBI1) {
  MachineFunction &MF = *CurMBB.getParent();

  // Create the fall-through block.
  MachineFunction::iterator MBBI = &CurMBB;
  MachineBasicBlock *NewMBB =MF.CreateMachineBasicBlock(CurMBB.getBasicBlock());
  CurMBB.getParent()->insert(++MBBI, NewMBB);

  // Move all the successors of this block to the specified block.
  NewMBB->transferSuccessors(&CurMBB);
 
  // Add an edge from CurMBB to NewMBB for the fall-through.
  CurMBB.addSuccessor(NewMBB);
  
  // Splice the code over.
  NewMBB->splice(NewMBB->end(), &CurMBB, BBI1, CurMBB.end());

  // For targets that use the register scavenger, we must maintain LiveIns.
  if (RS) {
    RS->enterBasicBlock(&CurMBB);
    if (!CurMBB.empty())
      RS->forward(prior(CurMBB.end()));
    BitVector RegsLiveAtExit(RegInfo->getNumRegs());
    RS->getRegsUsed(RegsLiveAtExit, false);
    for (unsigned int i=0, e=RegInfo->getNumRegs(); i!=e; i++)
      if (RegsLiveAtExit[i])
        NewMBB->addLiveIn(i);
  }

  return NewMBB;
}

/// EstimateRuntime - Make a rough estimate for how long it will take to run
/// the specified code.
static unsigned EstimateRuntime(MachineBasicBlock::iterator I,
                                MachineBasicBlock::iterator E) {
  unsigned Time = 0;
  for (; I != E; ++I) {
    const TargetInstrDesc &TID = I->getDesc();
    if (TID.isCall())
      Time += 10;
    else if (TID.mayLoad() || TID.mayStore())
      Time += 2;
    else
      ++Time;
  }
  return Time;
}

// CurMBB needs to add an unconditional branch to SuccMBB (we removed these
// branches temporarily for tail merging).  In the case where CurMBB ends
// with a conditional branch to the next block, optimize by reversing the
// test and conditionally branching to SuccMBB instead.

static void FixTail(MachineBasicBlock* CurMBB, MachineBasicBlock *SuccBB,
                    const TargetInstrInfo *TII) {
  MachineFunction *MF = CurMBB->getParent();
  MachineFunction::iterator I = next(MachineFunction::iterator(CurMBB));
  MachineBasicBlock *TBB = 0, *FBB = 0;
  SmallVector<MachineOperand, 4> Cond;
  if (I != MF->end() &&
      !TII->AnalyzeBranch(*CurMBB, TBB, FBB, Cond)) {
    MachineBasicBlock *NextBB = I;
    if (TBB == NextBB && !Cond.empty() && !FBB) {
      if (!TII->ReverseBranchCondition(Cond)) {
        TII->RemoveBranch(*CurMBB);
        TII->InsertBranch(*CurMBB, SuccBB, NULL, Cond);
        return;
      }
    }
  }
  TII->InsertBranch(*CurMBB, SuccBB, NULL, SmallVector<MachineOperand, 0>());
}

static bool MergeCompare(const std::pair<unsigned,MachineBasicBlock*> &p,
                         const std::pair<unsigned,MachineBasicBlock*> &q) {
    if (p.first < q.first)
      return true;
     else if (p.first > q.first)
      return false;
    else if (p.second->getNumber() < q.second->getNumber())
      return true;
    else if (p.second->getNumber() > q.second->getNumber())
      return false;
    else {
      // _GLIBCXX_DEBUG checks strict weak ordering, which involves comparing
      // an object with itself.
#ifndef _GLIBCXX_DEBUG
      assert(0 && "Predecessor appears twice");
#endif
      return false;
    }
}

/// ComputeSameTails - Look through all the blocks in MergePotentials that have
/// hash CurHash (guaranteed to match the last element).   Build the vector 
/// SameTails of all those that have the (same) largest number of instructions
/// in common of any pair of these blocks.  SameTails entries contain an
/// iterator into MergePotentials (from which the MachineBasicBlock can be 
/// found) and a MachineBasicBlock::iterator into that MBB indicating the 
/// instruction where the matching code sequence begins.
/// Order of elements in SameTails is the reverse of the order in which
/// those blocks appear in MergePotentials (where they are not necessarily
/// consecutive).
unsigned BranchFolder::ComputeSameTails(unsigned CurHash, 
                                        unsigned minCommonTailLength) {
  unsigned maxCommonTailLength = 0U;
  SameTails.clear();
  MachineBasicBlock::iterator TrialBBI1, TrialBBI2;
  MPIterator HighestMPIter = prior(MergePotentials.end());
  for (MPIterator CurMPIter = prior(MergePotentials.end()),
                  B = MergePotentials.begin(); 
       CurMPIter!=B && CurMPIter->first==CurHash;
       --CurMPIter) {
    for (MPIterator I = prior(CurMPIter); I->first==CurHash ; --I) {
      unsigned CommonTailLen = ComputeCommonTailLength(
                                        CurMPIter->second,
                                        I->second,
                                        TrialBBI1, TrialBBI2);
      // If we will have to split a block, there should be at least
      // minCommonTailLength instructions in common; if not, at worst
      // we will be replacing a fallthrough into the common tail with a
      // branch, which at worst breaks even with falling through into
      // the duplicated common tail, so 1 instruction in common is enough.
      // We will always pick a block we do not have to split as the common
      // tail if there is one.
      // (Empty blocks will get forwarded and need not be considered.)
      if (CommonTailLen >= minCommonTailLength ||
          (CommonTailLen > 0 &&
           (TrialBBI1==CurMPIter->second->begin() ||
            TrialBBI2==I->second->begin()))) {
        if (CommonTailLen > maxCommonTailLength) {
          SameTails.clear();
          maxCommonTailLength = CommonTailLen;
          HighestMPIter = CurMPIter;
          SameTails.push_back(std::make_pair(CurMPIter, TrialBBI1));
        }
        if (HighestMPIter == CurMPIter &&
            CommonTailLen == maxCommonTailLength)
          SameTails.push_back(std::make_pair(I, TrialBBI2));
      }
      if (I==B)
        break;
    }
  }
  return maxCommonTailLength;
}

/// RemoveBlocksWithHash - Remove all blocks with hash CurHash from
/// MergePotentials, restoring branches at ends of blocks as appropriate.
void BranchFolder::RemoveBlocksWithHash(unsigned CurHash, 
                                        MachineBasicBlock* SuccBB,
                                        MachineBasicBlock* PredBB) {
  MPIterator CurMPIter, B;
  for (CurMPIter = prior(MergePotentials.end()), B = MergePotentials.begin(); 
       CurMPIter->first==CurHash;
       --CurMPIter) {
    // Put the unconditional branch back, if we need one.
    MachineBasicBlock *CurMBB = CurMPIter->second;
    if (SuccBB && CurMBB != PredBB)
      FixTail(CurMBB, SuccBB, TII);
    if (CurMPIter==B)
      break;
  }
  if (CurMPIter->first!=CurHash)
    CurMPIter++;
  MergePotentials.erase(CurMPIter, MergePotentials.end());
}

/// CreateCommonTailOnlyBlock - None of the blocks to be tail-merged consist
/// only of the common tail.  Create a block that does by splitting one.
unsigned BranchFolder::CreateCommonTailOnlyBlock(MachineBasicBlock *&PredBB,
                                             unsigned maxCommonTailLength) {
  unsigned i, commonTailIndex;
  unsigned TimeEstimate = ~0U;
  for (i=0, commonTailIndex=0; i<SameTails.size(); i++) {
    // Use PredBB if possible; that doesn't require a new branch.
    if (SameTails[i].first->second==PredBB) {
      commonTailIndex = i;
      break;
    }
    // Otherwise, make a (fairly bogus) choice based on estimate of
    // how long it will take the various blocks to execute.
    unsigned t = EstimateRuntime(SameTails[i].first->second->begin(), 
                                 SameTails[i].second);
    if (t<=TimeEstimate) {
      TimeEstimate = t;
      commonTailIndex = i;
    }
  }

  MachineBasicBlock::iterator BBI = SameTails[commonTailIndex].second;
  MachineBasicBlock *MBB = SameTails[commonTailIndex].first->second;

  DOUT << "\nSplitting " << MBB->getNumber() << ", size " << 
          maxCommonTailLength;

  MachineBasicBlock *newMBB = SplitMBBAt(*MBB, BBI);
  SameTails[commonTailIndex].first->second = newMBB;
  SameTails[commonTailIndex].second = newMBB->begin();
  // If we split PredBB, newMBB is the new predecessor.
  if (PredBB==MBB)
    PredBB = newMBB;

  return commonTailIndex;
}

// See if any of the blocks in MergePotentials (which all have a common single
// successor, or all have no successor) can be tail-merged.  If there is a
// successor, any blocks in MergePotentials that are not tail-merged and
// are not immediately before Succ must have an unconditional branch to
// Succ added (but the predecessor/successor lists need no adjustment).  
// The lone predecessor of Succ that falls through into Succ,
// if any, is given in PredBB.

bool BranchFolder::TryMergeBlocks(MachineBasicBlock *SuccBB,
                                  MachineBasicBlock* PredBB) {
  // It doesn't make sense to save a single instruction since tail merging
  // will add a jump.
  // FIXME: Ask the target to provide the threshold?
  unsigned minCommonTailLength = (SuccBB ? 1 : 2) + 1;
  MadeChange = false;
  
  DOUT << "\nTryMergeBlocks " << MergePotentials.size() << '\n';

  // Sort by hash value so that blocks with identical end sequences sort
  // together.
  std::stable_sort(MergePotentials.begin(), MergePotentials.end(),MergeCompare);

  // Walk through equivalence sets looking for actual exact matches.
  while (MergePotentials.size() > 1) {
    unsigned CurHash  = prior(MergePotentials.end())->first;
    
    // Build SameTails, identifying the set of blocks with this hash code
    // and with the maximum number of instructions in common.
    unsigned maxCommonTailLength = ComputeSameTails(CurHash, 
                                                    minCommonTailLength);

    // If we didn't find any pair that has at least minCommonTailLength 
    // instructions in common, remove all blocks with this hash code and retry.
    if (SameTails.empty()) {
      RemoveBlocksWithHash(CurHash, SuccBB, PredBB);
      continue;
    }

    // If one of the blocks is the entire common tail (and not the entry
    // block, which we can't jump to), we can treat all blocks with this same
    // tail at once.  Use PredBB if that is one of the possibilities, as that
    // will not introduce any extra branches.
    MachineBasicBlock *EntryBB = MergePotentials.begin()->second->
                                getParent()->begin();
    unsigned int commonTailIndex, i;
    for (commonTailIndex=SameTails.size(), i=0; i<SameTails.size(); i++) {
      MachineBasicBlock *MBB = SameTails[i].first->second;
      if (MBB->begin() == SameTails[i].second && MBB != EntryBB) {
        commonTailIndex = i;
        if (MBB==PredBB)
          break;
      }
    }

    if (commonTailIndex==SameTails.size()) {
      // None of the blocks consist entirely of the common tail.
      // Split a block so that one does.
      commonTailIndex = CreateCommonTailOnlyBlock(PredBB,  maxCommonTailLength);
    }

    MachineBasicBlock *MBB = SameTails[commonTailIndex].first->second;
    // MBB is common tail.  Adjust all other BB's to jump to this one.
    // Traversal must be forwards so erases work.
    DOUT << "\nUsing common tail " << MBB->getNumber() << " for ";
    for (unsigned int i=0; i<SameTails.size(); ++i) {
      if (commonTailIndex==i)
        continue;
      DOUT << SameTails[i].first->second->getNumber() << ",";
      // Hack the end off BB i, making it jump to BB commonTailIndex instead.
      ReplaceTailWithBranchTo(SameTails[i].second, MBB);
      // BB i is no longer a predecessor of SuccBB; remove it from the worklist.
      MergePotentials.erase(SameTails[i].first);
    }
    DOUT << "\n";
    // We leave commonTailIndex in the worklist in case there are other blocks
    // that match it with a smaller number of instructions.
    MadeChange = true;
  }
  return MadeChange;
}

bool BranchFolder::TailMergeBlocks(MachineFunction &MF) {

  if (!EnableTailMerge) return false;
 
  MadeChange = false;

  // First find blocks with no successors.
  MergePotentials.clear();
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    if (I->succ_empty())
      MergePotentials.push_back(std::make_pair(HashEndOfMBB(I, 2U), I));
  }
  // See if we can do any tail merging on those.
  if (MergePotentials.size() < TailMergeThreshold &&
      MergePotentials.size() >= 2)
    MadeChange |= TryMergeBlocks(NULL, NULL);

  // Look at blocks (IBB) with multiple predecessors (PBB).
  // We change each predecessor to a canonical form, by
  // (1) temporarily removing any unconditional branch from the predecessor
  // to IBB, and
  // (2) alter conditional branches so they branch to the other block
  // not IBB; this may require adding back an unconditional branch to IBB 
  // later, where there wasn't one coming in.  E.g.
  //   Bcc IBB
  //   fallthrough to QBB
  // here becomes
  //   Bncc QBB
  // with a conceptual B to IBB after that, which never actually exists.
  // With those changes, we see whether the predecessors' tails match,
  // and merge them if so.  We change things out of canonical form and
  // back to the way they were later in the process.  (OptimizeBranches
  // would undo some of this, but we can't use it, because we'd get into
  // a compile-time infinite loop repeatedly doing and undoing the same
  // transformations.)

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    if (I->pred_size() >= 2 && I->pred_size() < TailMergeThreshold) {
      MachineBasicBlock *IBB = I;
      MachineBasicBlock *PredBB = prior(I);
      MergePotentials.clear();
      for (MachineBasicBlock::pred_iterator P = I->pred_begin(), 
                                            E2 = I->pred_end();
           P != E2; ++P) {
        MachineBasicBlock* PBB = *P;
        // Skip blocks that loop to themselves, can't tail merge these.
        if (PBB==IBB)
          continue;
        MachineBasicBlock *TBB = 0, *FBB = 0;
        SmallVector<MachineOperand, 4> Cond;
        if (!TII->AnalyzeBranch(*PBB, TBB, FBB, Cond)) {
          // Failing case:  IBB is the target of a cbr, and
          // we cannot reverse the branch.
          SmallVector<MachineOperand, 4> NewCond(Cond);
          if (!Cond.empty() && TBB==IBB) {
            if (TII->ReverseBranchCondition(NewCond))
              continue;
            // This is the QBB case described above
            if (!FBB)
              FBB = next(MachineFunction::iterator(PBB));
          }
          // Failing case:  the only way IBB can be reached from PBB is via
          // exception handling.  Happens for landing pads.  Would be nice
          // to have a bit in the edge so we didn't have to do all this.
          if (IBB->isLandingPad()) {
            MachineFunction::iterator IP = PBB;  IP++;
            MachineBasicBlock* PredNextBB = NULL;
            if (IP!=MF.end())
              PredNextBB = IP;
            if (TBB==NULL) {
              if (IBB!=PredNextBB)      // fallthrough
                continue;
            } else if (FBB) {
              if (TBB!=IBB && FBB!=IBB)   // cbr then ubr
                continue;
            } else if (Cond.empty()) {
              if (TBB!=IBB)               // ubr
                continue;
            } else {
              if (TBB!=IBB && IBB!=PredNextBB)  // cbr
                continue;
            }
          }
          // Remove the unconditional branch at the end, if any.
          if (TBB && (Cond.empty() || FBB)) {
            TII->RemoveBranch(*PBB);
            if (!Cond.empty())
              // reinsert conditional branch only, for now
              TII->InsertBranch(*PBB, (TBB==IBB) ? FBB : TBB, 0, NewCond);
          }
          MergePotentials.push_back(std::make_pair(HashEndOfMBB(PBB, 1U), *P));
        }
      }
    if (MergePotentials.size() >= 2)
      MadeChange |= TryMergeBlocks(I, PredBB);
    // Reinsert an unconditional branch if needed.
    // The 1 below can occur as a result of removing blocks in TryMergeBlocks.
    PredBB = prior(I);      // this may have been changed in TryMergeBlocks
    if (MergePotentials.size()==1 && 
        MergePotentials.begin()->second != PredBB)
      FixTail(MergePotentials.begin()->second, I, TII);
    }
  }
  return MadeChange;
}

//===----------------------------------------------------------------------===//
//  Branch Optimization
//===----------------------------------------------------------------------===//

bool BranchFolder::OptimizeBranches(MachineFunction &MF) {
  MadeChange = false;
  
  // Make sure blocks are numbered in order
  MF.RenumberBlocks();

  for (MachineFunction::iterator I = ++MF.begin(), E = MF.end(); I != E; ) {
    MachineBasicBlock *MBB = I++;
    OptimizeBlock(MBB);
    
    // If it is dead, remove it.
    if (MBB->pred_empty()) {
      RemoveDeadBlock(MBB);
      MadeChange = true;
      ++NumDeadBlocks;
    }
  }
  return MadeChange;
}


/// CanFallThrough - Return true if the specified block (with the specified
/// branch condition) can implicitly transfer control to the block after it by
/// falling off the end of it.  This should return false if it can reach the
/// block after it, but it uses an explicit branch to do so (e.g. a table jump).
///
/// True is a conservative answer.
///
bool BranchFolder::CanFallThrough(MachineBasicBlock *CurBB,
                                  bool BranchUnAnalyzable,
                                  MachineBasicBlock *TBB, 
                                  MachineBasicBlock *FBB,
                                  const SmallVectorImpl<MachineOperand> &Cond) {
  MachineFunction::iterator Fallthrough = CurBB;
  ++Fallthrough;
  // If FallthroughBlock is off the end of the function, it can't fall through.
  if (Fallthrough == CurBB->getParent()->end())
    return false;
  
  // If FallthroughBlock isn't a successor of CurBB, no fallthrough is possible.
  if (!CurBB->isSuccessor(Fallthrough))
    return false;
  
  // If we couldn't analyze the branch, assume it could fall through.
  if (BranchUnAnalyzable) return true;
  
  // If there is no branch, control always falls through.
  if (TBB == 0) return true;

  // If there is some explicit branch to the fallthrough block, it can obviously
  // reach, even though the branch should get folded to fall through implicitly.
  if (MachineFunction::iterator(TBB) == Fallthrough ||
      MachineFunction::iterator(FBB) == Fallthrough)
    return true;
  
  // If it's an unconditional branch to some block not the fall through, it 
  // doesn't fall through.
  if (Cond.empty()) return false;
  
  // Otherwise, if it is conditional and has no explicit false block, it falls
  // through.
  return FBB == 0;
}

/// CanFallThrough - Return true if the specified can implicitly transfer
/// control to the block after it by falling off the end of it.  This should
/// return false if it can reach the block after it, but it uses an explicit
/// branch to do so (e.g. a table jump).
///
/// True is a conservative answer.
///
bool BranchFolder::CanFallThrough(MachineBasicBlock *CurBB) {
  MachineBasicBlock *TBB = 0, *FBB = 0;
  SmallVector<MachineOperand, 4> Cond;
  bool CurUnAnalyzable = TII->AnalyzeBranch(*CurBB, TBB, FBB, Cond);
  return CanFallThrough(CurBB, CurUnAnalyzable, TBB, FBB, Cond);
}

/// IsBetterFallthrough - Return true if it would be clearly better to
/// fall-through to MBB1 than to fall through into MBB2.  This has to return
/// a strict ordering, returning true for both (MBB1,MBB2) and (MBB2,MBB1) will
/// result in infinite loops.
static bool IsBetterFallthrough(MachineBasicBlock *MBB1, 
                                MachineBasicBlock *MBB2) {
  // Right now, we use a simple heuristic.  If MBB2 ends with a call, and
  // MBB1 doesn't, we prefer to fall through into MBB1.  This allows us to
  // optimize branches that branch to either a return block or an assert block
  // into a fallthrough to the return.
  if (MBB1->empty() || MBB2->empty()) return false;
 
  // If there is a clear successor ordering we make sure that one block
  // will fall through to the next
  if (MBB1->isSuccessor(MBB2)) return true;
  if (MBB2->isSuccessor(MBB1)) return false;

  MachineInstr *MBB1I = --MBB1->end();
  MachineInstr *MBB2I = --MBB2->end();
  return MBB2I->getDesc().isCall() && !MBB1I->getDesc().isCall();
}

/// OptimizeBlock - Analyze and optimize control flow related to the specified
/// block.  This is never called on the entry block.
void BranchFolder::OptimizeBlock(MachineBasicBlock *MBB) {
  MachineFunction::iterator FallThrough = MBB;
  ++FallThrough;
  
  // If this block is empty, make everyone use its fall-through, not the block
  // explicitly.  Landing pads should not do this since the landing-pad table
  // points to this block.
  if (MBB->empty() && !MBB->isLandingPad()) {
    // Dead block?  Leave for cleanup later.
    if (MBB->pred_empty()) return;
    
    if (FallThrough == MBB->getParent()->end()) {
      // TODO: Simplify preds to not branch here if possible!
    } else {
      // Rewrite all predecessors of the old block to go to the fallthrough
      // instead.
      while (!MBB->pred_empty()) {
        MachineBasicBlock *Pred = *(MBB->pred_end()-1);
        Pred->ReplaceUsesOfBlockWith(MBB, FallThrough);
      }
      
      // If MBB was the target of a jump table, update jump tables to go to the
      // fallthrough instead.
      MBB->getParent()->getJumpTableInfo()->
        ReplaceMBBInJumpTables(MBB, FallThrough);
      MadeChange = true;
    }
    return;
  }

  // Check to see if we can simplify the terminator of the block before this
  // one.
  MachineBasicBlock &PrevBB = *prior(MachineFunction::iterator(MBB));

  MachineBasicBlock *PriorTBB = 0, *PriorFBB = 0;
  SmallVector<MachineOperand, 4> PriorCond;
  bool PriorUnAnalyzable =
    TII->AnalyzeBranch(PrevBB, PriorTBB, PriorFBB, PriorCond);
  if (!PriorUnAnalyzable) {
    // If the CFG for the prior block has extra edges, remove them.
    MadeChange |= PrevBB.CorrectExtraCFGEdges(PriorTBB, PriorFBB,
                                              !PriorCond.empty());
    
    // If the previous branch is conditional and both conditions go to the same
    // destination, remove the branch, replacing it with an unconditional one or
    // a fall-through.
    if (PriorTBB && PriorTBB == PriorFBB) {
      TII->RemoveBranch(PrevBB);
      PriorCond.clear(); 
      if (PriorTBB != MBB)
        TII->InsertBranch(PrevBB, PriorTBB, 0, PriorCond);
      MadeChange = true;
      ++NumBranchOpts;
      return OptimizeBlock(MBB);
    }
    
    // If the previous branch *only* branches to *this* block (conditional or
    // not) remove the branch.
    if (PriorTBB == MBB && PriorFBB == 0) {
      TII->RemoveBranch(PrevBB);
      MadeChange = true;
      ++NumBranchOpts;
      return OptimizeBlock(MBB);
    }
    
    // If the prior block branches somewhere else on the condition and here if
    // the condition is false, remove the uncond second branch.
    if (PriorFBB == MBB) {
      TII->RemoveBranch(PrevBB);
      TII->InsertBranch(PrevBB, PriorTBB, 0, PriorCond);
      MadeChange = true;
      ++NumBranchOpts;
      return OptimizeBlock(MBB);
    }
    
    // If the prior block branches here on true and somewhere else on false, and
    // if the branch condition is reversible, reverse the branch to create a
    // fall-through.
    if (PriorTBB == MBB) {
      SmallVector<MachineOperand, 4> NewPriorCond(PriorCond);
      if (!TII->ReverseBranchCondition(NewPriorCond)) {
        TII->RemoveBranch(PrevBB);
        TII->InsertBranch(PrevBB, PriorFBB, 0, NewPriorCond);
        MadeChange = true;
        ++NumBranchOpts;
        return OptimizeBlock(MBB);
      }
    }
    
    // If this block doesn't fall through (e.g. it ends with an uncond branch or
    // has no successors) and if the pred falls through into this block, and if
    // it would otherwise fall through into the block after this, move this
    // block to the end of the function.
    //
    // We consider it more likely that execution will stay in the function (e.g.
    // due to loops) than it is to exit it.  This asserts in loops etc, moving
    // the assert condition out of the loop body.
    if (!PriorCond.empty() && PriorFBB == 0 &&
        MachineFunction::iterator(PriorTBB) == FallThrough &&
        !CanFallThrough(MBB)) {
      bool DoTransform = true;
      
      // We have to be careful that the succs of PredBB aren't both no-successor
      // blocks.  If neither have successors and if PredBB is the second from
      // last block in the function, we'd just keep swapping the two blocks for
      // last.  Only do the swap if one is clearly better to fall through than
      // the other.
      if (FallThrough == --MBB->getParent()->end() &&
          !IsBetterFallthrough(PriorTBB, MBB))
        DoTransform = false;

      // We don't want to do this transformation if we have control flow like:
      //   br cond BB2
      // BB1:
      //   ..
      //   jmp BBX
      // BB2:
      //   ..
      //   ret
      //
      // In this case, we could actually be moving the return block *into* a
      // loop!
      if (DoTransform && !MBB->succ_empty() &&
          (!CanFallThrough(PriorTBB) || PriorTBB->empty()))
        DoTransform = false;
      
      
      if (DoTransform) {
        // Reverse the branch so we will fall through on the previous true cond.
        SmallVector<MachineOperand, 4> NewPriorCond(PriorCond);
        if (!TII->ReverseBranchCondition(NewPriorCond)) {
          DOUT << "\nMoving MBB: " << *MBB;
          DOUT << "To make fallthrough to: " << *PriorTBB << "\n";
          
          TII->RemoveBranch(PrevBB);
          TII->InsertBranch(PrevBB, MBB, 0, NewPriorCond);

          // Move this block to the end of the function.
          MBB->moveAfter(--MBB->getParent()->end());
          MadeChange = true;
          ++NumBranchOpts;
          return;
        }
      }
    }
  }
  
  // Analyze the branch in the current block.
  MachineBasicBlock *CurTBB = 0, *CurFBB = 0;
  SmallVector<MachineOperand, 4> CurCond;
  bool CurUnAnalyzable = TII->AnalyzeBranch(*MBB, CurTBB, CurFBB, CurCond);
  if (!CurUnAnalyzable) {
    // If the CFG for the prior block has extra edges, remove them.
    MadeChange |= MBB->CorrectExtraCFGEdges(CurTBB, CurFBB, !CurCond.empty());

    // If this is a two-way branch, and the FBB branches to this block, reverse 
    // the condition so the single-basic-block loop is faster.  Instead of:
    //    Loop: xxx; jcc Out; jmp Loop
    // we want:
    //    Loop: xxx; jncc Loop; jmp Out
    if (CurTBB && CurFBB && CurFBB == MBB && CurTBB != MBB) {
      SmallVector<MachineOperand, 4> NewCond(CurCond);
      if (!TII->ReverseBranchCondition(NewCond)) {
        TII->RemoveBranch(*MBB);
        TII->InsertBranch(*MBB, CurFBB, CurTBB, NewCond);
        MadeChange = true;
        ++NumBranchOpts;
        return OptimizeBlock(MBB);
      }
    }
    
    
    // If this branch is the only thing in its block, see if we can forward
    // other blocks across it.
    if (CurTBB && CurCond.empty() && CurFBB == 0 && 
        MBB->begin()->getDesc().isBranch() && CurTBB != MBB) {
      // This block may contain just an unconditional branch.  Because there can
      // be 'non-branch terminators' in the block, try removing the branch and
      // then seeing if the block is empty.
      TII->RemoveBranch(*MBB);

      // If this block is just an unconditional branch to CurTBB, we can
      // usually completely eliminate the block.  The only case we cannot
      // completely eliminate the block is when the block before this one
      // falls through into MBB and we can't understand the prior block's branch
      // condition.
      if (MBB->empty()) {
        bool PredHasNoFallThrough = TII->BlockHasNoFallThrough(PrevBB);
        if (PredHasNoFallThrough || !PriorUnAnalyzable ||
            !PrevBB.isSuccessor(MBB)) {
          // If the prior block falls through into us, turn it into an
          // explicit branch to us to make updates simpler.
          if (!PredHasNoFallThrough && PrevBB.isSuccessor(MBB) && 
              PriorTBB != MBB && PriorFBB != MBB) {
            if (PriorTBB == 0) {
              assert(PriorCond.empty() && PriorFBB == 0 &&
                     "Bad branch analysis");
              PriorTBB = MBB;
            } else {
              assert(PriorFBB == 0 && "Machine CFG out of date!");
              PriorFBB = MBB;
            }
            TII->RemoveBranch(PrevBB);
            TII->InsertBranch(PrevBB, PriorTBB, PriorFBB, PriorCond);
          }

          // Iterate through all the predecessors, revectoring each in-turn.
          size_t PI = 0;
          bool DidChange = false;
          bool HasBranchToSelf = false;
          while(PI != MBB->pred_size()) {
            MachineBasicBlock *PMBB = *(MBB->pred_begin() + PI);
            if (PMBB == MBB) {
              // If this block has an uncond branch to itself, leave it.
              ++PI;
              HasBranchToSelf = true;
            } else {
              DidChange = true;
              PMBB->ReplaceUsesOfBlockWith(MBB, CurTBB);
            }
          }

          // Change any jumptables to go to the new MBB.
          MBB->getParent()->getJumpTableInfo()->
            ReplaceMBBInJumpTables(MBB, CurTBB);
          if (DidChange) {
            ++NumBranchOpts;
            MadeChange = true;
            if (!HasBranchToSelf) return;
          }
        }
      }
      
      // Add the branch back if the block is more than just an uncond branch.
      TII->InsertBranch(*MBB, CurTBB, 0, CurCond);
    }
  }

  // If the prior block doesn't fall through into this block, and if this
  // block doesn't fall through into some other block, see if we can find a
  // place to move this block where a fall-through will happen.
  if (!CanFallThrough(&PrevBB, PriorUnAnalyzable,
                      PriorTBB, PriorFBB, PriorCond)) {
    // Now we know that there was no fall-through into this block, check to
    // see if it has a fall-through into its successor.
    bool CurFallsThru = CanFallThrough(MBB, CurUnAnalyzable, CurTBB, CurFBB, 
                                       CurCond);

    if (!MBB->isLandingPad()) {
      // Check all the predecessors of this block.  If one of them has no fall
      // throughs, move this block right after it.
      for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
           E = MBB->pred_end(); PI != E; ++PI) {
        // Analyze the branch at the end of the pred.
        MachineBasicBlock *PredBB = *PI;
        MachineFunction::iterator PredFallthrough = PredBB; ++PredFallthrough;
        if (PredBB != MBB && !CanFallThrough(PredBB)
            && (!CurFallsThru || !CurTBB || !CurFBB)
            && (!CurFallsThru || MBB->getNumber() >= PredBB->getNumber())) {
          // If the current block doesn't fall through, just move it.
          // If the current block can fall through and does not end with a
          // conditional branch, we need to append an unconditional jump to 
          // the (current) next block.  To avoid a possible compile-time
          // infinite loop, move blocks only backward in this case.
          // Also, if there are already 2 branches here, we cannot add a third;
          // this means we have the case
          // Bcc next
          // B elsewhere
          // next:
          if (CurFallsThru) {
            MachineBasicBlock *NextBB = next(MachineFunction::iterator(MBB));
            CurCond.clear();
            TII->InsertBranch(*MBB, NextBB, 0, CurCond);
          }
          MBB->moveAfter(PredBB);
          MadeChange = true;
          return OptimizeBlock(MBB);
        }
      }
    }
        
    if (!CurFallsThru) {
      // Check all successors to see if we can move this block before it.
      for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
           E = MBB->succ_end(); SI != E; ++SI) {
        // Analyze the branch at the end of the block before the succ.
        MachineBasicBlock *SuccBB = *SI;
        MachineFunction::iterator SuccPrev = SuccBB; --SuccPrev;
        std::vector<MachineOperand> SuccPrevCond;
        
        // If this block doesn't already fall-through to that successor, and if
        // the succ doesn't already have a block that can fall through into it,
        // and if the successor isn't an EH destination, we can arrange for the
        // fallthrough to happen.
        if (SuccBB != MBB && !CanFallThrough(SuccPrev) &&
            !SuccBB->isLandingPad()) {
          MBB->moveBefore(SuccBB);
          MadeChange = true;
          return OptimizeBlock(MBB);
        }
      }
      
      // Okay, there is no really great place to put this block.  If, however,
      // the block before this one would be a fall-through if this block were
      // removed, move this block to the end of the function.
      if (FallThrough != MBB->getParent()->end() &&
          PrevBB.isSuccessor(FallThrough)) {
        MBB->moveAfter(--MBB->getParent()->end());
        MadeChange = true;
        return;
      }
    }
  }
}
