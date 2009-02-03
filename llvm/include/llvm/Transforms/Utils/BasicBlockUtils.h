//===-- Transform/Utils/BasicBlockUtils.h - BasicBlock Utils ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform manipulations on basic blocks, and
// instructions contained within basic blocks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_BASICBLOCK_H
#define LLVM_TRANSFORMS_UTILS_BASICBLOCK_H

// FIXME: Move to this file: BasicBlock::removePredecessor, BB::splitBasicBlock

#include "llvm/BasicBlock.h"
#include "llvm/Support/CFG.h"

namespace llvm {

class Instruction;
class Pass;
class AliasAnalysis;

/// DeleteDeadBlock - Delete the specified block, which must have no
/// predecessors.
void DeleteDeadBlock(BasicBlock *BB);
  
  
/// FoldSingleEntryPHINodes - We know that BB has one predecessor.  If there are
/// any single-entry PHI nodes in it, fold them away.  This handles the case
/// when all entries to the PHI nodes in a block are guaranteed equal, such as
/// when the block has exactly one predecessor.
void FoldSingleEntryPHINodes(BasicBlock *BB);
  
  
/// MergeBlockIntoPredecessor - Attempts to merge a block into its predecessor,
/// if possible.  The return value indicates success or failure.
bool MergeBlockIntoPredecessor(BasicBlock* BB, Pass* P = 0);

// ReplaceInstWithValue - Replace all uses of an instruction (specified by BI)
// with a value, then remove and delete the original instruction.
//
void ReplaceInstWithValue(BasicBlock::InstListType &BIL,
                          BasicBlock::iterator &BI, Value *V);

// ReplaceInstWithInst - Replace the instruction specified by BI with the
// instruction specified by I.  The original instruction is deleted and BI is
// updated to point to the new instruction.
//
void ReplaceInstWithInst(BasicBlock::InstListType &BIL,
                         BasicBlock::iterator &BI, Instruction *I);

// ReplaceInstWithInst - Replace the instruction specified by From with the
// instruction specified by To.
//
void ReplaceInstWithInst(Instruction *From, Instruction *To);

/// FindAvailableLoadedValue - Scan the ScanBB block backwards (starting at the
/// instruction before ScanFrom) checking to see if we have the value at the
/// memory address *Ptr locally available within a small number of instructions.
/// If the value is available, return it.
///
/// If not, return the iterator for the last validated instruction that the 
/// value would be live through.  If we scanned the entire block and didn't find
/// something that invalidates *Ptr or provides it, ScanFrom would be left at
/// begin() and this returns null.  ScanFrom could also be left 
///
/// MaxInstsToScan specifies the maximum instructions to scan in the block.  If
/// it is set to 0, it will scan the whole block. You can also optionally
/// specify an alias analysis implementation, which makes this more precise.
Value *FindAvailableLoadedValue(Value *Ptr, BasicBlock *ScanBB,
                                BasicBlock::iterator &ScanFrom,
                                unsigned MaxInstsToScan = 6,
                                AliasAnalysis *AA = 0);
    
  

// RemoveSuccessor - Change the specified terminator instruction such that its
// successor #SuccNum no longer exists.  Because this reduces the outgoing
// degree of the current basic block, the actual terminator instruction itself
// may have to be changed.  In the case where the last successor of the block is
// deleted, a return instruction is inserted in its place which can cause a
// suprising change in program behavior if it is not expected.
//
void RemoveSuccessor(TerminatorInst *TI, unsigned SuccNum);

/// isCriticalEdge - Return true if the specified edge is a critical edge.
/// Critical edges are edges from a block with multiple successors to a block
/// with multiple predecessors.
///
bool isCriticalEdge(const TerminatorInst *TI, unsigned SuccNum,
                    bool AllowIdenticalEdges = false);

/// SplitCriticalEdge - If this edge is a critical edge, insert a new node to
/// split the critical edge.  This will update DominatorTree and
/// DominatorFrontier information if it is available, thus calling this pass
/// will not invalidate either of them. This returns true if the edge was split,
/// false otherwise.  
///
/// If MergeIdenticalEdges is true (not the default), *all* edges from TI to the
/// specified successor will be merged into the same critical edge block.  
/// This is most commonly interesting with switch instructions, which may 
/// have many edges to any one destination.  This ensures that all edges to that
/// dest go to one block instead of each going to a different block, but isn't 
/// the standard definition of a "critical edge".
///
bool SplitCriticalEdge(TerminatorInst *TI, unsigned SuccNum, Pass *P = 0,
                       bool MergeIdenticalEdges = false);

inline bool SplitCriticalEdge(BasicBlock *BB, succ_iterator SI, Pass *P = 0) {
  return SplitCriticalEdge(BB->getTerminator(), SI.getSuccessorIndex(), P);
}

/// SplitCriticalEdge - If the edge from *PI to BB is not critical, return
/// false.  Otherwise, split all edges between the two blocks and return true.
/// This updates all of the same analyses as the other SplitCriticalEdge
/// function.  If P is specified, it updates the analyses
/// described above.
inline bool SplitCriticalEdge(BasicBlock *Succ, pred_iterator PI, Pass *P = 0) {
  bool MadeChange = false;
  TerminatorInst *TI = (*PI)->getTerminator();
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
    if (TI->getSuccessor(i) == Succ)
      MadeChange |= SplitCriticalEdge(TI, i, P);
  return MadeChange;
}

/// SplitCriticalEdge - If an edge from Src to Dst is critical, split the edge
/// and return true, otherwise return false.  This method requires that there be
/// an edge between the two blocks.  If P is specified, it updates the analyses
/// described above.
inline bool SplitCriticalEdge(BasicBlock *Src, BasicBlock *Dst, Pass *P = 0,
                              bool MergeIdenticalEdges = false) {
  TerminatorInst *TI = Src->getTerminator();
  unsigned i = 0;
  while (1) {
    assert(i != TI->getNumSuccessors() && "Edge doesn't exist!");
    if (TI->getSuccessor(i) == Dst)
      return SplitCriticalEdge(TI, i, P, MergeIdenticalEdges);
    ++i;
  }
}

/// SplitEdge -  Split the edge connecting specified block. Pass P must 
/// not be NULL. 
BasicBlock *SplitEdge(BasicBlock *From, BasicBlock *To, Pass *P);

/// SplitBlock - Split the specified block at the specified instruction - every
/// thing before SplitPt stays in Old and everything starting with SplitPt moves
/// to a new block.  The two blocks are joined by an unconditional branch and
/// the loop info is updated.
///
BasicBlock *SplitBlock(BasicBlock *Old, Instruction *SplitPt, Pass *P);
 
/// SplitBlockPredecessors - This method transforms BB by introducing a new
/// basic block into the function, and moving some of the predecessors of BB to
/// be predecessors of the new block.  The new predecessors are indicated by the
/// Preds array, which has NumPreds elements in it.  The new block is given a
/// suffix of 'Suffix'.  This function returns the new block.
///
/// This currently updates the LLVM IR, AliasAnalysis, DominatorTree and
/// DominanceFrontier, but no other analyses.
BasicBlock *SplitBlockPredecessors(BasicBlock *BB, BasicBlock *const *Preds,
                                   unsigned NumPreds, const char *Suffix,
                                   Pass *P = 0);
  
} // End llvm namespace

#endif
