//===-- BasicBlockUtils.cpp - BasicBlock Utilities -------------------------==//
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

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Constant.h"
#include "llvm/Type.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ValueHandle.h"
#include <algorithm>
using namespace llvm;

/// DeleteDeadBlock - Delete the specified block, which must have no
/// predecessors.
void llvm::DeleteDeadBlock(BasicBlock *BB) {
  assert((pred_begin(BB) == pred_end(BB) ||
         // Can delete self loop.
         BB->getSinglePredecessor() == BB) && "Block is not dead!");
  TerminatorInst *BBTerm = BB->getTerminator();
  
  // Loop through all of our successors and make sure they know that one
  // of their predecessors is going away.
  for (unsigned i = 0, e = BBTerm->getNumSuccessors(); i != e; ++i)
    BBTerm->getSuccessor(i)->removePredecessor(BB);
  
  // Zap all the instructions in the block.
  while (!BB->empty()) {
    Instruction &I = BB->back();
    // If this instruction is used, replace uses with an arbitrary value.
    // Because control flow can't get here, we don't care what we replace the
    // value with.  Note that since this block is unreachable, and all values
    // contained within it must dominate their uses, that all uses will
    // eventually be removed (they are themselves dead).
    if (!I.use_empty())
      I.replaceAllUsesWith(UndefValue::get(I.getType()));
    BB->getInstList().pop_back();
  }
  
  // Zap the block!
  BB->eraseFromParent();
}

/// FoldSingleEntryPHINodes - We know that BB has one predecessor.  If there are
/// any single-entry PHI nodes in it, fold them away.  This handles the case
/// when all entries to the PHI nodes in a block are guaranteed equal, such as
/// when the block has exactly one predecessor.
void llvm::FoldSingleEntryPHINodes(BasicBlock *BB) {
  if (!isa<PHINode>(BB->begin()))
    return;
  
  while (PHINode *PN = dyn_cast<PHINode>(BB->begin())) {
    if (PN->getIncomingValue(0) != PN)
      PN->replaceAllUsesWith(PN->getIncomingValue(0));
    else
      PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
    PN->eraseFromParent();
  }
}


/// DeleteDeadPHIs - Examine each PHI in the given block and delete it if it
/// is dead. Also recursively delete any operands that become dead as
/// a result. This includes tracing the def-use list from the PHI to see if
/// it is ultimately unused or if it reaches an unused cycle.
void llvm::DeleteDeadPHIs(BasicBlock *BB) {
  // Recursively deleting a PHI may cause multiple PHIs to be deleted
  // or RAUW'd undef, so use an array of WeakVH for the PHIs to delete.
  SmallVector<WeakVH, 8> PHIs;
  for (BasicBlock::iterator I = BB->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    PHIs.push_back(PN);

  for (unsigned i = 0, e = PHIs.size(); i != e; ++i)
    if (PHINode *PN = dyn_cast_or_null<PHINode>(PHIs[i].operator Value*()))
      RecursivelyDeleteDeadPHINode(PN);
}

/// MergeBlockIntoPredecessor - Attempts to merge a block into its predecessor,
/// if possible.  The return value indicates success or failure.
bool llvm::MergeBlockIntoPredecessor(BasicBlock* BB, Pass* P) {
  pred_iterator PI(pred_begin(BB)), PE(pred_end(BB));
  // Can't merge the entry block.
  if (pred_begin(BB) == pred_end(BB)) return false;
  
  BasicBlock *PredBB = *PI++;
  for (; PI != PE; ++PI)  // Search all predecessors, see if they are all same
    if (*PI != PredBB) {
      PredBB = 0;       // There are multiple different predecessors...
      break;
    }
  
  // Can't merge if there are multiple predecessors.
  if (!PredBB) return false;
  // Don't break self-loops.
  if (PredBB == BB) return false;
  // Don't break invokes.
  if (isa<InvokeInst>(PredBB->getTerminator())) return false;
  
  succ_iterator SI(succ_begin(PredBB)), SE(succ_end(PredBB));
  BasicBlock* OnlySucc = BB;
  for (; SI != SE; ++SI)
    if (*SI != OnlySucc) {
      OnlySucc = 0;     // There are multiple distinct successors!
      break;
    }
  
  // Can't merge if there are multiple successors.
  if (!OnlySucc) return false;

  // Can't merge if there is PHI loop.
  for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (PN->getIncomingValue(i) == PN)
          return false;
    } else
      break;
  }

  // Begin by getting rid of unneeded PHIs.
  while (PHINode *PN = dyn_cast<PHINode>(&BB->front())) {
    PN->replaceAllUsesWith(PN->getIncomingValue(0));
    BB->getInstList().pop_front();  // Delete the phi node...
  }
  
  // Delete the unconditional branch from the predecessor...
  PredBB->getInstList().pop_back();
  
  // Move all definitions in the successor to the predecessor...
  PredBB->getInstList().splice(PredBB->end(), BB->getInstList());
  
  // Make all PHI nodes that referred to BB now refer to Pred as their
  // source...
  BB->replaceAllUsesWith(PredBB);
  
  // Inherit predecessors name if it exists.
  if (!PredBB->hasName())
    PredBB->takeName(BB);
  
  // Finally, erase the old block and update dominator info.
  if (P) {
    if (DominatorTree* DT = P->getAnalysisIfAvailable<DominatorTree>()) {
      DomTreeNode* DTN = DT->getNode(BB);
      DomTreeNode* PredDTN = DT->getNode(PredBB);
  
      if (DTN) {
        SmallPtrSet<DomTreeNode*, 8> Children(DTN->begin(), DTN->end());
        for (SmallPtrSet<DomTreeNode*, 8>::iterator DI = Children.begin(),
             DE = Children.end(); DI != DE; ++DI)
          DT->changeImmediateDominator(*DI, PredDTN);

        DT->eraseNode(BB);
      }
    }
  }
  
  BB->eraseFromParent();
  
  
  return true;
}

/// ReplaceInstWithValue - Replace all uses of an instruction (specified by BI)
/// with a value, then remove and delete the original instruction.
///
void llvm::ReplaceInstWithValue(BasicBlock::InstListType &BIL,
                                BasicBlock::iterator &BI, Value *V) {
  Instruction &I = *BI;
  // Replaces all of the uses of the instruction with uses of the value
  I.replaceAllUsesWith(V);

  // Make sure to propagate a name if there is one already.
  if (I.hasName() && !V->hasName())
    V->takeName(&I);

  // Delete the unnecessary instruction now...
  BI = BIL.erase(BI);
}


/// ReplaceInstWithInst - Replace the instruction specified by BI with the
/// instruction specified by I.  The original instruction is deleted and BI is
/// updated to point to the new instruction.
///
void llvm::ReplaceInstWithInst(BasicBlock::InstListType &BIL,
                               BasicBlock::iterator &BI, Instruction *I) {
  assert(I->getParent() == 0 &&
         "ReplaceInstWithInst: Instruction already inserted into basic block!");

  // Insert the new instruction into the basic block...
  BasicBlock::iterator New = BIL.insert(BI, I);

  // Replace all uses of the old instruction, and delete it.
  ReplaceInstWithValue(BIL, BI, I);

  // Move BI back to point to the newly inserted instruction
  BI = New;
}

/// ReplaceInstWithInst - Replace the instruction specified by From with the
/// instruction specified by To.
///
void llvm::ReplaceInstWithInst(Instruction *From, Instruction *To) {
  BasicBlock::iterator BI(From);
  ReplaceInstWithInst(From->getParent()->getInstList(), BI, To);
}

/// RemoveSuccessor - Change the specified terminator instruction such that its
/// successor SuccNum no longer exists.  Because this reduces the outgoing
/// degree of the current basic block, the actual terminator instruction itself
/// may have to be changed.  In the case where the last successor of the block 
/// is deleted, a return instruction is inserted in its place which can cause a
/// surprising change in program behavior if it is not expected.
///
void llvm::RemoveSuccessor(TerminatorInst *TI, unsigned SuccNum) {
  assert(SuccNum < TI->getNumSuccessors() &&
         "Trying to remove a nonexistant successor!");

  // If our old successor block contains any PHI nodes, remove the entry in the
  // PHI nodes that comes from this branch...
  //
  BasicBlock *BB = TI->getParent();
  TI->getSuccessor(SuccNum)->removePredecessor(BB);

  TerminatorInst *NewTI = 0;
  switch (TI->getOpcode()) {
  case Instruction::Br:
    // If this is a conditional branch... convert to unconditional branch.
    if (TI->getNumSuccessors() == 2) {
      cast<BranchInst>(TI)->setUnconditionalDest(TI->getSuccessor(1-SuccNum));
    } else {                    // Otherwise convert to a return instruction...
      Value *RetVal = 0;

      // Create a value to return... if the function doesn't return null...
      if (BB->getParent()->getReturnType() != Type::getVoidTy(TI->getContext()))
        RetVal = Constant::getNullValue(BB->getParent()->getReturnType());

      // Create the return...
      NewTI = ReturnInst::Create(TI->getContext(), RetVal);
    }
    break;

  case Instruction::Invoke:    // Should convert to call
  case Instruction::Switch:    // Should remove entry
  default:
  case Instruction::Ret:       // Cannot happen, has no successors!
    llvm_unreachable("Unhandled terminator instruction type in RemoveSuccessor!");
  }

  if (NewTI)   // If it's a different instruction, replace.
    ReplaceInstWithInst(TI, NewTI);
}

/// SplitEdge -  Split the edge connecting specified block. Pass P must 
/// not be NULL. 
BasicBlock *llvm::SplitEdge(BasicBlock *BB, BasicBlock *Succ, Pass *P) {
  TerminatorInst *LatchTerm = BB->getTerminator();
  unsigned SuccNum = 0;
#ifndef NDEBUG
  unsigned e = LatchTerm->getNumSuccessors();
#endif
  for (unsigned i = 0; ; ++i) {
    assert(i != e && "Didn't find edge?");
    if (LatchTerm->getSuccessor(i) == Succ) {
      SuccNum = i;
      break;
    }
  }
  
  // If this is a critical edge, let SplitCriticalEdge do it.
  if (SplitCriticalEdge(BB->getTerminator(), SuccNum, P))
    return LatchTerm->getSuccessor(SuccNum);

  // If the edge isn't critical, then BB has a single successor or Succ has a
  // single pred.  Split the block.
  BasicBlock::iterator SplitPoint;
  if (BasicBlock *SP = Succ->getSinglePredecessor()) {
    // If the successor only has a single pred, split the top of the successor
    // block.
    assert(SP == BB && "CFG broken");
    SP = NULL;
    return SplitBlock(Succ, Succ->begin(), P);
  } else {
    // Otherwise, if BB has a single successor, split it at the bottom of the
    // block.
    assert(BB->getTerminator()->getNumSuccessors() == 1 &&
           "Should have a single succ!"); 
    return SplitBlock(BB, BB->getTerminator(), P);
  }
}

/// SplitBlock - Split the specified block at the specified instruction - every
/// thing before SplitPt stays in Old and everything starting with SplitPt moves
/// to a new block.  The two blocks are joined by an unconditional branch and
/// the loop info is updated.
///
BasicBlock *llvm::SplitBlock(BasicBlock *Old, Instruction *SplitPt, Pass *P) {
  BasicBlock::iterator SplitIt = SplitPt;
  while (isa<PHINode>(SplitIt))
    ++SplitIt;
  BasicBlock *New = Old->splitBasicBlock(SplitIt, Old->getName()+".split");

  // The new block lives in whichever loop the old one did.
  if (LoopInfo* LI = P->getAnalysisIfAvailable<LoopInfo>())
    if (Loop *L = LI->getLoopFor(Old))
      L->addBasicBlockToLoop(New, LI->getBase());

  if (DominatorTree *DT = P->getAnalysisIfAvailable<DominatorTree>())
    {
      // Old dominates New. New node domiantes all other nodes dominated by Old.
      DomTreeNode *OldNode = DT->getNode(Old);
      std::vector<DomTreeNode *> Children;
      for (DomTreeNode::iterator I = OldNode->begin(), E = OldNode->end();
           I != E; ++I) 
        Children.push_back(*I);

      DomTreeNode *NewNode =   DT->addNewBlock(New,Old);

      for (std::vector<DomTreeNode *>::iterator I = Children.begin(),
             E = Children.end(); I != E; ++I) 
        DT->changeImmediateDominator(*I, NewNode);
    }

  if (DominanceFrontier *DF = P->getAnalysisIfAvailable<DominanceFrontier>())
    DF->splitBlock(Old);
    
  return New;
}


/// SplitBlockPredecessors - This method transforms BB by introducing a new
/// basic block into the function, and moving some of the predecessors of BB to
/// be predecessors of the new block.  The new predecessors are indicated by the
/// Preds array, which has NumPreds elements in it.  The new block is given a
/// suffix of 'Suffix'.
///
/// This currently updates the LLVM IR, AliasAnalysis, DominatorTree and
/// DominanceFrontier, but no other analyses.
BasicBlock *llvm::SplitBlockPredecessors(BasicBlock *BB, 
                                         BasicBlock *const *Preds,
                                         unsigned NumPreds, const char *Suffix,
                                         Pass *P) {
  // Create new basic block, insert right before the original block.
  BasicBlock *NewBB = BasicBlock::Create(BB->getContext(), BB->getName()+Suffix,
                                         BB->getParent(), BB);
  
  // The new block unconditionally branches to the old block.
  BranchInst *BI = BranchInst::Create(BB, NewBB);
  
  // Move the edges from Preds to point to NewBB instead of BB.
  for (unsigned i = 0; i != NumPreds; ++i)
    Preds[i]->getTerminator()->replaceUsesOfWith(BB, NewBB);
  
  // Update dominator tree and dominator frontier if available.
  DominatorTree *DT = P ? P->getAnalysisIfAvailable<DominatorTree>() : 0;
  if (DT)
    DT->splitBlock(NewBB);
  if (DominanceFrontier *DF = P ? P->getAnalysisIfAvailable<DominanceFrontier>():0)
    DF->splitBlock(NewBB);
  AliasAnalysis *AA = P ? P->getAnalysisIfAvailable<AliasAnalysis>() : 0;
  
  
  // Insert a new PHI node into NewBB for every PHI node in BB and that new PHI
  // node becomes an incoming value for BB's phi node.  However, if the Preds
  // list is empty, we need to insert dummy entries into the PHI nodes in BB to
  // account for the newly created predecessor.
  if (NumPreds == 0) {
    // Insert dummy values as the incoming value.
    for (BasicBlock::iterator I = BB->begin(); isa<PHINode>(I); ++I)
      cast<PHINode>(I)->addIncoming(UndefValue::get(I->getType()), NewBB);
    return NewBB;
  }
  
  // Otherwise, create a new PHI node in NewBB for each PHI node in BB.
  for (BasicBlock::iterator I = BB->begin(); isa<PHINode>(I); ) {
    PHINode *PN = cast<PHINode>(I++);
    
    // Check to see if all of the values coming in are the same.  If so, we
    // don't need to create a new PHI node.
    Value *InVal = PN->getIncomingValueForBlock(Preds[0]);
    for (unsigned i = 1; i != NumPreds; ++i)
      if (InVal != PN->getIncomingValueForBlock(Preds[i])) {
        InVal = 0;
        break;
      }
    
    if (InVal) {
      // If all incoming values for the new PHI would be the same, just don't
      // make a new PHI.  Instead, just remove the incoming values from the old
      // PHI.
      for (unsigned i = 0; i != NumPreds; ++i)
        PN->removeIncomingValue(Preds[i], false);
    } else {
      // If the values coming into the block are not the same, we need a PHI.
      // Create the new PHI node, insert it into NewBB at the end of the block
      PHINode *NewPHI =
        PHINode::Create(PN->getType(), PN->getName()+".ph", BI);
      if (AA) AA->copyValue(PN, NewPHI);
      
      // Move all of the PHI values for 'Preds' to the new PHI.
      for (unsigned i = 0; i != NumPreds; ++i) {
        Value *V = PN->removeIncomingValue(Preds[i], false);
        NewPHI->addIncoming(V, Preds[i]);
      }
      InVal = NewPHI;
    }
    
    // Add an incoming value to the PHI node in the loop for the preheader
    // edge.
    PN->addIncoming(InVal, NewBB);
    
    // Check to see if we can eliminate this phi node.
    if (Value *V = PN->hasConstantValue(DT != 0)) {
      Instruction *I = dyn_cast<Instruction>(V);
      if (!I || DT == 0 || DT->dominates(I, PN)) {
        PN->replaceAllUsesWith(V);
        if (AA) AA->deleteValue(PN);
        PN->eraseFromParent();
      }
    }
  }
  
  return NewBB;
}

/// FindFunctionBackedges - Analyze the specified function to find all of the
/// loop backedges in the function and return them.  This is a relatively cheap
/// (compared to computing dominators and loop info) analysis.
///
/// The output is added to Result, as pairs of <from,to> edge info.
void llvm::FindFunctionBackedges(const Function &F,
     SmallVectorImpl<std::pair<const BasicBlock*,const BasicBlock*> > &Result) {
  const BasicBlock *BB = &F.getEntryBlock();
  if (succ_begin(BB) == succ_end(BB))
    return;
  
  SmallPtrSet<const BasicBlock*, 8> Visited;
  SmallVector<std::pair<const BasicBlock*, succ_const_iterator>, 8> VisitStack;
  SmallPtrSet<const BasicBlock*, 8> InStack;
  
  Visited.insert(BB);
  VisitStack.push_back(std::make_pair(BB, succ_begin(BB)));
  InStack.insert(BB);
  do {
    std::pair<const BasicBlock*, succ_const_iterator> &Top = VisitStack.back();
    const BasicBlock *ParentBB = Top.first;
    succ_const_iterator &I = Top.second;
    
    bool FoundNew = false;
    while (I != succ_end(ParentBB)) {
      BB = *I++;
      if (Visited.insert(BB)) {
        FoundNew = true;
        break;
      }
      // Successor is in VisitStack, it's a back edge.
      if (InStack.count(BB))
        Result.push_back(std::make_pair(ParentBB, BB));
    }
    
    if (FoundNew) {
      // Go down one level if there is a unvisited successor.
      InStack.insert(BB);
      VisitStack.push_back(std::make_pair(BB, succ_begin(BB)));
    } else {
      // Go up one level.
      InStack.erase(VisitStack.pop_back_val().first);
    }
  } while (!VisitStack.empty());
  
  
}



/// AreEquivalentAddressValues - Test if A and B will obviously have the same
/// value. This includes recognizing that %t0 and %t1 will have the same
/// value in code like this:
///   %t0 = getelementptr \@a, 0, 3
///   store i32 0, i32* %t0
///   %t1 = getelementptr \@a, 0, 3
///   %t2 = load i32* %t1
///
static bool AreEquivalentAddressValues(const Value *A, const Value *B) {
  // Test if the values are trivially equivalent.
  if (A == B) return true;
  
  // Test if the values come from identical arithmetic instructions.
  // Use isIdenticalToWhenDefined instead of isIdenticalTo because
  // this function is only used when one address use dominates the
  // other, which means that they'll always either have the same
  // value or one of them will have an undefined value.
  if (isa<BinaryOperator>(A) || isa<CastInst>(A) ||
      isa<PHINode>(A) || isa<GetElementPtrInst>(A))
    if (const Instruction *BI = dyn_cast<Instruction>(B))
      if (cast<Instruction>(A)->isIdenticalToWhenDefined(BI))
        return true;
  
  // Otherwise they may not be equivalent.
  return false;
}

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
Value *llvm::FindAvailableLoadedValue(Value *Ptr, BasicBlock *ScanBB,
                                      BasicBlock::iterator &ScanFrom,
                                      unsigned MaxInstsToScan,
                                      AliasAnalysis *AA) {
  if (MaxInstsToScan == 0) MaxInstsToScan = ~0U;

  // If we're using alias analysis to disambiguate get the size of *Ptr.
  unsigned AccessSize = 0;
  if (AA) {
    const Type *AccessTy = cast<PointerType>(Ptr->getType())->getElementType();
    AccessSize = AA->getTypeStoreSize(AccessTy);
  }
  
  while (ScanFrom != ScanBB->begin()) {
    // We must ignore debug info directives when counting (otherwise they
    // would affect codegen).
    Instruction *Inst = --ScanFrom;
    if (isa<DbgInfoIntrinsic>(Inst))
      continue;
    // We skip pointer-to-pointer bitcasts, which are NOPs.
    // It is necessary for correctness to skip those that feed into a
    // llvm.dbg.declare, as these are not present when debugging is off.
    if (isa<BitCastInst>(Inst) && isa<PointerType>(Inst->getType()))
      continue;

    // Restore ScanFrom to expected value in case next test succeeds
    ScanFrom++;
   
    // Don't scan huge blocks.
    if (MaxInstsToScan-- == 0) return 0;
    
    --ScanFrom;
    // If this is a load of Ptr, the loaded value is available.
    if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
      if (AreEquivalentAddressValues(LI->getOperand(0), Ptr))
        return LI;
    
    if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
      // If this is a store through Ptr, the value is available!
      if (AreEquivalentAddressValues(SI->getOperand(1), Ptr))
        return SI->getOperand(0);
      
      // If Ptr is an alloca and this is a store to a different alloca, ignore
      // the store.  This is a trivial form of alias analysis that is important
      // for reg2mem'd code.
      if ((isa<AllocaInst>(Ptr) || isa<GlobalVariable>(Ptr)) &&
          (isa<AllocaInst>(SI->getOperand(1)) ||
           isa<GlobalVariable>(SI->getOperand(1))))
        continue;
      
      // If we have alias analysis and it says the store won't modify the loaded
      // value, ignore the store.
      if (AA &&
          (AA->getModRefInfo(SI, Ptr, AccessSize) & AliasAnalysis::Mod) == 0)
        continue;
      
      // Otherwise the store that may or may not alias the pointer, bail out.
      ++ScanFrom;
      return 0;
    }
    
    // If this is some other instruction that may clobber Ptr, bail out.
    if (Inst->mayWriteToMemory()) {
      // If alias analysis claims that it really won't modify the load,
      // ignore it.
      if (AA &&
          (AA->getModRefInfo(Inst, Ptr, AccessSize) & AliasAnalysis::Mod) == 0)
        continue;
      
      // May modify the pointer, bail out.
      ++ScanFrom;
      return 0;
    }
  }
  
  // Got to the start of the block, we didn't find it, but are done for this
  // block.
  return 0;
}

/// CopyPrecedingStopPoint - If I is immediately preceded by a StopPoint,
/// make a copy of the stoppoint before InsertPos (presumably before copying
/// or moving I).
void llvm::CopyPrecedingStopPoint(Instruction *I, 
                                  BasicBlock::iterator InsertPos) {
  if (I != I->getParent()->begin()) {
    BasicBlock::iterator BBI = I;  --BBI;
    if (DbgStopPointInst *DSPI = dyn_cast<DbgStopPointInst>(BBI)) {
      CallInst *newDSPI = DSPI->clone(I->getContext());
      newDSPI->insertBefore(InsertPos);
    }
  }
}
