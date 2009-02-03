//===- LoopIndexSplit.cpp - Loop Index Splitting Pass ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Loop Index Splitting Pass. This pass handles three
// kinds of loops.
//
// [1] Loop is eliminated when loop body is executed only once. For example,
// for (i = 0; i < N; ++i) {
//   if ( i == X) {
//     ...
//   }
// }
//
// [2] Loop's iteration space is shrunk if loop body is executed for certain
//     range only. For example,
// 
// for (i = 0; i < N; ++i) {
//   if ( i > A && i < B) {
//     ...
//   }
// }
// is trnasformed to iterators from A to B, if A > 0 and B < N.
//
// [3] Loop is split if the loop body is dominated by an branch. For example,
//
// for (i = LB; i < UB; ++i) { if (i < SV) A; else B; }
//
// is transformed into
// AEV = BSV = SV
// for (i = LB; i < min(UB, AEV); ++i)
//    A;
// for (i = max(LB, BSV); i < UB; ++i);
//    B;
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-index-split"

#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"

using namespace llvm;

STATISTIC(NumIndexSplit, "Number of loop index split");
STATISTIC(NumIndexSplitRemoved, "Number of loops eliminated by loop index split");
STATISTIC(NumRestrictBounds, "Number of loop iteration space restricted");

namespace {

  class VISIBILITY_HIDDEN LoopIndexSplit : public LoopPass {

  public:
    static char ID; // Pass ID, replacement for typeid
    LoopIndexSplit() : LoopPass(&ID) {}

    // Index split Loop L. Return true if loop is split.
    bool runOnLoop(Loop *L, LPPassManager &LPM);

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<ScalarEvolution>();
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<DominatorTree>();
      AU.addRequired<DominanceFrontier>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<DominanceFrontier>();
    }

  private:
    /// processOneIterationLoop -- Eliminate loop if loop body is executed 
    /// only once. For example,
    /// for (i = 0; i < N; ++i) {
    ///   if ( i == X) {
    ///     ...
    ///   }
    /// }
    ///
    bool processOneIterationLoop();

    // -- Routines used by updateLoopIterationSpace();

    /// updateLoopIterationSpace -- Update loop's iteration space if loop 
    /// body is executed for certain IV range only. For example,
    /// 
    /// for (i = 0; i < N; ++i) {
    ///   if ( i > A && i < B) {
    ///     ...
    ///   }
    /// }
    /// is transformed to iterators from A to B, if A > 0 and B < N.
    ///
    bool updateLoopIterationSpace();

    /// restrictLoopBound - Op dominates loop body. Op compares an IV based value
    /// with a loop invariant value. Update loop's lower and upper bound based on
    /// the loop invariant value.
    bool restrictLoopBound(ICmpInst &Op);

    // --- Routines used by splitLoop(). --- /

    bool splitLoop();

    /// removeBlocks - Remove basic block DeadBB and all blocks dominated by 
    /// DeadBB. This routine is used to remove split condition's dead branch, 
    /// dominated by DeadBB. LiveBB dominates split conidition's other branch.
    void removeBlocks(BasicBlock *DeadBB, Loop *LP, BasicBlock *LiveBB);
    
    /// moveExitCondition - Move exit condition EC into split condition block.
    void moveExitCondition(BasicBlock *CondBB, BasicBlock *ActiveBB,
                           BasicBlock *ExitBB, ICmpInst *EC, ICmpInst *SC,
                           PHINode *IV, Instruction *IVAdd, Loop *LP,
                           unsigned);
    
    /// updatePHINodes - CFG has been changed. 
    /// Before 
    ///   - ExitBB's single predecessor was Latch
    ///   - Latch's second successor was Header
    /// Now
    ///   - ExitBB's single predecessor was Header
    ///   - Latch's one and only successor was Header
    ///
    /// Update ExitBB PHINodes' to reflect this change.
    void updatePHINodes(BasicBlock *ExitBB, BasicBlock *Latch, 
                        BasicBlock *Header,
                        PHINode *IV, Instruction *IVIncrement, Loop *LP);

    // --- Utility routines --- /

    /// cleanBlock - A block is considered clean if all non terminal 
    /// instructions are either PHINodes or IV based values.
    bool cleanBlock(BasicBlock *BB);

    /// IVisLT - If Op is comparing IV based value with an loop invariant and 
    /// IV based value is less than  the loop invariant then return the loop 
    /// invariant. Otherwise return NULL.
    Value * IVisLT(ICmpInst &Op);

    /// IVisLE - If Op is comparing IV based value with an loop invariant and 
    /// IV based value is less than or equal to the loop invariant then 
    /// return the loop invariant. Otherwise return NULL.
    Value * IVisLE(ICmpInst &Op);

    /// IVisGT - If Op is comparing IV based value with an loop invariant and 
    /// IV based value is greater than  the loop invariant then return the loop 
    /// invariant. Otherwise return NULL.
    Value * IVisGT(ICmpInst &Op);

    /// IVisGE - If Op is comparing IV based value with an loop invariant and 
    /// IV based value is greater than or equal to the loop invariant then 
    /// return the loop invariant. Otherwise return NULL.
    Value * IVisGE(ICmpInst &Op);

  private:

    // Current Loop information.
    Loop *L;
    LPPassManager *LPM;
    LoopInfo *LI;
    ScalarEvolution *SE;
    DominatorTree *DT;
    DominanceFrontier *DF;

    PHINode *IndVar;
    ICmpInst *ExitCondition;
    ICmpInst *SplitCondition;
    Value *IVStartValue;
    Value *IVExitValue;
    Instruction *IVIncrement;
    SmallPtrSet<Value *, 4> IVBasedValues;
  };
}

char LoopIndexSplit::ID = 0;
static RegisterPass<LoopIndexSplit>
X("loop-index-split", "Index Split Loops");

Pass *llvm::createLoopIndexSplitPass() {
  return new LoopIndexSplit();
}

// Index split Loop L. Return true if loop is split.
bool LoopIndexSplit::runOnLoop(Loop *IncomingLoop, LPPassManager &LPM_Ref) {
  L = IncomingLoop;
  LPM = &LPM_Ref;

  // FIXME - Nested loops make dominator info updates tricky. 
  if (!L->getSubLoops().empty())
    return false;

  SE = &getAnalysis<ScalarEvolution>();
  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();
  DF = &getAnalysis<DominanceFrontier>();

  // Initialize loop data.
  IndVar = L->getCanonicalInductionVariable();
  if (!IndVar) return false;

  bool P1InLoop = L->contains(IndVar->getIncomingBlock(1));
  IVStartValue = IndVar->getIncomingValue(!P1InLoop);
  IVIncrement = dyn_cast<Instruction>(IndVar->getIncomingValue(P1InLoop));
  if (!IVIncrement) return false;
  
  IVBasedValues.clear();
  IVBasedValues.insert(IndVar);
  IVBasedValues.insert(IVIncrement);
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) 
    for(BasicBlock::iterator BI = (*I)->begin(), BE = (*I)->end(); 
        BI != BE; ++BI) {
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(BI)) 
        if (BO != IVIncrement 
            && (BO->getOpcode() == Instruction::Add
                || BO->getOpcode() == Instruction::Sub))
          if (IVBasedValues.count(BO->getOperand(0))
              && L->isLoopInvariant(BO->getOperand(1)))
            IVBasedValues.insert(BO);
    }

  // Reject loop if loop exit condition is not suitable.
  SmallVector<BasicBlock *, 2> EBs;
  L->getExitingBlocks(EBs);
  if (EBs.size() != 1)
    return false;
  BranchInst *EBR = dyn_cast<BranchInst>(EBs[0]->getTerminator());
  if (!EBR) return false;
  ExitCondition = dyn_cast<ICmpInst>(EBR->getCondition());
  if (!ExitCondition) return false;
  if (EBs[0] != L->getLoopLatch()) return false;
  IVExitValue = ExitCondition->getOperand(1);
  if (!L->isLoopInvariant(IVExitValue))
    IVExitValue = ExitCondition->getOperand(0);
  if (!L->isLoopInvariant(IVExitValue))
    return false;

  // If start value is more then exit value where induction variable
  // increments by 1 then we are potentially dealing with an infinite loop.
  // Do not index split this loop.
  if (ConstantInt *SV = dyn_cast<ConstantInt>(IVStartValue))
    if (ConstantInt *EV = dyn_cast<ConstantInt>(IVExitValue))
      if (SV->getSExtValue() > EV->getSExtValue())
        return false;

  if (processOneIterationLoop())
    return true;

  if (updateLoopIterationSpace())
    return true;

  if (splitLoop())
    return true;

  return false;
}

// --- Helper routines --- 
// isUsedOutsideLoop - Returns true iff V is used outside the loop L.
static bool isUsedOutsideLoop(Value *V, Loop *L) {
  for(Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ++UI)
    if (!L->contains(cast<Instruction>(*UI)->getParent()))
      return true;
  return false;
}

// Return V+1
static Value *getPlusOne(Value *V, bool Sign, Instruction *InsertPt) {
  ConstantInt *One = ConstantInt::get(V->getType(), 1, Sign);
  return BinaryOperator::CreateAdd(V, One, "lsp", InsertPt);
}

// Return V-1
static Value *getMinusOne(Value *V, bool Sign, Instruction *InsertPt) {
  ConstantInt *One = ConstantInt::get(V->getType(), 1, Sign);
  return BinaryOperator::CreateSub(V, One, "lsp", InsertPt);
}

// Return min(V1, V1)
static Value *getMin(Value *V1, Value *V2, bool Sign, Instruction *InsertPt) {
 
  Value *C = new ICmpInst(Sign ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                          V1, V2, "lsp", InsertPt);
  return SelectInst::Create(C, V1, V2, "lsp", InsertPt);
}

// Return max(V1, V2)
static Value *getMax(Value *V1, Value *V2, bool Sign, Instruction *InsertPt) {
 
  Value *C = new ICmpInst(Sign ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                          V1, V2, "lsp", InsertPt);
  return SelectInst::Create(C, V2, V1, "lsp", InsertPt);
}

/// processOneIterationLoop -- Eliminate loop if loop body is executed 
/// only once. For example,
/// for (i = 0; i < N; ++i) {
///   if ( i == X) {
///     ...
///   }
/// }
///
bool LoopIndexSplit::processOneIterationLoop() {
  SplitCondition = NULL;
  BasicBlock *Latch = L->getLoopLatch();
  BasicBlock *Header = L->getHeader();
  BranchInst *BR = dyn_cast<BranchInst>(Header->getTerminator());
  if (!BR) return false;
  if (!isa<BranchInst>(Latch->getTerminator())) return false;
  if (BR->isUnconditional()) return false;
  SplitCondition = dyn_cast<ICmpInst>(BR->getCondition());
  if (!SplitCondition) return false;
  if (SplitCondition == ExitCondition) return false;
  if (SplitCondition->getPredicate() != ICmpInst::ICMP_EQ) return false;
  if (BR->getOperand(1) != Latch) return false;
  if (!IVBasedValues.count(SplitCondition->getOperand(0))
      && !IVBasedValues.count(SplitCondition->getOperand(1)))
    return false;

  // If IV is used outside the loop then this loop traversal is required.
  // FIXME: Calculate and use last IV value. 
  if (isUsedOutsideLoop(IVIncrement, L))
    return false;

  // If BR operands are not IV or not loop invariants then skip this loop.
  Value *OPV = SplitCondition->getOperand(0);
  Value *SplitValue = SplitCondition->getOperand(1);
  if (!L->isLoopInvariant(SplitValue)) {
    Value *T = SplitValue;
    SplitValue = OPV;
    OPV = T;
  }
  if (!L->isLoopInvariant(SplitValue))
    return false;
  Instruction *OPI = dyn_cast<Instruction>(OPV);
  if (!OPI) return false;
  if (OPI->getParent() != Header || isUsedOutsideLoop(OPI, L))
    return false;
  
  if (!cleanBlock(Header))
    return false;

  if (!cleanBlock(Latch))
    return false;
    
  // If the merge point for BR is not loop latch then skip this loop.
  if (BR->getSuccessor(0) != Latch) {
    DominanceFrontier::iterator DF0 = DF->find(BR->getSuccessor(0));
    assert (DF0 != DF->end() && "Unable to find dominance frontier");
    if (!DF0->second.count(Latch))
      return false;
  }
  
  if (BR->getSuccessor(1) != Latch) {
    DominanceFrontier::iterator DF1 = DF->find(BR->getSuccessor(1));
    assert (DF1 != DF->end() && "Unable to find dominance frontier");
    if (!DF1->second.count(Latch))
      return false;
  }
    
  // Now, Current loop L contains compare instruction
  // that compares induction variable, IndVar, against loop invariant. And
  // entire (i.e. meaningful) loop body is dominated by this compare
  // instruction. In such case eliminate 
  // loop structure surrounding this loop body. For example,
  //     for (int i = start; i < end; ++i) {
  //         if ( i == somevalue) {
  //           loop_body
  //         }
  //     }
  // can be transformed into
  //     if (somevalue >= start && somevalue < end) {
  //        i = somevalue;
  //        loop_body
  //     }

  // Replace index variable with split value in loop body. Loop body is executed
  // only when index variable is equal to split value.
  IndVar->replaceAllUsesWith(SplitValue);

  // Replace split condition in header.
  // Transform 
  //      SplitCondition : icmp eq i32 IndVar, SplitValue
  // into
  //      c1 = icmp uge i32 SplitValue, StartValue
  //      c2 = icmp ult i32 SplitValue, ExitValue
  //      and i32 c1, c2 
  Instruction *C1 = new ICmpInst(ExitCondition->isSignedPredicate() ? 
                                 ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE,
                                 SplitValue, IVStartValue, "lisplit", BR);

  CmpInst::Predicate C2P  = ExitCondition->getPredicate();
  BranchInst *LatchBR = cast<BranchInst>(Latch->getTerminator());
  if (LatchBR->getOperand(0) != Header)
    C2P = CmpInst::getInversePredicate(C2P);
  Instruction *C2 = new ICmpInst(C2P, SplitValue, IVExitValue, "lisplit", BR);
  Instruction *NSplitCond = BinaryOperator::CreateAnd(C1, C2, "lisplit", BR);

  SplitCondition->replaceAllUsesWith(NSplitCond);
  SplitCondition->eraseFromParent();

  // Remove Latch to Header edge.
  BasicBlock *LatchSucc = NULL;
  Header->removePredecessor(Latch);
  for (succ_iterator SI = succ_begin(Latch), E = succ_end(Latch);
       SI != E; ++SI) {
    if (Header != *SI)
      LatchSucc = *SI;
  }
  LatchBR->setUnconditionalDest(LatchSucc);

  // Remove IVIncrement
  IVIncrement->replaceAllUsesWith(UndefValue::get(IVIncrement->getType()));
  IVIncrement->eraseFromParent();
  
  LPM->deleteLoopFromQueue(L);

  // Update Dominator Info.
  // Only CFG change done is to remove Latch to Header edge. This
  // does not change dominator tree because Latch did not dominate
  // Header.
  if (DF) {
    DominanceFrontier::iterator HeaderDF = DF->find(Header);
    if (HeaderDF != DF->end()) 
      DF->removeFromFrontier(HeaderDF, Header);

    DominanceFrontier::iterator LatchDF = DF->find(Latch);
    if (LatchDF != DF->end()) 
      DF->removeFromFrontier(LatchDF, Header);
  }

  ++NumIndexSplitRemoved;
  return true;
}

/// restrictLoopBound - Op dominates loop body. Op compares an IV based value 
/// with a loop invariant value. Update loop's lower and upper bound based on 
/// the loop invariant value.
bool LoopIndexSplit::restrictLoopBound(ICmpInst &Op) {
  bool Sign = Op.isSignedPredicate();
  Instruction *PHTerm = L->getLoopPreheader()->getTerminator();

  if (IVisGT(*ExitCondition) || IVisGE(*ExitCondition)) {
    BranchInst *EBR = 
      cast<BranchInst>(ExitCondition->getParent()->getTerminator());
    ExitCondition->setPredicate(ExitCondition->getInversePredicate());
    BasicBlock *T = EBR->getSuccessor(0);
    EBR->setSuccessor(0, EBR->getSuccessor(1));
    EBR->setSuccessor(1, T);
  }

  // New upper and lower bounds.
  Value *NLB = NULL;
  Value *NUB = NULL;
  if (Value *V = IVisLT(Op)) {
    // Restrict upper bound.
    if (IVisLE(*ExitCondition)) 
      V = getMinusOne(V, Sign, PHTerm);
    NUB = getMin(V, IVExitValue, Sign, PHTerm);
  } else if (Value *V = IVisLE(Op)) {
    // Restrict upper bound.
    if (IVisLT(*ExitCondition)) 
      V = getPlusOne(V, Sign, PHTerm);
    NUB = getMin(V, IVExitValue, Sign, PHTerm);
  } else if (Value *V = IVisGT(Op)) {
    // Restrict lower bound.
    V = getPlusOne(V, Sign, PHTerm);
    NLB = getMax(V, IVStartValue, Sign, PHTerm);
  } else if (Value *V = IVisGE(Op))
    // Restrict lower bound.
    NLB = getMax(V, IVStartValue, Sign, PHTerm);

  if (!NLB && !NUB) 
    return false;

  if (NLB) {
    unsigned i = IndVar->getBasicBlockIndex(L->getLoopPreheader());
    IndVar->setIncomingValue(i, NLB);
  }

  if (NUB) {
    unsigned i = (ExitCondition->getOperand(0) != IVExitValue);
    ExitCondition->setOperand(i, NUB);
  }
  return true;
}

/// updateLoopIterationSpace -- Update loop's iteration space if loop 
/// body is executed for certain IV range only. For example,
/// 
/// for (i = 0; i < N; ++i) {
///   if ( i > A && i < B) {
///     ...
///   }
/// }
/// is transformed to iterators from A to B, if A > 0 and B < N.
///
bool LoopIndexSplit::updateLoopIterationSpace() {
  SplitCondition = NULL;
  if (ExitCondition->getPredicate() == ICmpInst::ICMP_NE
      || ExitCondition->getPredicate() == ICmpInst::ICMP_EQ)
    return false;
  BasicBlock *Latch = L->getLoopLatch();
  BasicBlock *Header = L->getHeader();
  BranchInst *BR = dyn_cast<BranchInst>(Header->getTerminator());
  if (!BR) return false;
  if (!isa<BranchInst>(Latch->getTerminator())) return false;
  if (BR->isUnconditional()) return false;
  BinaryOperator *AND = dyn_cast<BinaryOperator>(BR->getCondition());
  if (!AND) return false;
  if (AND->getOpcode() != Instruction::And) return false;
  ICmpInst *Op0 = dyn_cast<ICmpInst>(AND->getOperand(0));
  ICmpInst *Op1 = dyn_cast<ICmpInst>(AND->getOperand(1));
  if (!Op0 || !Op1)
    return false;
  IVBasedValues.insert(AND);
  IVBasedValues.insert(Op0);
  IVBasedValues.insert(Op1);
  if (!cleanBlock(Header)) return false;
  BasicBlock *ExitingBlock = ExitCondition->getParent();
  if (!cleanBlock(ExitingBlock)) return false;

  // Verify that loop exiting block has only two predecessor, where one pred
  // is split condition block. The other predecessor will become exiting block's
  // dominator after CFG is updated. TODO : Handle CFG's where exiting block has
  // more then two predecessors. This requires extra work in updating dominator
  // information.
  BasicBlock *ExitingBBPred = NULL;
  for (pred_iterator PI = pred_begin(ExitingBlock), PE = pred_end(ExitingBlock);
       PI != PE; ++PI) {
    BasicBlock *BB = *PI;
    if (Header == BB)
      continue;
    if (ExitingBBPred)
      return false;
    else
      ExitingBBPred = BB;
  }

  if (!restrictLoopBound(*Op0))
    return false;

  if (!restrictLoopBound(*Op1))
    return false;

  // Update CFG.
  if (BR->getSuccessor(0) == ExitingBlock)
    BR->setUnconditionalDest(BR->getSuccessor(1));
  else
    BR->setUnconditionalDest(BR->getSuccessor(0));

  AND->eraseFromParent();
  if (Op0->use_empty())
    Op0->eraseFromParent();
  if (Op1->use_empty())
    Op1->eraseFromParent();

  // Update domiantor info. Now, ExitingBlock has only one predecessor, 
  // ExitingBBPred, and it is ExitingBlock's immediate domiantor.
  DT->changeImmediateDominator(ExitingBlock, ExitingBBPred);

  BasicBlock *ExitBlock = ExitingBlock->getTerminator()->getSuccessor(1);
  if (L->contains(ExitBlock))
    ExitBlock = ExitingBlock->getTerminator()->getSuccessor(0);

  // If ExitingBlock is a member of the loop basic blocks' DF list then
  // replace ExitingBlock with header and exit block in the DF list
  DominanceFrontier::iterator ExitingBlockDF = DF->find(ExitingBlock);
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    if (BB == Header || BB == ExitingBlock)
      continue;
    DominanceFrontier::iterator BBDF = DF->find(BB);
    DominanceFrontier::DomSetType::iterator DomSetI = BBDF->second.begin();
    DominanceFrontier::DomSetType::iterator DomSetE = BBDF->second.end();
    while (DomSetI != DomSetE) {
      DominanceFrontier::DomSetType::iterator CurrentItr = DomSetI;
      ++DomSetI;
      BasicBlock *DFBB = *CurrentItr;
      if (DFBB == ExitingBlock) {
        BBDF->second.erase(DFBB);
        for (DominanceFrontier::DomSetType::iterator 
               EBI = ExitingBlockDF->second.begin(),
               EBE = ExitingBlockDF->second.end(); EBI != EBE; ++EBI) 
          BBDF->second.insert(*EBI);
      }
    }
  }
  NumRestrictBounds++;
  return true;
}

/// removeBlocks - Remove basic block DeadBB and all blocks dominated by DeadBB.
/// This routine is used to remove split condition's dead branch, dominated by
/// DeadBB. LiveBB dominates split conidition's other branch.
void LoopIndexSplit::removeBlocks(BasicBlock *DeadBB, Loop *LP, 
                                  BasicBlock *LiveBB) {

  // First update DeadBB's dominance frontier. 
  SmallVector<BasicBlock *, 8> FrontierBBs;
  DominanceFrontier::iterator DeadBBDF = DF->find(DeadBB);
  if (DeadBBDF != DF->end()) {
    SmallVector<BasicBlock *, 8> PredBlocks;
    
    DominanceFrontier::DomSetType DeadBBSet = DeadBBDF->second;
    for (DominanceFrontier::DomSetType::iterator DeadBBSetI = DeadBBSet.begin(),
           DeadBBSetE = DeadBBSet.end(); DeadBBSetI != DeadBBSetE; ++DeadBBSetI) 
      {
      BasicBlock *FrontierBB = *DeadBBSetI;
      FrontierBBs.push_back(FrontierBB);

      // Rremove any PHI incoming edge from blocks dominated by DeadBB.
      PredBlocks.clear();
      for(pred_iterator PI = pred_begin(FrontierBB), PE = pred_end(FrontierBB);
          PI != PE; ++PI) {
        BasicBlock *P = *PI;
        if (P == DeadBB || DT->dominates(DeadBB, P))
          PredBlocks.push_back(P);
      }

      for(BasicBlock::iterator FBI = FrontierBB->begin(), FBE = FrontierBB->end();
          FBI != FBE; ++FBI) {
        if (PHINode *PN = dyn_cast<PHINode>(FBI)) {
          for(SmallVector<BasicBlock *, 8>::iterator PI = PredBlocks.begin(),
                PE = PredBlocks.end(); PI != PE; ++PI) {
            BasicBlock *P = *PI;
            PN->removeIncomingValue(P);
          }
        }
        else
          break;
      }      
    }
  }
  
  // Now remove DeadBB and all nodes dominated by DeadBB in df order.
  SmallVector<BasicBlock *, 32> WorkList;
  DomTreeNode *DN = DT->getNode(DeadBB);
  for (df_iterator<DomTreeNode*> DI = df_begin(DN),
         E = df_end(DN); DI != E; ++DI) {
    BasicBlock *BB = DI->getBlock();
    WorkList.push_back(BB);
    BB->replaceAllUsesWith(UndefValue::get(Type::LabelTy));
  }

  while (!WorkList.empty()) {
    BasicBlock *BB = WorkList.back(); WorkList.pop_back();
    for(BasicBlock::iterator BBI = BB->begin(), BBE = BB->end(); 
        BBI != BBE; ) {
      Instruction *I = BBI;
      ++BBI;
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
    }
    LPM->deleteSimpleAnalysisValue(BB, LP);
    DT->eraseNode(BB);
    DF->removeBlock(BB);
    LI->removeBlock(BB);
    BB->eraseFromParent();
  }

  // Update Frontier BBs' dominator info.
  while (!FrontierBBs.empty()) {
    BasicBlock *FBB = FrontierBBs.back(); FrontierBBs.pop_back();
    BasicBlock *NewDominator = FBB->getSinglePredecessor();
    if (!NewDominator) {
      pred_iterator PI = pred_begin(FBB), PE = pred_end(FBB);
      NewDominator = *PI;
      ++PI;
      if (NewDominator != LiveBB) {
        for(; PI != PE; ++PI) {
          BasicBlock *P = *PI;
          if (P == LiveBB) {
            NewDominator = LiveBB;
            break;
          }
          NewDominator = DT->findNearestCommonDominator(NewDominator, P);
        }
      }
    }
    assert (NewDominator && "Unable to fix dominator info.");
    DT->changeImmediateDominator(FBB, NewDominator);
    DF->changeImmediateDominator(FBB, NewDominator, DT);
  }

}

// moveExitCondition - Move exit condition EC into split condition block CondBB.
void LoopIndexSplit::moveExitCondition(BasicBlock *CondBB, BasicBlock *ActiveBB,
                                       BasicBlock *ExitBB, ICmpInst *EC, 
                                       ICmpInst *SC, PHINode *IV, 
                                       Instruction *IVAdd, Loop *LP,
                                       unsigned ExitValueNum) {

  BasicBlock *ExitingBB = EC->getParent();
  Instruction *CurrentBR = CondBB->getTerminator();

  // Move exit condition into split condition block.
  EC->moveBefore(CurrentBR);
  EC->setOperand(ExitValueNum == 0 ? 1 : 0, IV);

  // Move exiting block's branch into split condition block. Update its branch
  // destination.
  BranchInst *ExitingBR = cast<BranchInst>(ExitingBB->getTerminator());
  ExitingBR->moveBefore(CurrentBR);
  BasicBlock *OrigDestBB = NULL;
  if (ExitingBR->getSuccessor(0) == ExitBB) {
    OrigDestBB = ExitingBR->getSuccessor(1);
    ExitingBR->setSuccessor(1, ActiveBB);
  }
  else {
    OrigDestBB = ExitingBR->getSuccessor(0);
    ExitingBR->setSuccessor(0, ActiveBB);
  }
    
  // Remove split condition and current split condition branch.
  SC->eraseFromParent();
  CurrentBR->eraseFromParent();

  // Connect exiting block to original destination.
  BranchInst::Create(OrigDestBB, ExitingBB);

  // Update PHINodes
  updatePHINodes(ExitBB, ExitingBB, CondBB, IV, IVAdd, LP);

  // Fix dominator info.
  // ExitBB is now dominated by CondBB
  DT->changeImmediateDominator(ExitBB, CondBB);
  DF->changeImmediateDominator(ExitBB, CondBB, DT);
  
  // Basicblocks dominated by ActiveBB may have ExitingBB or
  // a basic block outside the loop in their DF list. If so,
  // replace it with CondBB.
  DomTreeNode *Node = DT->getNode(ActiveBB);
  for (df_iterator<DomTreeNode *> DI = df_begin(Node), DE = df_end(Node);
       DI != DE; ++DI) {
    BasicBlock *BB = DI->getBlock();
    DominanceFrontier::iterator BBDF = DF->find(BB);
    DominanceFrontier::DomSetType::iterator DomSetI = BBDF->second.begin();
    DominanceFrontier::DomSetType::iterator DomSetE = BBDF->second.end();
    while (DomSetI != DomSetE) {
      DominanceFrontier::DomSetType::iterator CurrentItr = DomSetI;
      ++DomSetI;
      BasicBlock *DFBB = *CurrentItr;
      if (DFBB == ExitingBB || !L->contains(DFBB)) {
        BBDF->second.erase(DFBB);
        BBDF->second.insert(CondBB);
      }
    }
  }
}

/// updatePHINodes - CFG has been changed. 
/// Before 
///   - ExitBB's single predecessor was Latch
///   - Latch's second successor was Header
/// Now
///   - ExitBB's single predecessor is Header
///   - Latch's one and only successor is Header
///
/// Update ExitBB PHINodes' to reflect this change.
void LoopIndexSplit::updatePHINodes(BasicBlock *ExitBB, BasicBlock *Latch, 
                                    BasicBlock *Header,
                                    PHINode *IV, Instruction *IVIncrement,
                                    Loop *LP) {

  for (BasicBlock::iterator BI = ExitBB->begin(), BE = ExitBB->end(); 
       BI != BE; ) {
    PHINode *PN = dyn_cast<PHINode>(BI);
    ++BI;
    if (!PN)
      break;

    Value *V = PN->getIncomingValueForBlock(Latch);
    if (PHINode *PHV = dyn_cast<PHINode>(V)) {
      // PHV is in Latch. PHV has one use is in ExitBB PHINode. And one use
      // in Header which is new incoming value for PN.
      Value *NewV = NULL;
      for (Value::use_iterator UI = PHV->use_begin(), E = PHV->use_end(); 
           UI != E; ++UI) 
        if (PHINode *U = dyn_cast<PHINode>(*UI)) 
          if (LP->contains(U->getParent())) {
            NewV = U;
            break;
          }

      // Add incoming value from header only if PN has any use inside the loop.
      if (NewV)
        PN->addIncoming(NewV, Header);

    } else if (Instruction *PHI = dyn_cast<Instruction>(V)) {
      // If this instruction is IVIncrement then IV is new incoming value 
      // from header otherwise this instruction must be incoming value from 
      // header because loop is in LCSSA form.
      if (PHI == IVIncrement)
        PN->addIncoming(IV, Header);
      else
        PN->addIncoming(V, Header);
    } else
      // Otherwise this is an incoming value from header because loop is in 
      // LCSSA form.
      PN->addIncoming(V, Header);
    
    // Remove incoming value from Latch.
    PN->removeIncomingValue(Latch);
  }
}

bool LoopIndexSplit::splitLoop() {
  SplitCondition = NULL;
  if (ExitCondition->getPredicate() == ICmpInst::ICMP_NE
      || ExitCondition->getPredicate() == ICmpInst::ICMP_EQ)
    return false;
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  BranchInst *SBR = NULL; // Split Condition Branch
  BranchInst *EBR = cast<BranchInst>(ExitCondition->getParent()->getTerminator());
  // If Exiting block includes loop variant instructions then this
  // loop may not be split safely.
  BasicBlock *ExitingBlock = ExitCondition->getParent();
  if (!cleanBlock(ExitingBlock)) return false;

  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BranchInst *BR = dyn_cast<BranchInst>((*I)->getTerminator());
    if (!BR || BR->isUnconditional()) continue;
    ICmpInst *CI = dyn_cast<ICmpInst>(BR->getCondition());
    if (!CI || CI == ExitCondition 
        || CI->getPredicate() == ICmpInst::ICMP_NE
        || CI->getPredicate() == ICmpInst::ICMP_EQ)
      continue;

    // Unable to handle triangle loops at the moment.
    // In triangle loop, split condition is in header and one of the
    // the split destination is loop latch. If split condition is EQ
    // then such loops are already handle in processOneIterationLoop().
    if (Header == (*I)
        && (Latch == BR->getSuccessor(0) || Latch == BR->getSuccessor(1)))
      continue;

    // If the block does not dominate the latch then this is not a diamond.
    // Such loop may not benefit from index split.
    if (!DT->dominates((*I), Latch))
      continue;

    // If split condition branches heads do not have single predecessor, 
    // SplitCondBlock, then is not possible to remove inactive branch.
    if (!BR->getSuccessor(0)->getSinglePredecessor() 
        || !BR->getSuccessor(1)->getSinglePredecessor())
      return false;

    // If the merge point for BR is not loop latch then skip this condition.
    if (BR->getSuccessor(0) != Latch) {
      DominanceFrontier::iterator DF0 = DF->find(BR->getSuccessor(0));
      assert (DF0 != DF->end() && "Unable to find dominance frontier");
      if (!DF0->second.count(Latch))
        continue;
    }
    
    if (BR->getSuccessor(1) != Latch) {
      DominanceFrontier::iterator DF1 = DF->find(BR->getSuccessor(1));
      assert (DF1 != DF->end() && "Unable to find dominance frontier");
      if (!DF1->second.count(Latch))
        continue;
    }
    SplitCondition = CI;
    SBR = BR;
    break;
  }
   
  if (!SplitCondition)
    return false;

  // If the predicate sign does not match then skip.
  if (ExitCondition->isSignedPredicate() != SplitCondition->isSignedPredicate())
    return false;

  unsigned EVOpNum = (ExitCondition->getOperand(1) == IVExitValue);
  unsigned SVOpNum = IVBasedValues.count(SplitCondition->getOperand(0));
  Value *SplitValue = SplitCondition->getOperand(SVOpNum);
  if (!L->isLoopInvariant(SplitValue))
    return false;
  if (!IVBasedValues.count(SplitCondition->getOperand(!SVOpNum)))
    return false;

  // Normalize loop conditions so that it is easier to calculate new loop
  // bounds.
  if (IVisGT(*ExitCondition) || IVisGE(*ExitCondition)) {
    ExitCondition->setPredicate(ExitCondition->getInversePredicate());
    BasicBlock *T = EBR->getSuccessor(0);
    EBR->setSuccessor(0, EBR->getSuccessor(1));
    EBR->setSuccessor(1, T);
  }

  if (IVisGT(*SplitCondition) || IVisGE(*SplitCondition)) {
    SplitCondition->setPredicate(SplitCondition->getInversePredicate());
    BasicBlock *T = SBR->getSuccessor(0);
    SBR->setSuccessor(0, SBR->getSuccessor(1));
    SBR->setSuccessor(1, T);
  }

  //[*] Calculate new loop bounds.
  Value *AEV = SplitValue;
  Value *BSV = SplitValue;
  bool Sign = SplitCondition->isSignedPredicate();
  Instruction *PHTerm = L->getLoopPreheader()->getTerminator();

  if (IVisLT(*ExitCondition)) {
    if (IVisLT(*SplitCondition)) {
      /* Do nothing */
    }
    else if (IVisLE(*SplitCondition)) {
      AEV = getPlusOne(SplitValue, Sign, PHTerm);
      BSV = getPlusOne(SplitValue, Sign, PHTerm);
    } else {
      assert (0 && "Unexpected split condition!");
    }
  }
  else if (IVisLE(*ExitCondition)) {
    if (IVisLT(*SplitCondition)) {
      AEV = getMinusOne(SplitValue, Sign, PHTerm);
    }
    else if (IVisLE(*SplitCondition)) {
      BSV = getPlusOne(SplitValue, Sign, PHTerm);
    } else {
      assert (0 && "Unexpected split condition!");
    }
  } else {
    assert (0 && "Unexpected exit condition!");
  }
  AEV = getMin(AEV, IVExitValue, Sign, PHTerm);
  BSV = getMax(BSV, IVStartValue, Sign, PHTerm);

  // [*] Clone Loop
  DenseMap<const Value *, Value *> ValueMap;
  Loop *BLoop = CloneLoop(L, LPM, LI, ValueMap, this);
  Loop *ALoop = L;

  // [*] ALoop's exiting edge enters BLoop's header.
  //    ALoop's original exit block becomes BLoop's exit block.
  PHINode *B_IndVar = cast<PHINode>(ValueMap[IndVar]);
  BasicBlock *A_ExitingBlock = ExitCondition->getParent();
  BranchInst *A_ExitInsn =
    dyn_cast<BranchInst>(A_ExitingBlock->getTerminator());
  assert (A_ExitInsn && "Unable to find suitable loop exit branch");
  BasicBlock *B_ExitBlock = A_ExitInsn->getSuccessor(1);
  BasicBlock *B_Header = BLoop->getHeader();
  if (ALoop->contains(B_ExitBlock)) {
    B_ExitBlock = A_ExitInsn->getSuccessor(0);
    A_ExitInsn->setSuccessor(0, B_Header);
  } else
    A_ExitInsn->setSuccessor(1, B_Header);

  // [*] Update ALoop's exit value using new exit value.
  ExitCondition->setOperand(EVOpNum, AEV);

  // [*] Update BLoop's header phi nodes. Remove incoming PHINode's from
  //     original loop's preheader. Add incoming PHINode values from
  //     ALoop's exiting block. Update BLoop header's domiantor info.

  // Collect inverse map of Header PHINodes.
  DenseMap<Value *, Value *> InverseMap;
  for (BasicBlock::iterator BI = ALoop->getHeader()->begin(), 
         BE = ALoop->getHeader()->end(); BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      PHINode *PNClone = cast<PHINode>(ValueMap[PN]);
      InverseMap[PNClone] = PN;
    } else
      break;
  }

  BasicBlock *A_Preheader = ALoop->getLoopPreheader();
  for (BasicBlock::iterator BI = B_Header->begin(), BE = B_Header->end();
       BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      // Remove incoming value from original preheader.
      PN->removeIncomingValue(A_Preheader);

      // Add incoming value from A_ExitingBlock.
      if (PN == B_IndVar)
        PN->addIncoming(BSV, A_ExitingBlock);
      else { 
        PHINode *OrigPN = cast<PHINode>(InverseMap[PN]);
        Value *V2 = NULL;
        // If loop header is also loop exiting block then
        // OrigPN is incoming value for B loop header.
        if (A_ExitingBlock == ALoop->getHeader())
          V2 = OrigPN;
        else
          V2 = OrigPN->getIncomingValueForBlock(A_ExitingBlock);
        PN->addIncoming(V2, A_ExitingBlock);
      }
    } else
      break;
  }

  DT->changeImmediateDominator(B_Header, A_ExitingBlock);
  DF->changeImmediateDominator(B_Header, A_ExitingBlock, DT);
  
  // [*] Update BLoop's exit block. Its new predecessor is BLoop's exit
  //     block. Remove incoming PHINode values from ALoop's exiting block.
  //     Add new incoming values from BLoop's incoming exiting value.
  //     Update BLoop exit block's dominator info..
  BasicBlock *B_ExitingBlock = cast<BasicBlock>(ValueMap[A_ExitingBlock]);
  for (BasicBlock::iterator BI = B_ExitBlock->begin(), BE = B_ExitBlock->end();
       BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      PN->addIncoming(ValueMap[PN->getIncomingValueForBlock(A_ExitingBlock)], 
                                                            B_ExitingBlock);
      PN->removeIncomingValue(A_ExitingBlock);
    } else
      break;
  }

  DT->changeImmediateDominator(B_ExitBlock, B_ExitingBlock);
  DF->changeImmediateDominator(B_ExitBlock, B_ExitingBlock, DT);

 //[*] Split ALoop's exit edge. This creates a new block which
  //    serves two purposes. First one is to hold PHINode defnitions
  //    to ensure that ALoop's LCSSA form. Second use it to act
  //    as a preheader for BLoop.
  BasicBlock *A_ExitBlock = SplitEdge(A_ExitingBlock, B_Header, this);

  //[*] Preserve ALoop's LCSSA form. Create new forwarding PHINodes
  //    in A_ExitBlock to redefine outgoing PHI definitions from ALoop.
  for(BasicBlock::iterator BI = B_Header->begin(), BE = B_Header->end();
      BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      Value *V1 = PN->getIncomingValueForBlock(A_ExitBlock);
      PHINode *newPHI = PHINode::Create(PN->getType(), PN->getName());
      newPHI->addIncoming(V1, A_ExitingBlock);
      A_ExitBlock->getInstList().push_front(newPHI);
      PN->removeIncomingValue(A_ExitBlock);
      PN->addIncoming(newPHI, A_ExitBlock);
    } else
      break;
  }

  //[*] Eliminate split condition's inactive branch from ALoop.
  BasicBlock *A_SplitCondBlock = SplitCondition->getParent();
  BranchInst *A_BR = cast<BranchInst>(A_SplitCondBlock->getTerminator());
  BasicBlock *A_InactiveBranch = NULL;
  BasicBlock *A_ActiveBranch = NULL;
  A_ActiveBranch = A_BR->getSuccessor(0);
  A_InactiveBranch = A_BR->getSuccessor(1);
  A_BR->setUnconditionalDest(A_ActiveBranch);
  removeBlocks(A_InactiveBranch, L, A_ActiveBranch);

  //[*] Eliminate split condition's inactive branch in from BLoop.
  BasicBlock *B_SplitCondBlock = cast<BasicBlock>(ValueMap[A_SplitCondBlock]);
  BranchInst *B_BR = cast<BranchInst>(B_SplitCondBlock->getTerminator());
  BasicBlock *B_InactiveBranch = NULL;
  BasicBlock *B_ActiveBranch = NULL;
  B_ActiveBranch = B_BR->getSuccessor(1);
  B_InactiveBranch = B_BR->getSuccessor(0);
  B_BR->setUnconditionalDest(B_ActiveBranch);
  removeBlocks(B_InactiveBranch, BLoop, B_ActiveBranch);

  BasicBlock *A_Header = ALoop->getHeader();
  if (A_ExitingBlock == A_Header)
    return true;

  //[*] Move exit condition into split condition block to avoid
  //    executing dead loop iteration.
  ICmpInst *B_ExitCondition = cast<ICmpInst>(ValueMap[ExitCondition]);
  Instruction *B_IndVarIncrement = cast<Instruction>(ValueMap[IVIncrement]);
  ICmpInst *B_SplitCondition = cast<ICmpInst>(ValueMap[SplitCondition]);

  moveExitCondition(A_SplitCondBlock, A_ActiveBranch, A_ExitBlock, ExitCondition,
                    cast<ICmpInst>(SplitCondition), IndVar, IVIncrement, 
                    ALoop, EVOpNum);

  moveExitCondition(B_SplitCondBlock, B_ActiveBranch, 
                    B_ExitBlock, B_ExitCondition,
                    B_SplitCondition, B_IndVar, B_IndVarIncrement, 
                    BLoop, EVOpNum);

  NumIndexSplit++;
  return true;
}

/// cleanBlock - A block is considered clean if all non terminal instructions 
/// are either, PHINodes, IV based.
bool LoopIndexSplit::cleanBlock(BasicBlock *BB) {
  Instruction *Terminator = BB->getTerminator();
  for(BasicBlock::iterator BI = BB->begin(), BE = BB->end(); 
      BI != BE; ++BI) {
    Instruction *I = BI;

    if (isa<PHINode>(I) || I == Terminator || I == ExitCondition
        || I == SplitCondition || IVBasedValues.count(I))
      continue;

    if (I->mayWriteToMemory())
      return false;

    // I is used only inside this block then it is OK.
    bool usedOutsideBB = false;
    for (Value::use_iterator UI = I->use_begin(), UE = I->use_end(); 
         UI != UE; ++UI) {
      Instruction *U = cast<Instruction>(UI);
      if (U->getParent() != BB)
        usedOutsideBB = true;
    }
    if (!usedOutsideBB)
      continue;

    // Otherwise we have a instruction that may not allow loop spliting.
    return false;
  }
  return true;
}

/// IVisLT - If Op is comparing IV based value with an loop invariant and 
/// IV based value is less than  the loop invariant then return the loop 
/// invariant. Otherwise return NULL.
Value * LoopIndexSplit::IVisLT(ICmpInst &Op) {
  ICmpInst::Predicate P = Op.getPredicate();
  if ((P == ICmpInst::ICMP_SLT || P == ICmpInst::ICMP_ULT) 
      && IVBasedValues.count(Op.getOperand(0)) 
      && L->isLoopInvariant(Op.getOperand(1)))
    return Op.getOperand(1);

  if ((P == ICmpInst::ICMP_SGT || P == ICmpInst::ICMP_UGT) 
      && IVBasedValues.count(Op.getOperand(1)) 
      && L->isLoopInvariant(Op.getOperand(0)))
    return Op.getOperand(0);

  return NULL;
}

/// IVisLE - If Op is comparing IV based value with an loop invariant and 
/// IV based value is less than or equal to the loop invariant then 
/// return the loop invariant. Otherwise return NULL.
Value * LoopIndexSplit::IVisLE(ICmpInst &Op) {
  ICmpInst::Predicate P = Op.getPredicate();
  if ((P == ICmpInst::ICMP_SLE || P == ICmpInst::ICMP_ULE)
      && IVBasedValues.count(Op.getOperand(0)) 
      && L->isLoopInvariant(Op.getOperand(1)))
    return Op.getOperand(1);

  if ((P == ICmpInst::ICMP_SGE || P == ICmpInst::ICMP_UGE) 
      && IVBasedValues.count(Op.getOperand(1)) 
      && L->isLoopInvariant(Op.getOperand(0)))
    return Op.getOperand(0);

  return NULL;
}

/// IVisGT - If Op is comparing IV based value with an loop invariant and 
/// IV based value is greater than  the loop invariant then return the loop 
/// invariant. Otherwise return NULL.
Value * LoopIndexSplit::IVisGT(ICmpInst &Op) {
  ICmpInst::Predicate P = Op.getPredicate();
  if ((P == ICmpInst::ICMP_SGT || P == ICmpInst::ICMP_UGT) 
      && IVBasedValues.count(Op.getOperand(0)) 
      && L->isLoopInvariant(Op.getOperand(1)))
    return Op.getOperand(1);

  if ((P == ICmpInst::ICMP_SLT || P == ICmpInst::ICMP_ULT) 
      && IVBasedValues.count(Op.getOperand(1)) 
      && L->isLoopInvariant(Op.getOperand(0)))
    return Op.getOperand(0);

  return NULL;
}

/// IVisGE - If Op is comparing IV based value with an loop invariant and 
/// IV based value is greater than or equal to the loop invariant then 
/// return the loop invariant. Otherwise return NULL.
Value * LoopIndexSplit::IVisGE(ICmpInst &Op) {
  ICmpInst::Predicate P = Op.getPredicate();
  if ((P == ICmpInst::ICMP_SGE || P == ICmpInst::ICMP_UGE)
      && IVBasedValues.count(Op.getOperand(0)) 
      && L->isLoopInvariant(Op.getOperand(1)))
    return Op.getOperand(1);

  if ((P == ICmpInst::ICMP_SLE || P == ICmpInst::ICMP_ULE) 
      && IVBasedValues.count(Op.getOperand(1)) 
      && L->isLoopInvariant(Op.getOperand(0)))
    return Op.getOperand(0);

  return NULL;
}

