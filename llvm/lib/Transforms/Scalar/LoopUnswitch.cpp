//===-- LoopUnswitch.cpp - Hoist loop-invariant conditionals in loop ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass transforms loops that contain branches on loop-invariant conditions
// to have multiple loops.  For example, it turns the left into the right code:
//
//  for (...)                  if (lic)
//    A                          for (...)
//    if (lic)                     A; B; C
//      B                      else
//    C                          for (...)
//                                 A; C
//
// This can increase the size of the code exponentially (doubling it every time
// a loop is unswitched) so we only unswitch if the resultant code will be
// smaller than a threshold.
//
// This pass expects LICM to be run before it to hoist invariant conditions out
// of the loop, to make the unswitching opportunity obvious.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-unswitch"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <set>
using namespace llvm;

STATISTIC(NumBranches, "Number of branches unswitched");
STATISTIC(NumSwitches, "Number of switches unswitched");
STATISTIC(NumSelects , "Number of selects unswitched");
STATISTIC(NumTrivial , "Number of unswitches that are trivial");
STATISTIC(NumSimplify, "Number of simplifications of unswitched code");

static cl::opt<unsigned>
Threshold("loop-unswitch-threshold", cl::desc("Max loop size to unswitch"),
          cl::init(10), cl::Hidden);
  
namespace {
  class VISIBILITY_HIDDEN LoopUnswitch : public LoopPass {
    LoopInfo *LI;  // Loop information
    LPPassManager *LPM;

    // LoopProcessWorklist - Used to check if second loop needs processing
    // after RewriteLoopBodyWithConditionConstant rewrites first loop.
    std::vector<Loop*> LoopProcessWorklist;
    SmallPtrSet<Value *,8> UnswitchedVals;
    
    bool OptimizeForSize;
    bool redoLoop;

    Loop *currentLoop;
    DominanceFrontier *DF;
    DominatorTree *DT;
    BasicBlock *loopHeader;
    BasicBlock *loopPreheader;
    
    // LoopBlocks contains all of the basic blocks of the loop, including the
    // preheader of the loop, the body of the loop, and the exit blocks of the 
    // loop, in that order.
    std::vector<BasicBlock*> LoopBlocks;
    // NewBlocks contained cloned copy of basic blocks from LoopBlocks.
    std::vector<BasicBlock*> NewBlocks;

  public:
    static char ID; // Pass ID, replacement for typeid
    explicit LoopUnswitch(bool Os = false) : 
      LoopPass(&ID), OptimizeForSize(Os), redoLoop(false), 
      currentLoop(NULL), DF(NULL), DT(NULL), loopHeader(NULL),
      loopPreheader(NULL) {}

    bool runOnLoop(Loop *L, LPPassManager &LPM);
    bool processCurrentLoop();

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<DominanceFrontier>();
    }

  private:

    /// RemoveLoopFromWorklist - If the specified loop is on the loop worklist,
    /// remove it.
    void RemoveLoopFromWorklist(Loop *L) {
      std::vector<Loop*>::iterator I = std::find(LoopProcessWorklist.begin(),
                                                 LoopProcessWorklist.end(), L);
      if (I != LoopProcessWorklist.end())
        LoopProcessWorklist.erase(I);
    }

    void initLoopData() {
      loopHeader = currentLoop->getHeader();
      loopPreheader = currentLoop->getLoopPreheader();
    }

    /// Split all of the edges from inside the loop to their exit blocks.
    /// Update the appropriate Phi nodes as we do so.
    void SplitExitEdges(Loop *L, const SmallVector<BasicBlock *, 8> &ExitBlocks);

    bool UnswitchIfProfitable(Value *LoopCond, Constant *Val);
    unsigned getLoopUnswitchCost(Value *LIC);
    void UnswitchTrivialCondition(Loop *L, Value *Cond, Constant *Val,
                                  BasicBlock *ExitBlock);
    void UnswitchNontrivialCondition(Value *LIC, Constant *OnVal, Loop *L);

    void RewriteLoopBodyWithConditionConstant(Loop *L, Value *LIC,
                                              Constant *Val, bool isEqual);

    void EmitPreheaderBranchOnCondition(Value *LIC, Constant *Val,
                                        BasicBlock *TrueDest, 
                                        BasicBlock *FalseDest,
                                        Instruction *InsertPt);

    void SimplifyCode(std::vector<Instruction*> &Worklist, Loop *L);
    void RemoveBlockIfDead(BasicBlock *BB,
                           std::vector<Instruction*> &Worklist, Loop *l);
    void RemoveLoopFromHierarchy(Loop *L);
    bool IsTrivialUnswitchCondition(Value *Cond, Constant **Val = 0,
                                    BasicBlock **LoopExit = 0);

  };
}
char LoopUnswitch::ID = 0;
static RegisterPass<LoopUnswitch> X("loop-unswitch", "Unswitch loops");

Pass *llvm::createLoopUnswitchPass(bool Os) { 
  return new LoopUnswitch(Os); 
}

/// FindLIVLoopCondition - Cond is a condition that occurs in L.  If it is
/// invariant in the loop, or has an invariant piece, return the invariant.
/// Otherwise, return null.
static Value *FindLIVLoopCondition(Value *Cond, Loop *L, bool &Changed) {
  // Constants should be folded, not unswitched on!
  if (isa<Constant>(Cond)) return 0;

  // TODO: Handle: br (VARIANT|INVARIANT).
  // TODO: Hoist simple expressions out of loops.
  if (L->isLoopInvariant(Cond)) return Cond;

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Cond))
    if (BO->getOpcode() == Instruction::And ||
        BO->getOpcode() == Instruction::Or) {
      // If either the left or right side is invariant, we can unswitch on this,
      // which will cause the branch to go away in one loop and the condition to
      // simplify in the other one.
      if (Value *LHS = FindLIVLoopCondition(BO->getOperand(0), L, Changed))
        return LHS;
      if (Value *RHS = FindLIVLoopCondition(BO->getOperand(1), L, Changed))
        return RHS;
    }
  
  return 0;
}

bool LoopUnswitch::runOnLoop(Loop *L, LPPassManager &LPM_Ref) {
  LI = &getAnalysis<LoopInfo>();
  LPM = &LPM_Ref;
  DF = getAnalysisIfAvailable<DominanceFrontier>();
  DT = getAnalysisIfAvailable<DominatorTree>();
  currentLoop = L;
  Function *F = currentLoop->getHeader()->getParent();
  bool Changed = false;
  do {
    assert(currentLoop->isLCSSAForm());
    redoLoop = false;
    Changed |= processCurrentLoop();
  } while(redoLoop);

  if (Changed) {
    // FIXME: Reconstruct dom info, because it is not preserved properly.
    if (DT)
      DT->runOnFunction(*F);
    if (DF)
      DF->runOnFunction(*F);
  }
  return Changed;
}

/// processCurrentLoop - Do actual work and unswitch loop if possible 
/// and profitable.
bool LoopUnswitch::processCurrentLoop() {
  bool Changed = false;

  // Loop over all of the basic blocks in the loop.  If we find an interior
  // block that is branching on a loop-invariant condition, we can unswitch this
  // loop.
  for (Loop::block_iterator I = currentLoop->block_begin(), 
         E = currentLoop->block_end();
       I != E; ++I) {
    TerminatorInst *TI = (*I)->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      // If this isn't branching on an invariant condition, we can't unswitch
      // it.
      if (BI->isConditional()) {
        // See if this, or some part of it, is loop invariant.  If so, we can
        // unswitch on it if we desire.
        Value *LoopCond = FindLIVLoopCondition(BI->getCondition(), 
                                               currentLoop, Changed);
        if (LoopCond && UnswitchIfProfitable(LoopCond, 
                                             ConstantInt::getTrue())) {
          ++NumBranches;
          return true;
        }
      }      
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      Value *LoopCond = FindLIVLoopCondition(SI->getCondition(), 
                                             currentLoop, Changed);
      if (LoopCond && SI->getNumCases() > 1) {
        // Find a value to unswitch on:
        // FIXME: this should chose the most expensive case!
        Constant *UnswitchVal = SI->getCaseValue(1);
        // Do not process same value again and again.
        if (!UnswitchedVals.insert(UnswitchVal))
          continue;

        if (UnswitchIfProfitable(LoopCond, UnswitchVal)) {
          ++NumSwitches;
          return true;
        }
      }
    }
    
    // Scan the instructions to check for unswitchable values.
    for (BasicBlock::iterator BBI = (*I)->begin(), E = (*I)->end(); 
         BBI != E; ++BBI)
      if (SelectInst *SI = dyn_cast<SelectInst>(BBI)) {
        Value *LoopCond = FindLIVLoopCondition(SI->getCondition(), 
                                               currentLoop, Changed);
        if (LoopCond && UnswitchIfProfitable(LoopCond, 
                                             ConstantInt::getTrue())) {
          ++NumSelects;
          return true;
        }
      }
  }
  return Changed;
}

/// isTrivialLoopExitBlock - Check to see if all paths from BB either:
///   1. Exit the loop with no side effects.
///   2. Branch to the latch block with no side-effects.
///
/// If these conditions are true, we return true and set ExitBB to the block we
/// exit through.
///
static bool isTrivialLoopExitBlockHelper(Loop *L, BasicBlock *BB,
                                         BasicBlock *&ExitBB,
                                         std::set<BasicBlock*> &Visited) {
  if (!Visited.insert(BB).second) {
    // Already visited and Ok, end of recursion.
    return true;
  } else if (!L->contains(BB)) {
    // Otherwise, this is a loop exit, this is fine so long as this is the
    // first exit.
    if (ExitBB != 0) return false;
    ExitBB = BB;
    return true;
  }
  
  // Otherwise, this is an unvisited intra-loop node.  Check all successors.
  for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI) {
    // Check to see if the successor is a trivial loop exit.
    if (!isTrivialLoopExitBlockHelper(L, *SI, ExitBB, Visited))
      return false;
  }

  // Okay, everything after this looks good, check to make sure that this block
  // doesn't include any side effects.
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    if (I->mayWriteToMemory())
      return false;
  
  return true;
}

/// isTrivialLoopExitBlock - Return true if the specified block unconditionally
/// leads to an exit from the specified loop, and has no side-effects in the 
/// process.  If so, return the block that is exited to, otherwise return null.
static BasicBlock *isTrivialLoopExitBlock(Loop *L, BasicBlock *BB) {
  std::set<BasicBlock*> Visited;
  Visited.insert(L->getHeader());  // Branches to header are ok.
  BasicBlock *ExitBB = 0;
  if (isTrivialLoopExitBlockHelper(L, BB, ExitBB, Visited))
    return ExitBB;
  return 0;
}

/// IsTrivialUnswitchCondition - Check to see if this unswitch condition is
/// trivial: that is, that the condition controls whether or not the loop does
/// anything at all.  If this is a trivial condition, unswitching produces no
/// code duplications (equivalently, it produces a simpler loop and a new empty
/// loop, which gets deleted).
///
/// If this is a trivial condition, return true, otherwise return false.  When
/// returning true, this sets Cond and Val to the condition that controls the
/// trivial condition: when Cond dynamically equals Val, the loop is known to
/// exit.  Finally, this sets LoopExit to the BB that the loop exits to when
/// Cond == Val.
///
bool LoopUnswitch::IsTrivialUnswitchCondition(Value *Cond, Constant **Val,
                                       BasicBlock **LoopExit) {
  BasicBlock *Header = currentLoop->getHeader();
  TerminatorInst *HeaderTerm = Header->getTerminator();
  
  BasicBlock *LoopExitBB = 0;
  if (BranchInst *BI = dyn_cast<BranchInst>(HeaderTerm)) {
    // If the header block doesn't end with a conditional branch on Cond, we
    // can't handle it.
    if (!BI->isConditional() || BI->getCondition() != Cond)
      return false;
  
    // Check to see if a successor of the branch is guaranteed to go to the
    // latch block or exit through a one exit block without having any 
    // side-effects.  If so, determine the value of Cond that causes it to do
    // this.
    if ((LoopExitBB = isTrivialLoopExitBlock(currentLoop, 
                                             BI->getSuccessor(0)))) {
      if (Val) *Val = ConstantInt::getTrue();
    } else if ((LoopExitBB = isTrivialLoopExitBlock(currentLoop, 
                                                    BI->getSuccessor(1)))) {
      if (Val) *Val = ConstantInt::getFalse();
    }
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(HeaderTerm)) {
    // If this isn't a switch on Cond, we can't handle it.
    if (SI->getCondition() != Cond) return false;
    
    // Check to see if a successor of the switch is guaranteed to go to the
    // latch block or exit through a one exit block without having any 
    // side-effects.  If so, determine the value of Cond that causes it to do
    // this.  Note that we can't trivially unswitch on the default case.
    for (unsigned i = 1, e = SI->getNumSuccessors(); i != e; ++i)
      if ((LoopExitBB = isTrivialLoopExitBlock(currentLoop, 
                                               SI->getSuccessor(i)))) {
        // Okay, we found a trivial case, remember the value that is trivial.
        if (Val) *Val = SI->getCaseValue(i);
        break;
      }
  }

  // If we didn't find a single unique LoopExit block, or if the loop exit block
  // contains phi nodes, this isn't trivial.
  if (!LoopExitBB || isa<PHINode>(LoopExitBB->begin()))
    return false;   // Can't handle this.
  
  if (LoopExit) *LoopExit = LoopExitBB;
  
  // We already know that nothing uses any scalar values defined inside of this
  // loop.  As such, we just have to check to see if this loop will execute any
  // side-effecting instructions (e.g. stores, calls, volatile loads) in the
  // part of the loop that the code *would* execute.  We already checked the
  // tail, check the header now.
  for (BasicBlock::iterator I = Header->begin(), E = Header->end(); I != E; ++I)
    if (I->mayWriteToMemory())
      return false;
  return true;
}

/// getLoopUnswitchCost - Return the cost (code size growth) that will happen if
/// we choose to unswitch current loop on the specified value.
///
unsigned LoopUnswitch::getLoopUnswitchCost(Value *LIC) {
  // If the condition is trivial, always unswitch.  There is no code growth for
  // this case.
  if (IsTrivialUnswitchCondition(LIC))
    return 0;
  
  // FIXME: This is really overly conservative.  However, more liberal 
  // estimations have thus far resulted in excessive unswitching, which is bad
  // both in compile time and in code size.  This should be replaced once
  // someone figures out how a good estimation.
  return currentLoop->getBlocks().size();
  
  unsigned Cost = 0;
  // FIXME: this is brain dead.  It should take into consideration code
  // shrinkage.
  for (Loop::block_iterator I = currentLoop->block_begin(), 
         E = currentLoop->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    // Do not include empty blocks in the cost calculation.  This happen due to
    // loop canonicalization and will be removed.
    if (BB->begin() == BasicBlock::iterator(BB->getTerminator()))
      continue;
    
    // Count basic blocks.
    ++Cost;
  }

  return Cost;
}

/// UnswitchIfProfitable - We have found that we can unswitch currentLoop when
/// LoopCond == Val to simplify the loop.  If we decide that this is profitable,
/// unswitch the loop, reprocess the pieces, then return true.
bool LoopUnswitch::UnswitchIfProfitable(Value *LoopCond, Constant *Val){

  initLoopData();
  Function *F = loopHeader->getParent();

  // Do not unswitch if the function is optimized for size.
  if (!F->isDeclaration() && F->hasFnAttr(Attribute::OptimizeForSize))
    return false;

  // Check to see if it would be profitable to unswitch current loop.
  unsigned Cost = getLoopUnswitchCost(LoopCond);

  // Do not do non-trivial unswitch while optimizing for size.
  if (Cost && OptimizeForSize)
    return false;

  if (Cost > Threshold) {
    // FIXME: this should estimate growth by the amount of code shared by the
    // resultant unswitched loops.
    //
    DOUT << "NOT unswitching loop %"
         << currentLoop->getHeader()->getName() << ", cost too high: "
         << currentLoop->getBlocks().size() << "\n";
    return false;
  }

  Constant *CondVal;
  BasicBlock *ExitBlock;
  if (IsTrivialUnswitchCondition(LoopCond, &CondVal, &ExitBlock)) {
    UnswitchTrivialCondition(currentLoop, LoopCond, CondVal, ExitBlock);
  } else {
    UnswitchNontrivialCondition(LoopCond, Val, currentLoop);
  }

  return true;
}

// RemapInstruction - Convert the instruction operands from referencing the
// current values into those specified by ValueMap.
//
static inline void RemapInstruction(Instruction *I,
                                    DenseMap<const Value *, Value*> &ValueMap) {
  for (unsigned op = 0, E = I->getNumOperands(); op != E; ++op) {
    Value *Op = I->getOperand(op);
    DenseMap<const Value *, Value*>::iterator It = ValueMap.find(Op);
    if (It != ValueMap.end()) Op = It->second;
    I->setOperand(op, Op);
  }
}

/// CloneLoop - Recursively clone the specified loop and all of its children,
/// mapping the blocks with the specified map.
static Loop *CloneLoop(Loop *L, Loop *PL, DenseMap<const Value*, Value*> &VM,
                       LoopInfo *LI, LPPassManager *LPM) {
  Loop *New = new Loop();

  LPM->insertLoop(New, PL);

  // Add all of the blocks in L to the new loop.
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I)
    if (LI->getLoopFor(*I) == L)
      New->addBasicBlockToLoop(cast<BasicBlock>(VM[*I]), LI->getBase());

  // Add all of the subloops to the new loop.
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    CloneLoop(*I, New, VM, LI, LPM);

  return New;
}

/// EmitPreheaderBranchOnCondition - Emit a conditional branch on two values
/// if LIC == Val, branch to TrueDst, otherwise branch to FalseDest.  Insert the
/// code immediately before InsertPt.
void LoopUnswitch::EmitPreheaderBranchOnCondition(Value *LIC, Constant *Val,
                                                  BasicBlock *TrueDest,
                                                  BasicBlock *FalseDest,
                                                  Instruction *InsertPt) {
  // Insert a conditional branch on LIC to the two preheaders.  The original
  // code is the true version and the new code is the false version.
  Value *BranchVal = LIC;
  if (!isa<ConstantInt>(Val) || Val->getType() != Type::Int1Ty)
    BranchVal = new ICmpInst(ICmpInst::ICMP_EQ, LIC, Val, "tmp", InsertPt);
  else if (Val != ConstantInt::getTrue())
    // We want to enter the new loop when the condition is true.
    std::swap(TrueDest, FalseDest);

  // Insert the new branch.
  BranchInst::Create(TrueDest, FalseDest, BranchVal, InsertPt);
}

/// UnswitchTrivialCondition - Given a loop that has a trivial unswitchable
/// condition in it (a cond branch from its header block to its latch block,
/// where the path through the loop that doesn't execute its body has no 
/// side-effects), unswitch it.  This doesn't involve any code duplication, just
/// moving the conditional branch outside of the loop and updating loop info.
void LoopUnswitch::UnswitchTrivialCondition(Loop *L, Value *Cond, 
                                            Constant *Val, 
                                            BasicBlock *ExitBlock) {
  DOUT << "loop-unswitch: Trivial-Unswitch loop %"
       << loopHeader->getName() << " [" << L->getBlocks().size()
       << " blocks] in Function " << L->getHeader()->getParent()->getName()
       << " on cond: " << *Val << " == " << *Cond << "\n";
  
  // First step, split the preheader, so that we know that there is a safe place
  // to insert the conditional branch.  We will change loopPreheader to have a
  // conditional branch on Cond.
  BasicBlock *NewPH = SplitEdge(loopPreheader, loopHeader, this);

  // Now that we have a place to insert the conditional branch, create a place
  // to branch to: this is the exit block out of the loop that we should
  // short-circuit to.
  
  // Split this block now, so that the loop maintains its exit block, and so
  // that the jump from the preheader can execute the contents of the exit block
  // without actually branching to it (the exit block should be dominated by the
  // loop header, not the preheader).
  assert(!L->contains(ExitBlock) && "Exit block is in the loop?");
  BasicBlock *NewExit = SplitBlock(ExitBlock, ExitBlock->begin(), this);
    
  // Okay, now we have a position to branch from and a position to branch to, 
  // insert the new conditional branch.
  EmitPreheaderBranchOnCondition(Cond, Val, NewExit, NewPH, 
                                 loopPreheader->getTerminator());
  LPM->deleteSimpleAnalysisValue(loopPreheader->getTerminator(), L);
  loopPreheader->getTerminator()->eraseFromParent();

  // We need to reprocess this loop, it could be unswitched again.
  redoLoop = true;
  
  // Now that we know that the loop is never entered when this condition is a
  // particular value, rewrite the loop with this info.  We know that this will
  // at least eliminate the old branch.
  RewriteLoopBodyWithConditionConstant(L, Cond, Val, false);
  ++NumTrivial;
}

/// SplitExitEdges - Split all of the edges from inside the loop to their exit
/// blocks.  Update the appropriate Phi nodes as we do so.
void LoopUnswitch::SplitExitEdges(Loop *L, 
                                const SmallVector<BasicBlock *, 8> &ExitBlocks) 
{

  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    BasicBlock *ExitBlock = ExitBlocks[i];
    std::vector<BasicBlock*> Preds(pred_begin(ExitBlock), pred_end(ExitBlock));

    for (unsigned j = 0, e = Preds.size(); j != e; ++j) {
      BasicBlock* NewExitBlock = SplitEdge(Preds[j], ExitBlock, this);
      BasicBlock* StartBlock = Preds[j];
      BasicBlock* EndBlock;
      if (NewExitBlock->getSinglePredecessor() == ExitBlock) {
        EndBlock = NewExitBlock;
        NewExitBlock = EndBlock->getSinglePredecessor();;
      } else {
        EndBlock = ExitBlock;
      }
      
      std::set<PHINode*> InsertedPHIs;
      PHINode* OldLCSSA = 0;
      for (BasicBlock::iterator I = EndBlock->begin();
           (OldLCSSA = dyn_cast<PHINode>(I)); ++I) {
        Value* OldValue = OldLCSSA->getIncomingValueForBlock(NewExitBlock);
        PHINode* NewLCSSA = PHINode::Create(OldLCSSA->getType(),
                                            OldLCSSA->getName() + ".us-lcssa",
                                            NewExitBlock->getTerminator());
        NewLCSSA->addIncoming(OldValue, StartBlock);
        OldLCSSA->setIncomingValue(OldLCSSA->getBasicBlockIndex(NewExitBlock),
                                   NewLCSSA);
        InsertedPHIs.insert(NewLCSSA);
      }

      BasicBlock::iterator InsertPt = EndBlock->getFirstNonPHI();
      for (BasicBlock::iterator I = NewExitBlock->begin();
         (OldLCSSA = dyn_cast<PHINode>(I)) && InsertedPHIs.count(OldLCSSA) == 0;
         ++I) {
        PHINode *NewLCSSA = PHINode::Create(OldLCSSA->getType(),
                                            OldLCSSA->getName() + ".us-lcssa",
                                            InsertPt);
        OldLCSSA->replaceAllUsesWith(NewLCSSA);
        NewLCSSA->addIncoming(OldLCSSA, NewExitBlock);
      }

    }    
  }

}

/// UnswitchNontrivialCondition - We determined that the loop is profitable 
/// to unswitch when LIC equal Val.  Split it into loop versions and test the 
/// condition outside of either loop.  Return the loops created as Out1/Out2.
void LoopUnswitch::UnswitchNontrivialCondition(Value *LIC, Constant *Val, 
                                               Loop *L) {
  Function *F = loopHeader->getParent();
  DOUT << "loop-unswitch: Unswitching loop %"
       << loopHeader->getName() << " [" << L->getBlocks().size()
       << " blocks] in Function " << F->getName()
       << " when '" << *Val << "' == " << *LIC << "\n";

  LoopBlocks.clear();
  NewBlocks.clear();

  // First step, split the preheader and exit blocks, and add these blocks to
  // the LoopBlocks list.
  BasicBlock *NewPreheader = SplitEdge(loopPreheader, loopHeader, this);
  LoopBlocks.push_back(NewPreheader);

  // We want the loop to come after the preheader, but before the exit blocks.
  LoopBlocks.insert(LoopBlocks.end(), L->block_begin(), L->block_end());

  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);

  // Split all of the edges from inside the loop to their exit blocks.  Update
  // the appropriate Phi nodes as we do so.
  SplitExitEdges(L, ExitBlocks);

  // The exit blocks may have been changed due to edge splitting, recompute.
  ExitBlocks.clear();
  L->getUniqueExitBlocks(ExitBlocks);

  // Add exit blocks to the loop blocks.
  LoopBlocks.insert(LoopBlocks.end(), ExitBlocks.begin(), ExitBlocks.end());

  // Next step, clone all of the basic blocks that make up the loop (including
  // the loop preheader and exit blocks), keeping track of the mapping between
  // the instructions and blocks.
  NewBlocks.reserve(LoopBlocks.size());
  DenseMap<const Value*, Value*> ValueMap;
  for (unsigned i = 0, e = LoopBlocks.size(); i != e; ++i) {
    BasicBlock *New = CloneBasicBlock(LoopBlocks[i], ValueMap, ".us", F);
    NewBlocks.push_back(New);
    ValueMap[LoopBlocks[i]] = New;  // Keep the BB mapping.
    LPM->cloneBasicBlockSimpleAnalysis(LoopBlocks[i], New, L);
  }

  // Splice the newly inserted blocks into the function right before the
  // original preheader.
  F->getBasicBlockList().splice(LoopBlocks[0], F->getBasicBlockList(),
                                NewBlocks[0], F->end());

  // Now we create the new Loop object for the versioned loop.
  Loop *NewLoop = CloneLoop(L, L->getParentLoop(), ValueMap, LI, LPM);
  Loop *ParentLoop = L->getParentLoop();
  if (ParentLoop) {
    // Make sure to add the cloned preheader and exit blocks to the parent loop
    // as well.
    ParentLoop->addBasicBlockToLoop(NewBlocks[0], LI->getBase());
  }
  
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    BasicBlock *NewExit = cast<BasicBlock>(ValueMap[ExitBlocks[i]]);
    // The new exit block should be in the same loop as the old one.
    if (Loop *ExitBBLoop = LI->getLoopFor(ExitBlocks[i]))
      ExitBBLoop->addBasicBlockToLoop(NewExit, LI->getBase());
    
    assert(NewExit->getTerminator()->getNumSuccessors() == 1 &&
           "Exit block should have been split to have one successor!");
    BasicBlock *ExitSucc = NewExit->getTerminator()->getSuccessor(0);

    // If the successor of the exit block had PHI nodes, add an entry for
    // NewExit.
    PHINode *PN;
    for (BasicBlock::iterator I = ExitSucc->begin();
         (PN = dyn_cast<PHINode>(I)); ++I) {
      Value *V = PN->getIncomingValueForBlock(ExitBlocks[i]);
      DenseMap<const Value *, Value*>::iterator It = ValueMap.find(V);
      if (It != ValueMap.end()) V = It->second;
      PN->addIncoming(V, NewExit);
    }
  }

  // Rewrite the code to refer to itself.
  for (unsigned i = 0, e = NewBlocks.size(); i != e; ++i)
    for (BasicBlock::iterator I = NewBlocks[i]->begin(),
           E = NewBlocks[i]->end(); I != E; ++I)
      RemapInstruction(I, ValueMap);
  
  // Rewrite the original preheader to select between versions of the loop.
  BranchInst *OldBR = cast<BranchInst>(loopPreheader->getTerminator());
  assert(OldBR->isUnconditional() && OldBR->getSuccessor(0) == LoopBlocks[0] &&
         "Preheader splitting did not work correctly!");

  // Emit the new branch that selects between the two versions of this loop.
  EmitPreheaderBranchOnCondition(LIC, Val, NewBlocks[0], LoopBlocks[0], OldBR);
  LPM->deleteSimpleAnalysisValue(OldBR, L);
  OldBR->eraseFromParent();

  LoopProcessWorklist.push_back(NewLoop);
  redoLoop = true;

  // Now we rewrite the original code to know that the condition is true and the
  // new code to know that the condition is false.
  RewriteLoopBodyWithConditionConstant(L      , LIC, Val, false);
  
  // It's possible that simplifying one loop could cause the other to be
  // deleted.  If so, don't simplify it.
  if (!LoopProcessWorklist.empty() && LoopProcessWorklist.back() == NewLoop)
    RewriteLoopBodyWithConditionConstant(NewLoop, LIC, Val, true);

}

/// RemoveFromWorklist - Remove all instances of I from the worklist vector
/// specified.
static void RemoveFromWorklist(Instruction *I, 
                               std::vector<Instruction*> &Worklist) {
  std::vector<Instruction*>::iterator WI = std::find(Worklist.begin(),
                                                     Worklist.end(), I);
  while (WI != Worklist.end()) {
    unsigned Offset = WI-Worklist.begin();
    Worklist.erase(WI);
    WI = std::find(Worklist.begin()+Offset, Worklist.end(), I);
  }
}

/// ReplaceUsesOfWith - When we find that I really equals V, remove I from the
/// program, replacing all uses with V and update the worklist.
static void ReplaceUsesOfWith(Instruction *I, Value *V, 
                              std::vector<Instruction*> &Worklist,
                              Loop *L, LPPassManager *LPM) {
  DOUT << "Replace with '" << *V << "': " << *I;

  // Add uses to the worklist, which may be dead now.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (Instruction *Use = dyn_cast<Instruction>(I->getOperand(i)))
      Worklist.push_back(Use);

  // Add users to the worklist which may be simplified now.
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI)
    Worklist.push_back(cast<Instruction>(*UI));
  LPM->deleteSimpleAnalysisValue(I, L);
  RemoveFromWorklist(I, Worklist);
  I->replaceAllUsesWith(V);
  I->eraseFromParent();
  ++NumSimplify;
}

/// RemoveBlockIfDead - If the specified block is dead, remove it, update loop
/// information, and remove any dead successors it has.
///
void LoopUnswitch::RemoveBlockIfDead(BasicBlock *BB,
                                     std::vector<Instruction*> &Worklist,
                                     Loop *L) {
  if (pred_begin(BB) != pred_end(BB)) {
    // This block isn't dead, since an edge to BB was just removed, see if there
    // are any easy simplifications we can do now.
    if (BasicBlock *Pred = BB->getSinglePredecessor()) {
      // If it has one pred, fold phi nodes in BB.
      while (isa<PHINode>(BB->begin()))
        ReplaceUsesOfWith(BB->begin(), 
                          cast<PHINode>(BB->begin())->getIncomingValue(0), 
                          Worklist, L, LPM);
      
      // If this is the header of a loop and the only pred is the latch, we now
      // have an unreachable loop.
      if (Loop *L = LI->getLoopFor(BB))
        if (loopHeader == BB && L->contains(Pred)) {
          // Remove the branch from the latch to the header block, this makes
          // the header dead, which will make the latch dead (because the header
          // dominates the latch).
          LPM->deleteSimpleAnalysisValue(Pred->getTerminator(), L);
          Pred->getTerminator()->eraseFromParent();
          new UnreachableInst(Pred);
          
          // The loop is now broken, remove it from LI.
          RemoveLoopFromHierarchy(L);
          
          // Reprocess the header, which now IS dead.
          RemoveBlockIfDead(BB, Worklist, L);
          return;
        }
      
      // If pred ends in a uncond branch, add uncond branch to worklist so that
      // the two blocks will get merged.
      if (BranchInst *BI = dyn_cast<BranchInst>(Pred->getTerminator()))
        if (BI->isUnconditional())
          Worklist.push_back(BI);
    }
    return;
  }

  DOUT << "Nuking dead block: " << *BB;
  
  // Remove the instructions in the basic block from the worklist.
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
    RemoveFromWorklist(I, Worklist);
    
    // Anything that uses the instructions in this basic block should have their
    // uses replaced with undefs.
    if (!I->use_empty())
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
  }
  
  // If this is the edge to the header block for a loop, remove the loop and
  // promote all subloops.
  if (Loop *BBLoop = LI->getLoopFor(BB)) {
    if (BBLoop->getLoopLatch() == BB)
      RemoveLoopFromHierarchy(BBLoop);
  }

  // Remove the block from the loop info, which removes it from any loops it
  // was in.
  LI->removeBlock(BB);
  
  
  // Remove phi node entries in successors for this block.
  TerminatorInst *TI = BB->getTerminator();
  SmallVector<BasicBlock*, 4> Succs;
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
    Succs.push_back(TI->getSuccessor(i));
    TI->getSuccessor(i)->removePredecessor(BB);
  }
  
  // Unique the successors, remove anything with multiple uses.
  array_pod_sort(Succs.begin(), Succs.end());
  Succs.erase(std::unique(Succs.begin(), Succs.end()), Succs.end());
  
  // Remove the basic block, including all of the instructions contained in it.
  LPM->deleteSimpleAnalysisValue(BB, L);  
  BB->eraseFromParent();
  // Remove successor blocks here that are not dead, so that we know we only
  // have dead blocks in this list.  Nondead blocks have a way of becoming dead,
  // then getting removed before we revisit them, which is badness.
  //
  for (unsigned i = 0; i != Succs.size(); ++i)
    if (pred_begin(Succs[i]) != pred_end(Succs[i])) {
      // One exception is loop headers.  If this block was the preheader for a
      // loop, then we DO want to visit the loop so the loop gets deleted.
      // We know that if the successor is a loop header, that this loop had to
      // be the preheader: the case where this was the latch block was handled
      // above and headers can only have two predecessors.
      if (!LI->isLoopHeader(Succs[i])) {
        Succs.erase(Succs.begin()+i);
        --i;
      }
    }
  
  for (unsigned i = 0, e = Succs.size(); i != e; ++i)
    RemoveBlockIfDead(Succs[i], Worklist, L);
}

/// RemoveLoopFromHierarchy - We have discovered that the specified loop has
/// become unwrapped, either because the backedge was deleted, or because the
/// edge into the header was removed.  If the edge into the header from the
/// latch block was removed, the loop is unwrapped but subloops are still alive,
/// so they just reparent loops.  If the loops are actually dead, they will be
/// removed later.
void LoopUnswitch::RemoveLoopFromHierarchy(Loop *L) {
  LPM->deleteLoopFromQueue(L);
  RemoveLoopFromWorklist(L);
}

// RewriteLoopBodyWithConditionConstant - We know either that the value LIC has
// the value specified by Val in the specified loop, or we know it does NOT have
// that value.  Rewrite any uses of LIC or of properties correlated to it.
void LoopUnswitch::RewriteLoopBodyWithConditionConstant(Loop *L, Value *LIC,
                                                        Constant *Val,
                                                        bool IsEqual) {
  assert(!isa<Constant>(LIC) && "Why are we unswitching on a constant?");
  
  // FIXME: Support correlated properties, like:
  //  for (...)
  //    if (li1 < li2)
  //      ...
  //    if (li1 > li2)
  //      ...
  
  // FOLD boolean conditions (X|LIC), (X&LIC).  Fold conditional branches,
  // selects, switches.
  std::vector<User*> Users(LIC->use_begin(), LIC->use_end());
  std::vector<Instruction*> Worklist;

  // If we know that LIC == Val, or that LIC == NotVal, just replace uses of LIC
  // in the loop with the appropriate one directly.
  if (IsEqual || (isa<ConstantInt>(Val) && Val->getType() == Type::Int1Ty)) {
    Value *Replacement;
    if (IsEqual)
      Replacement = Val;
    else
      Replacement = ConstantInt::get(Type::Int1Ty, 
                                     !cast<ConstantInt>(Val)->getZExtValue());
    
    for (unsigned i = 0, e = Users.size(); i != e; ++i)
      if (Instruction *U = cast<Instruction>(Users[i])) {
        if (!L->contains(U->getParent()))
          continue;
        U->replaceUsesOfWith(LIC, Replacement);
        Worklist.push_back(U);
      }
  } else {
    // Otherwise, we don't know the precise value of LIC, but we do know that it
    // is certainly NOT "Val".  As such, simplify any uses in the loop that we
    // can.  This case occurs when we unswitch switch statements.
    for (unsigned i = 0, e = Users.size(); i != e; ++i)
      if (Instruction *U = cast<Instruction>(Users[i])) {
        if (!L->contains(U->getParent()))
          continue;

        Worklist.push_back(U);

        // If we know that LIC is not Val, use this info to simplify code.
        if (SwitchInst *SI = dyn_cast<SwitchInst>(U)) {
          for (unsigned i = 1, e = SI->getNumCases(); i != e; ++i) {
            if (SI->getCaseValue(i) == Val) {
              // Found a dead case value.  Don't remove PHI nodes in the 
              // successor if they become single-entry, those PHI nodes may
              // be in the Users list.
              
              // FIXME: This is a hack.  We need to keep the successor around
              // and hooked up so as to preserve the loop structure, because
              // trying to update it is complicated.  So instead we preserve the
              // loop structure and put the block on an dead code path.
              
              BasicBlock *SISucc = SI->getSuccessor(i);
              BasicBlock* Old = SI->getParent();
              BasicBlock* Split = SplitBlock(Old, SI, this);
              
              Instruction* OldTerm = Old->getTerminator();
              BranchInst::Create(Split, SISucc,
                                 ConstantInt::getTrue(), OldTerm);

              LPM->deleteSimpleAnalysisValue(Old->getTerminator(), L);
              Old->getTerminator()->eraseFromParent();
              
              PHINode *PN;
              for (BasicBlock::iterator II = SISucc->begin();
                   (PN = dyn_cast<PHINode>(II)); ++II) {
                Value *InVal = PN->removeIncomingValue(Split, false);
                PN->addIncoming(InVal, Old);
              }

              SI->removeCase(i);
              break;
            }
          }
        }
        
        // TODO: We could do other simplifications, for example, turning 
        // LIC == Val -> false.
      }
  }
  
  SimplifyCode(Worklist, L);
}

/// SimplifyCode - Okay, now that we have simplified some instructions in the 
/// loop, walk over it and constant prop, dce, and fold control flow where
/// possible.  Note that this is effectively a very simple loop-structure-aware
/// optimizer.  During processing of this loop, L could very well be deleted, so
/// it must not be used.
///
/// FIXME: When the loop optimizer is more mature, separate this out to a new
/// pass.
///
void LoopUnswitch::SimplifyCode(std::vector<Instruction*> &Worklist, Loop *L) {
  while (!Worklist.empty()) {
    Instruction *I = Worklist.back();
    Worklist.pop_back();
    
    // Simple constant folding.
    if (Constant *C = ConstantFoldInstruction(I)) {
      ReplaceUsesOfWith(I, C, Worklist, L, LPM);
      continue;
    }
    
    // Simple DCE.
    if (isInstructionTriviallyDead(I)) {
      DOUT << "Remove dead instruction '" << *I;
      
      // Add uses to the worklist, which may be dead now.
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
        if (Instruction *Use = dyn_cast<Instruction>(I->getOperand(i)))
          Worklist.push_back(Use);
      LPM->deleteSimpleAnalysisValue(I, L);
      RemoveFromWorklist(I, Worklist);
      I->eraseFromParent();
      ++NumSimplify;
      continue;
    }
    
    // Special case hacks that appear commonly in unswitched code.
    switch (I->getOpcode()) {
    case Instruction::Select:
      if (ConstantInt *CB = dyn_cast<ConstantInt>(I->getOperand(0))) {
        ReplaceUsesOfWith(I, I->getOperand(!CB->getZExtValue()+1), Worklist, L,
                          LPM);
        continue;
      }
      break;
    case Instruction::And:
      if (isa<ConstantInt>(I->getOperand(0)) && 
          I->getOperand(0)->getType() == Type::Int1Ty)   // constant -> RHS
        cast<BinaryOperator>(I)->swapOperands();
      if (ConstantInt *CB = dyn_cast<ConstantInt>(I->getOperand(1))) 
        if (CB->getType() == Type::Int1Ty) {
          if (CB->isOne())      // X & 1 -> X
            ReplaceUsesOfWith(I, I->getOperand(0), Worklist, L, LPM);
          else                  // X & 0 -> 0
            ReplaceUsesOfWith(I, I->getOperand(1), Worklist, L, LPM);
          continue;
        }
      break;
    case Instruction::Or:
      if (isa<ConstantInt>(I->getOperand(0)) &&
          I->getOperand(0)->getType() == Type::Int1Ty)   // constant -> RHS
        cast<BinaryOperator>(I)->swapOperands();
      if (ConstantInt *CB = dyn_cast<ConstantInt>(I->getOperand(1)))
        if (CB->getType() == Type::Int1Ty) {
          if (CB->isOne())   // X | 1 -> 1
            ReplaceUsesOfWith(I, I->getOperand(1), Worklist, L, LPM);
          else                  // X | 0 -> X
            ReplaceUsesOfWith(I, I->getOperand(0), Worklist, L, LPM);
          continue;
        }
      break;
    case Instruction::Br: {
      BranchInst *BI = cast<BranchInst>(I);
      if (BI->isUnconditional()) {
        // If BI's parent is the only pred of the successor, fold the two blocks
        // together.
        BasicBlock *Pred = BI->getParent();
        BasicBlock *Succ = BI->getSuccessor(0);
        BasicBlock *SinglePred = Succ->getSinglePredecessor();
        if (!SinglePred) continue;  // Nothing to do.
        assert(SinglePred == Pred && "CFG broken");

        DOUT << "Merging blocks: " << Pred->getName() << " <- " 
             << Succ->getName() << "\n";
        
        // Resolve any single entry PHI nodes in Succ.
        while (PHINode *PN = dyn_cast<PHINode>(Succ->begin()))
          ReplaceUsesOfWith(PN, PN->getIncomingValue(0), Worklist, L, LPM);
        
        // Move all of the successor contents from Succ to Pred.
        Pred->getInstList().splice(BI, Succ->getInstList(), Succ->begin(),
                                   Succ->end());
        LPM->deleteSimpleAnalysisValue(BI, L);
        BI->eraseFromParent();
        RemoveFromWorklist(BI, Worklist);
        
        // If Succ has any successors with PHI nodes, update them to have
        // entries coming from Pred instead of Succ.
        Succ->replaceAllUsesWith(Pred);
        
        // Remove Succ from the loop tree.
        LI->removeBlock(Succ);
        LPM->deleteSimpleAnalysisValue(Succ, L);
        Succ->eraseFromParent();
        ++NumSimplify;
      } else if (ConstantInt *CB = dyn_cast<ConstantInt>(BI->getCondition())){
        // Conditional branch.  Turn it into an unconditional branch, then
        // remove dead blocks.
        break;  // FIXME: Enable.

        DOUT << "Folded branch: " << *BI;
        BasicBlock *DeadSucc = BI->getSuccessor(CB->getZExtValue());
        BasicBlock *LiveSucc = BI->getSuccessor(!CB->getZExtValue());
        DeadSucc->removePredecessor(BI->getParent(), true);
        Worklist.push_back(BranchInst::Create(LiveSucc, BI));
        LPM->deleteSimpleAnalysisValue(BI, L);
        BI->eraseFromParent();
        RemoveFromWorklist(BI, Worklist);
        ++NumSimplify;

        RemoveBlockIfDead(DeadSucc, Worklist, L);
      }
      break;
    }
    }
  }
}
