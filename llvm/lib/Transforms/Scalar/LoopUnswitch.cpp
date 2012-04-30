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
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <set>
using namespace llvm;

STATISTIC(NumBranches, "Number of branches unswitched");
STATISTIC(NumSwitches, "Number of switches unswitched");
STATISTIC(NumSelects , "Number of selects unswitched");
STATISTIC(NumTrivial , "Number of unswitches that are trivial");
STATISTIC(NumSimplify, "Number of simplifications of unswitched code");
STATISTIC(TotalInsts,  "Total number of instructions analyzed");

// The specific value of 100 here was chosen based only on intuition and a
// few specific examples.
static cl::opt<unsigned>
Threshold("loop-unswitch-threshold", cl::desc("Max loop size to unswitch"),
          cl::init(100), cl::Hidden);

namespace {

  class LUAnalysisCache {

    typedef DenseMap<const SwitchInst*, SmallPtrSet<const Value *, 8> >
      UnswitchedValsMap;

    typedef UnswitchedValsMap::iterator UnswitchedValsIt;

    struct LoopProperties {
      unsigned CanBeUnswitchedCount;
      unsigned SizeEstimation;
      UnswitchedValsMap UnswitchedVals;
    };

    // Here we use std::map instead of DenseMap, since we need to keep valid
    // LoopProperties pointer for current loop for better performance.
    typedef std::map<const Loop*, LoopProperties> LoopPropsMap;
    typedef LoopPropsMap::iterator LoopPropsMapIt;

    LoopPropsMap LoopsProperties;
    UnswitchedValsMap* CurLoopInstructions;
    LoopProperties* CurrentLoopProperties;

    // Max size of code we can produce on remained iterations.
    unsigned MaxSize;

    public:

      LUAnalysisCache() :
        CurLoopInstructions(NULL), CurrentLoopProperties(NULL),
        MaxSize(Threshold)
      {}

      // Analyze loop. Check its size, calculate is it possible to unswitch
      // it. Returns true if we can unswitch this loop.
      bool countLoop(const Loop* L);

      // Clean all data related to given loop.
      void forgetLoop(const Loop* L);

      // Mark case value as unswitched.
      // Since SI instruction can be partly unswitched, in order to avoid
      // extra unswitching in cloned loops keep track all unswitched values.
      void setUnswitched(const SwitchInst* SI, const Value* V);

      // Check was this case value unswitched before or not.
      bool isUnswitched(const SwitchInst* SI, const Value* V);

      // Clone all loop-unswitch related loop properties.
      // Redistribute unswitching quotas.
      // Note, that new loop data is stored inside the VMap.
      void cloneData(const Loop* NewLoop, const Loop* OldLoop,
                     const ValueToValueMapTy& VMap);
  };

  class LoopUnswitch : public LoopPass {
    LoopInfo *LI;  // Loop information
    LPPassManager *LPM;

    // LoopProcessWorklist - Used to check if second loop needs processing
    // after RewriteLoopBodyWithConditionConstant rewrites first loop.
    std::vector<Loop*> LoopProcessWorklist;

    LUAnalysisCache BranchesInfo;

    bool OptimizeForSize;
    bool redoLoop;

    Loop *currentLoop;
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
      LoopPass(ID), OptimizeForSize(Os), redoLoop(false),
      currentLoop(NULL), DT(NULL), loopHeader(NULL),
      loopPreheader(NULL) {
        initializeLoopUnswitchPass(*PassRegistry::getPassRegistry());
      }

    bool runOnLoop(Loop *L, LPPassManager &LPM);
    bool processCurrentLoop();

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<ScalarEvolution>();
    }

  private:

    virtual void releaseMemory() {
      BranchesInfo.forgetLoop(currentLoop);
    }

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

// Analyze loop. Check its size, calculate is it possible to unswitch
// it. Returns true if we can unswitch this loop.
bool LUAnalysisCache::countLoop(const Loop* L) {

  std::pair<LoopPropsMapIt, bool> InsertRes =
      LoopsProperties.insert(std::make_pair(L, LoopProperties()));

  LoopProperties& Props = InsertRes.first->second;

  if (InsertRes.second) {
    // New loop.

    // Limit the number of instructions to avoid causing significant code
    // expansion, and the number of basic blocks, to avoid loops with
    // large numbers of branches which cause loop unswitching to go crazy.
    // This is a very ad-hoc heuristic.

    // FIXME: This is overly conservative because it does not take into
    // consideration code simplification opportunities and code that can
    // be shared by the resultant unswitched loops.
    CodeMetrics Metrics;
    for (Loop::block_iterator I = L->block_begin(),
           E = L->block_end();
         I != E; ++I)
      Metrics.analyzeBasicBlock(*I);

    Props.SizeEstimation = std::min(Metrics.NumInsts, Metrics.NumBlocks * 5);
    Props.CanBeUnswitchedCount = MaxSize / (Props.SizeEstimation);
    MaxSize -= Props.SizeEstimation * Props.CanBeUnswitchedCount;
  }

  if (!Props.CanBeUnswitchedCount) {
    DEBUG(dbgs() << "NOT unswitching loop %"
          << L->getHeader()->getName() << ", cost too high: "
          << L->getBlocks().size() << "\n");

    return false;
  }

  // Be careful. This links are good only before new loop addition.
  CurrentLoopProperties = &Props;
  CurLoopInstructions = &Props.UnswitchedVals;

  return true;
}

// Clean all data related to given loop.
void LUAnalysisCache::forgetLoop(const Loop* L) {

  LoopPropsMapIt LIt = LoopsProperties.find(L);

  if (LIt != LoopsProperties.end()) {
    LoopProperties& Props = LIt->second;
    MaxSize += Props.CanBeUnswitchedCount * Props.SizeEstimation;
    LoopsProperties.erase(LIt);
  }

  CurrentLoopProperties = NULL;
  CurLoopInstructions = NULL;
}

// Mark case value as unswitched.
// Since SI instruction can be partly unswitched, in order to avoid
// extra unswitching in cloned loops keep track all unswitched values.
void LUAnalysisCache::setUnswitched(const SwitchInst* SI, const Value* V) {
  (*CurLoopInstructions)[SI].insert(V);
}

// Check was this case value unswitched before or not.
bool LUAnalysisCache::isUnswitched(const SwitchInst* SI, const Value* V) {
  return (*CurLoopInstructions)[SI].count(V);
}

// Clone all loop-unswitch related loop properties.
// Redistribute unswitching quotas.
// Note, that new loop data is stored inside the VMap.
void LUAnalysisCache::cloneData(const Loop* NewLoop, const Loop* OldLoop,
                     const ValueToValueMapTy& VMap) {

  LoopProperties& NewLoopProps = LoopsProperties[NewLoop];
  LoopProperties& OldLoopProps = *CurrentLoopProperties;
  UnswitchedValsMap& Insts = OldLoopProps.UnswitchedVals;

  // Reallocate "can-be-unswitched quota"

  --OldLoopProps.CanBeUnswitchedCount;
  unsigned Quota = OldLoopProps.CanBeUnswitchedCount;
  NewLoopProps.CanBeUnswitchedCount = Quota / 2;
  OldLoopProps.CanBeUnswitchedCount = Quota - Quota / 2;

  NewLoopProps.SizeEstimation = OldLoopProps.SizeEstimation;

  // Clone unswitched values info:
  // for new loop switches we clone info about values that was
  // already unswitched and has redundant successors.
  for (UnswitchedValsIt I = Insts.begin(); I != Insts.end(); ++I) {
    const SwitchInst* OldInst = I->first;
    Value* NewI = VMap.lookup(OldInst);
    const SwitchInst* NewInst = cast_or_null<SwitchInst>(NewI);
    assert(NewInst && "All instructions that are in SrcBB must be in VMap.");

    NewLoopProps.UnswitchedVals[NewInst] = OldLoopProps.UnswitchedVals[OldInst];
  }
}

char LoopUnswitch::ID = 0;
INITIALIZE_PASS_BEGIN(LoopUnswitch, "loop-unswitch", "Unswitch loops",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_END(LoopUnswitch, "loop-unswitch", "Unswitch loops",
                      false, false)

Pass *llvm::createLoopUnswitchPass(bool Os) {
  return new LoopUnswitch(Os);
}

/// FindLIVLoopCondition - Cond is a condition that occurs in L.  If it is
/// invariant in the loop, or has an invariant piece, return the invariant.
/// Otherwise, return null.
static Value *FindLIVLoopCondition(Value *Cond, Loop *L, bool &Changed) {

  // We started analyze new instruction, increment scanned instructions counter.
  ++TotalInsts;

  // We can never unswitch on vector conditions.
  if (Cond->getType()->isVectorTy())
    return 0;

  // Constants should be folded, not unswitched on!
  if (isa<Constant>(Cond)) return 0;

  // TODO: Handle: br (VARIANT|INVARIANT).

  // Hoist simple values out.
  if (L->makeLoopInvariant(Cond, Changed))
    return Cond;

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
  DT = getAnalysisIfAvailable<DominatorTree>();
  currentLoop = L;
  Function *F = currentLoop->getHeader()->getParent();
  bool Changed = false;
  do {
    assert(currentLoop->isLCSSAForm(*DT));
    redoLoop = false;
    Changed |= processCurrentLoop();
  } while(redoLoop);

  if (Changed) {
    // FIXME: Reconstruct dom info, because it is not preserved properly.
    if (DT)
      DT->runOnFunction(*F);
  }
  return Changed;
}

/// processCurrentLoop - Do actual work and unswitch loop if possible
/// and profitable.
bool LoopUnswitch::processCurrentLoop() {
  bool Changed = false;

  initLoopData();

  // If LoopSimplify was unable to form a preheader, don't do any unswitching.
  if (!loopPreheader)
    return false;

  // Loops with indirectbr cannot be cloned.
  if (!currentLoop->isSafeToClone())
    return false;

  // Without dedicated exits, splitting the exit edge may fail.
  if (!currentLoop->hasDedicatedExits())
    return false;

  LLVMContext &Context = loopHeader->getContext();

  // Probably we reach the quota of branches for this loop. If so
  // stop unswitching.
  if (!BranchesInfo.countLoop(currentLoop))
    return false;

  // Loop over all of the basic blocks in the loop.  If we find an interior
  // block that is branching on a loop-invariant condition, we can unswitch this
  // loop.
  for (Loop::block_iterator I = currentLoop->block_begin(),
         E = currentLoop->block_end(); I != E; ++I) {
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
                                             ConstantInt::getTrue(Context))) {
          ++NumBranches;
          return true;
        }
      }
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      Value *LoopCond = FindLIVLoopCondition(SI->getCondition(),
                                             currentLoop, Changed);
      unsigned NumCases = SI->getNumCases();
      if (LoopCond && NumCases) {
        // Find a value to unswitch on:
        // FIXME: this should chose the most expensive case!
        // FIXME: scan for a case with a non-critical edge?
        Constant *UnswitchVal = NULL;

        // Do not process same value again and again.
        // At this point we have some cases already unswitched and
        // some not yet unswitched. Let's find the first not yet unswitched one.
        for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
             i != e; ++i) {
          Constant* UnswitchValCandidate = i.getCaseValue();
          if (!BranchesInfo.isUnswitched(SI, UnswitchValCandidate)) {
            UnswitchVal = UnswitchValCandidate;
            break;
          }
        }

        if (!UnswitchVal)
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
                                             ConstantInt::getTrue(Context))) {
          ++NumSelects;
          return true;
        }
      }
  }
  return Changed;
}

/// isTrivialLoopExitBlock - Check to see if all paths from BB exit the
/// loop with no side effects (including infinite loops).
///
/// If true, we return true and set ExitBB to the block we
/// exit through.
///
static bool isTrivialLoopExitBlockHelper(Loop *L, BasicBlock *BB,
                                         BasicBlock *&ExitBB,
                                         std::set<BasicBlock*> &Visited) {
  if (!Visited.insert(BB).second) {
    // Already visited. Without more analysis, this could indicate an infinite
    // loop.
    return false;
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
    if (I->mayHaveSideEffects())
      return false;

  return true;
}

/// isTrivialLoopExitBlock - Return true if the specified block unconditionally
/// leads to an exit from the specified loop, and has no side-effects in the
/// process.  If so, return the block that is exited to, otherwise return null.
static BasicBlock *isTrivialLoopExitBlock(Loop *L, BasicBlock *BB) {
  std::set<BasicBlock*> Visited;
  Visited.insert(L->getHeader());  // Branches to header make infinite loops.
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
  LLVMContext &Context = Header->getContext();

  BasicBlock *LoopExitBB = 0;
  if (BranchInst *BI = dyn_cast<BranchInst>(HeaderTerm)) {
    // If the header block doesn't end with a conditional branch on Cond, we
    // can't handle it.
    if (!BI->isConditional() || BI->getCondition() != Cond)
      return false;

    // Check to see if a successor of the branch is guaranteed to
    // exit through a unique exit block without having any
    // side-effects.  If so, determine the value of Cond that causes it to do
    // this.
    if ((LoopExitBB = isTrivialLoopExitBlock(currentLoop,
                                             BI->getSuccessor(0)))) {
      if (Val) *Val = ConstantInt::getTrue(Context);
    } else if ((LoopExitBB = isTrivialLoopExitBlock(currentLoop,
                                                    BI->getSuccessor(1)))) {
      if (Val) *Val = ConstantInt::getFalse(Context);
    }
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(HeaderTerm)) {
    // If this isn't a switch on Cond, we can't handle it.
    if (SI->getCondition() != Cond) return false;

    // Check to see if a successor of the switch is guaranteed to go to the
    // latch block or exit through a one exit block without having any
    // side-effects.  If so, determine the value of Cond that causes it to do
    // this.
    // Note that we can't trivially unswitch on the default case or
    // on already unswitched cases.
    for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
         i != e; ++i) {
      BasicBlock* LoopExitCandidate;
      if ((LoopExitCandidate = isTrivialLoopExitBlock(currentLoop,
                                               i.getCaseSuccessor()))) {
        // Okay, we found a trivial case, remember the value that is trivial.
        ConstantInt* CaseVal = i.getCaseValue();

        // Check that it was not unswitched before, since already unswitched
        // trivial vals are looks trivial too.
        if (BranchesInfo.isUnswitched(SI, CaseVal))
          continue;
        LoopExitBB = LoopExitCandidate;
        if (Val) *Val = CaseVal;
        break;
      }
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
    if (I->mayHaveSideEffects())
      return false;
  return true;
}

/// UnswitchIfProfitable - We have found that we can unswitch currentLoop when
/// LoopCond == Val to simplify the loop.  If we decide that this is profitable,
/// unswitch the loop, reprocess the pieces, then return true.
bool LoopUnswitch::UnswitchIfProfitable(Value *LoopCond, Constant *Val) {
  Function *F = loopHeader->getParent();
  Constant *CondVal = 0;
  BasicBlock *ExitBlock = 0;

  if (IsTrivialUnswitchCondition(LoopCond, &CondVal, &ExitBlock)) {
    // If the condition is trivial, always unswitch. There is no code growth
    // for this case.
    UnswitchTrivialCondition(currentLoop, LoopCond, CondVal, ExitBlock);
    return true;
  }

  // Check to see if it would be profitable to unswitch current loop.

  // Do not do non-trivial unswitch while optimizing for size.
  if (OptimizeForSize || F->hasFnAttr(Attribute::OptimizeForSize))
    return false;

  UnswitchNontrivialCondition(LoopCond, Val, currentLoop);
  return true;
}

/// CloneLoop - Recursively clone the specified loop and all of its children,
/// mapping the blocks with the specified map.
static Loop *CloneLoop(Loop *L, Loop *PL, ValueToValueMapTy &VM,
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
  if (!isa<ConstantInt>(Val) ||
      Val->getType() != Type::getInt1Ty(LIC->getContext()))
    BranchVal = new ICmpInst(InsertPt, ICmpInst::ICMP_EQ, LIC, Val);
  else if (Val != ConstantInt::getTrue(Val->getContext()))
    // We want to enter the new loop when the condition is true.
    std::swap(TrueDest, FalseDest);

  // Insert the new branch.
  BranchInst *BI = BranchInst::Create(TrueDest, FalseDest, BranchVal, InsertPt);

  // If either edge is critical, split it. This helps preserve LoopSimplify
  // form for enclosing loops.
  SplitCriticalEdge(BI, 0, this);
  SplitCriticalEdge(BI, 1, this);
}

/// UnswitchTrivialCondition - Given a loop that has a trivial unswitchable
/// condition in it (a cond branch from its header block to its latch block,
/// where the path through the loop that doesn't execute its body has no
/// side-effects), unswitch it.  This doesn't involve any code duplication, just
/// moving the conditional branch outside of the loop and updating loop info.
void LoopUnswitch::UnswitchTrivialCondition(Loop *L, Value *Cond,
                                            Constant *Val,
                                            BasicBlock *ExitBlock) {
  DEBUG(dbgs() << "loop-unswitch: Trivial-Unswitch loop %"
        << loopHeader->getName() << " [" << L->getBlocks().size()
        << " blocks] in Function " << L->getHeader()->getParent()->getName()
        << " on cond: " << *Val << " == " << *Cond << "\n");

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
                                const SmallVector<BasicBlock *, 8> &ExitBlocks){

  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    BasicBlock *ExitBlock = ExitBlocks[i];
    SmallVector<BasicBlock *, 4> Preds(pred_begin(ExitBlock),
                                       pred_end(ExitBlock));

    // Although SplitBlockPredecessors doesn't preserve loop-simplify in
    // general, if we call it on all predecessors of all exits then it does.
    if (!ExitBlock->isLandingPad()) {
      SplitBlockPredecessors(ExitBlock, Preds, ".us-lcssa", this);
    } else {
      SmallVector<BasicBlock*, 2> NewBBs;
      SplitLandingPadPredecessors(ExitBlock, Preds, ".us-lcssa", ".us-lcssa",
                                  this, NewBBs);
    }
  }
}

/// UnswitchNontrivialCondition - We determined that the loop is profitable
/// to unswitch when LIC equal Val.  Split it into loop versions and test the
/// condition outside of either loop.  Return the loops created as Out1/Out2.
void LoopUnswitch::UnswitchNontrivialCondition(Value *LIC, Constant *Val,
                                               Loop *L) {
  Function *F = loopHeader->getParent();
  DEBUG(dbgs() << "loop-unswitch: Unswitching loop %"
        << loopHeader->getName() << " [" << L->getBlocks().size()
        << " blocks] in Function " << F->getName()
        << " when '" << *Val << "' == " << *LIC << "\n");

  if (ScalarEvolution *SE = getAnalysisIfAvailable<ScalarEvolution>())
    SE->forgetLoop(L);

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
  ValueToValueMapTy VMap;
  for (unsigned i = 0, e = LoopBlocks.size(); i != e; ++i) {
    BasicBlock *NewBB = CloneBasicBlock(LoopBlocks[i], VMap, ".us", F);

    NewBlocks.push_back(NewBB);
    VMap[LoopBlocks[i]] = NewBB;  // Keep the BB mapping.
    LPM->cloneBasicBlockSimpleAnalysis(LoopBlocks[i], NewBB, L);
  }

  // Splice the newly inserted blocks into the function right before the
  // original preheader.
  F->getBasicBlockList().splice(NewPreheader, F->getBasicBlockList(),
                                NewBlocks[0], F->end());

  // Now we create the new Loop object for the versioned loop.
  Loop *NewLoop = CloneLoop(L, L->getParentLoop(), VMap, LI, LPM);

  // Recalculate unswitching quota, inherit simplified switches info for NewBB,
  // Probably clone more loop-unswitch related loop properties.
  BranchesInfo.cloneData(NewLoop, L, VMap);

  Loop *ParentLoop = L->getParentLoop();
  if (ParentLoop) {
    // Make sure to add the cloned preheader and exit blocks to the parent loop
    // as well.
    ParentLoop->addBasicBlockToLoop(NewBlocks[0], LI->getBase());
  }

  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    BasicBlock *NewExit = cast<BasicBlock>(VMap[ExitBlocks[i]]);
    // The new exit block should be in the same loop as the old one.
    if (Loop *ExitBBLoop = LI->getLoopFor(ExitBlocks[i]))
      ExitBBLoop->addBasicBlockToLoop(NewExit, LI->getBase());

    assert(NewExit->getTerminator()->getNumSuccessors() == 1 &&
           "Exit block should have been split to have one successor!");
    BasicBlock *ExitSucc = NewExit->getTerminator()->getSuccessor(0);

    // If the successor of the exit block had PHI nodes, add an entry for
    // NewExit.
    PHINode *PN;
    for (BasicBlock::iterator I = ExitSucc->begin(); isa<PHINode>(I); ++I) {
      PN = cast<PHINode>(I);
      Value *V = PN->getIncomingValueForBlock(ExitBlocks[i]);
      ValueToValueMapTy::iterator It = VMap.find(V);
      if (It != VMap.end()) V = It->second;
      PN->addIncoming(V, NewExit);
    }

    if (LandingPadInst *LPad = NewExit->getLandingPadInst()) {
      PN = PHINode::Create(LPad->getType(), 0, "",
                           ExitSucc->getFirstInsertionPt());

      for (pred_iterator I = pred_begin(ExitSucc), E = pred_end(ExitSucc);
           I != E; ++I) {
        BasicBlock *BB = *I;
        LandingPadInst *LPI = BB->getLandingPadInst();
        LPI->replaceAllUsesWith(PN);
        PN->addIncoming(LPI, BB);
      }
    }
  }

  // Rewrite the code to refer to itself.
  for (unsigned i = 0, e = NewBlocks.size(); i != e; ++i)
    for (BasicBlock::iterator I = NewBlocks[i]->begin(),
           E = NewBlocks[i]->end(); I != E; ++I)
      RemapInstruction(I, VMap,RF_NoModuleLevelChanges|RF_IgnoreMissingEntries);

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

  // Keep a WeakVH holding onto LIC.  If the first call to RewriteLoopBody
  // deletes the instruction (for example by simplifying a PHI that feeds into
  // the condition that we're unswitching on), we don't rewrite the second
  // iteration.
  WeakVH LICHandle(LIC);

  // Now we rewrite the original code to know that the condition is true and the
  // new code to know that the condition is false.
  RewriteLoopBodyWithConditionConstant(L, LIC, Val, false);

  // It's possible that simplifying one loop could cause the other to be
  // changed to another value or a constant.  If its a constant, don't simplify
  // it.
  if (!LoopProcessWorklist.empty() && LoopProcessWorklist.back() == NewLoop &&
      LICHandle && !isa<Constant>(LICHandle))
    RewriteLoopBodyWithConditionConstant(NewLoop, LICHandle, Val, true);
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
  DEBUG(dbgs() << "Replace with '" << *V << "': " << *I);

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
          new UnreachableInst(BB->getContext(), Pred);

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

  DEBUG(dbgs() << "Nuking dead block: " << *BB);

  // Remove the instructions in the basic block from the worklist.
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
    RemoveFromWorklist(I, Worklist);

    // Anything that uses the instructions in this basic block should have their
    // uses replaced with undefs.
    // If I is not void type then replaceAllUsesWith undef.
    // This allows ValueHandlers and custom metadata to adjust itself.
    if (!I->getType()->isVoidTy())
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
  }

  // If this is the edge to the header block for a loop, remove the loop and
  // promote all subloops.
  if (Loop *BBLoop = LI->getLoopFor(BB)) {
    if (BBLoop->getLoopLatch() == BB) {
      RemoveLoopFromHierarchy(BBLoop);
      if (currentLoop == BBLoop) {
        currentLoop = 0;
        redoLoop = false;
      }
    }
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
  std::vector<Instruction*> Worklist;
  LLVMContext &Context = Val->getContext();


  // If we know that LIC == Val, or that LIC == NotVal, just replace uses of LIC
  // in the loop with the appropriate one directly.
  if (IsEqual || (isa<ConstantInt>(Val) &&
      Val->getType()->isIntegerTy(1))) {
    Value *Replacement;
    if (IsEqual)
      Replacement = Val;
    else
      Replacement = ConstantInt::get(Type::getInt1Ty(Val->getContext()),
                                     !cast<ConstantInt>(Val)->getZExtValue());

    for (Value::use_iterator UI = LIC->use_begin(), E = LIC->use_end();
         UI != E; ++UI) {
      Instruction *U = dyn_cast<Instruction>(*UI);
      if (!U || !L->contains(U))
        continue;
      Worklist.push_back(U);
    }

    for (std::vector<Instruction*>::iterator UI = Worklist.begin();
         UI != Worklist.end(); ++UI)
      (*UI)->replaceUsesOfWith(LIC, Replacement);

    SimplifyCode(Worklist, L);
    return;
  }

  // Otherwise, we don't know the precise value of LIC, but we do know that it
  // is certainly NOT "Val".  As such, simplify any uses in the loop that we
  // can.  This case occurs when we unswitch switch statements.
  for (Value::use_iterator UI = LIC->use_begin(), E = LIC->use_end();
       UI != E; ++UI) {
    Instruction *U = dyn_cast<Instruction>(*UI);
    if (!U || !L->contains(U))
      continue;

    Worklist.push_back(U);

    // TODO: We could do other simplifications, for example, turning
    // 'icmp eq LIC, Val' -> false.

    // If we know that LIC is not Val, use this info to simplify code.
    SwitchInst *SI = dyn_cast<SwitchInst>(U);
    if (SI == 0 || !isa<ConstantInt>(Val)) continue;

    SwitchInst::CaseIt DeadCase = SI->findCaseValue(cast<ConstantInt>(Val));
    // Default case is live for multiple values.
    if (DeadCase == SI->case_default()) continue;

    // Found a dead case value.  Don't remove PHI nodes in the
    // successor if they become single-entry, those PHI nodes may
    // be in the Users list.

    BasicBlock *Switch = SI->getParent();
    BasicBlock *SISucc = DeadCase.getCaseSuccessor();
    BasicBlock *Latch = L->getLoopLatch();

    BranchesInfo.setUnswitched(SI, Val);

    if (!SI->findCaseDest(SISucc)) continue;  // Edge is critical.
    // If the DeadCase successor dominates the loop latch, then the
    // transformation isn't safe since it will delete the sole predecessor edge
    // to the latch.
    if (Latch && DT->dominates(SISucc, Latch))
      continue;

    // FIXME: This is a hack.  We need to keep the successor around
    // and hooked up so as to preserve the loop structure, because
    // trying to update it is complicated.  So instead we preserve the
    // loop structure and put the block on a dead code path.
    SplitEdge(Switch, SISucc, this);
    // Compute the successors instead of relying on the return value
    // of SplitEdge, since it may have split the switch successor
    // after PHI nodes.
    BasicBlock *NewSISucc = DeadCase.getCaseSuccessor();
    BasicBlock *OldSISucc = *succ_begin(NewSISucc);
    // Create an "unreachable" destination.
    BasicBlock *Abort = BasicBlock::Create(Context, "us-unreachable",
                                           Switch->getParent(),
                                           OldSISucc);
    new UnreachableInst(Context, Abort);
    // Force the new case destination to branch to the "unreachable"
    // block while maintaining a (dead) CFG edge to the old block.
    NewSISucc->getTerminator()->eraseFromParent();
    BranchInst::Create(Abort, OldSISucc,
                       ConstantInt::getTrue(Context), NewSISucc);
    // Release the PHI operands for this edge.
    for (BasicBlock::iterator II = NewSISucc->begin();
         PHINode *PN = dyn_cast<PHINode>(II); ++II)
      PN->setIncomingValue(PN->getBasicBlockIndex(Switch),
                           UndefValue::get(PN->getType()));
    // Tell the domtree about the new block. We don't fully update the
    // domtree here -- instead we force it to do a full recomputation
    // after the pass is complete -- but we do need to inform it of
    // new blocks.
    if (DT)
      DT->addNewBlock(Abort, NewSISucc);
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

    // Simple DCE.
    if (isInstructionTriviallyDead(I)) {
      DEBUG(dbgs() << "Remove dead instruction '" << *I);

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

    // See if instruction simplification can hack this up.  This is common for
    // things like "select false, X, Y" after unswitching made the condition be
    // 'false'.
    if (Value *V = SimplifyInstruction(I, 0, 0, DT))
      if (LI->replacementPreservesLCSSAForm(I, V)) {
        ReplaceUsesOfWith(I, V, Worklist, L, LPM);
        continue;
      }

    // Special case hacks that appear commonly in unswitched code.
    if (BranchInst *BI = dyn_cast<BranchInst>(I)) {
      if (BI->isUnconditional()) {
        // If BI's parent is the only pred of the successor, fold the two blocks
        // together.
        BasicBlock *Pred = BI->getParent();
        BasicBlock *Succ = BI->getSuccessor(0);
        BasicBlock *SinglePred = Succ->getSinglePredecessor();
        if (!SinglePred) continue;  // Nothing to do.
        assert(SinglePred == Pred && "CFG broken");

        DEBUG(dbgs() << "Merging blocks: " << Pred->getName() << " <- "
              << Succ->getName() << "\n");

        // Resolve any single entry PHI nodes in Succ.
        while (PHINode *PN = dyn_cast<PHINode>(Succ->begin()))
          ReplaceUsesOfWith(PN, PN->getIncomingValue(0), Worklist, L, LPM);

        // If Succ has any successors with PHI nodes, update them to have
        // entries coming from Pred instead of Succ.
        Succ->replaceAllUsesWith(Pred);

        // Move all of the successor contents from Succ to Pred.
        Pred->getInstList().splice(BI, Succ->getInstList(), Succ->begin(),
                                   Succ->end());
        LPM->deleteSimpleAnalysisValue(BI, L);
        BI->eraseFromParent();
        RemoveFromWorklist(BI, Worklist);

        // Remove Succ from the loop tree.
        LI->removeBlock(Succ);
        LPM->deleteSimpleAnalysisValue(Succ, L);
        Succ->eraseFromParent();
        ++NumSimplify;
        continue;
      }

      if (ConstantInt *CB = dyn_cast<ConstantInt>(BI->getCondition())){
        // Conditional branch.  Turn it into an unconditional branch, then
        // remove dead blocks.
        continue;  // FIXME: Enable.

        DEBUG(dbgs() << "Folded branch: " << *BI);
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
      continue;
    }
  }
}
