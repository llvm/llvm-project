//===-- UnrollLoop.cpp - Loop unrolling utilities -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements some loop unrolling utilities. It does not define any
// actual pass or policy, but provides a single function to perform loop
// unrolling.
//
// The process of unrolling can produce extraneous basic blocks linked with
// unconditional branches.  This will be corrected in the future.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <assert.h>
#include <cmath>
#include <numeric>
#include <vector>

namespace llvm {
class DataLayout;
class Value;
} // namespace llvm

using namespace llvm;

#define DEBUG_TYPE "loop-unroll"

// TODO: Should these be here or in LoopUnroll?
STATISTIC(NumCompletelyUnrolled, "Number of loops completely unrolled");
STATISTIC(NumUnrolled, "Number of loops unrolled (completely or otherwise)");
STATISTIC(NumUnrolledNotLatch, "Number of loops unrolled without a conditional "
                               "latch (completely or otherwise)");

static cl::opt<bool>
UnrollRuntimeEpilog("unroll-runtime-epilog", cl::init(false), cl::Hidden,
                    cl::desc("Allow runtime unrolled loops to be unrolled "
                             "with epilog instead of prolog."));

static cl::opt<bool> UnrollUniformWeights(
    "unroll-uniform-weights", cl::init(false), cl::Hidden,
    cl::desc("If new branch weights must be found, work harder to keep them "
             "uniform."));

static cl::opt<bool>
UnrollVerifyDomtree("unroll-verify-domtree", cl::Hidden,
                    cl::desc("Verify domtree after unrolling"),
#ifdef EXPENSIVE_CHECKS
    cl::init(true)
#else
    cl::init(false)
#endif
                    );

static cl::opt<bool>
UnrollVerifyLoopInfo("unroll-verify-loopinfo", cl::Hidden,
                    cl::desc("Verify loopinfo after unrolling"),
#ifdef EXPENSIVE_CHECKS
    cl::init(true)
#else
    cl::init(false)
#endif
                    );

static cl::opt<bool> UnrollAddParallelReductions(
    "unroll-add-parallel-reductions", cl::init(false), cl::Hidden,
    cl::desc("Allow unrolling to add parallel reduction phis."));

/// Check if unrolling created a situation where we need to insert phi nodes to
/// preserve LCSSA form.
/// \param Blocks is a vector of basic blocks representing unrolled loop.
/// \param L is the outer loop.
/// It's possible that some of the blocks are in L, and some are not. In this
/// case, if there is a use is outside L, and definition is inside L, we need to
/// insert a phi-node, otherwise LCSSA will be broken.
/// The function is just a helper function for llvm::UnrollLoop that returns
/// true if this situation occurs, indicating that LCSSA needs to be fixed.
static bool needToInsertPhisForLCSSA(Loop *L,
                                     const std::vector<BasicBlock *> &Blocks,
                                     LoopInfo *LI) {
  for (BasicBlock *BB : Blocks) {
    if (LI->getLoopFor(BB) == L)
      continue;
    for (Instruction &I : *BB) {
      for (Use &U : I.operands()) {
        if (const auto *Def = dyn_cast<Instruction>(U)) {
          Loop *DefLoop = LI->getLoopFor(Def->getParent());
          if (!DefLoop)
            continue;
          if (DefLoop->contains(L))
            return true;
        }
      }
    }
  }
  return false;
}

/// Adds ClonedBB to LoopInfo, creates a new loop for ClonedBB if necessary
/// and adds a mapping from the original loop to the new loop to NewLoops.
/// Returns nullptr if no new loop was created and a pointer to the
/// original loop OriginalBB was part of otherwise.
const Loop* llvm::addClonedBlockToLoopInfo(BasicBlock *OriginalBB,
                                           BasicBlock *ClonedBB, LoopInfo *LI,
                                           NewLoopsMap &NewLoops) {
  // Figure out which loop New is in.
  const Loop *OldLoop = LI->getLoopFor(OriginalBB);
  assert(OldLoop && "Should (at least) be in the loop being unrolled!");

  Loop *&NewLoop = NewLoops[OldLoop];
  if (!NewLoop) {
    // Found a new sub-loop.
    assert(OriginalBB == OldLoop->getHeader() &&
           "Header should be first in RPO");

    NewLoop = LI->AllocateLoop();
    Loop *NewLoopParent = NewLoops.lookup(OldLoop->getParentLoop());

    if (NewLoopParent)
      NewLoopParent->addChildLoop(NewLoop);
    else
      LI->addTopLevelLoop(NewLoop);

    NewLoop->addBasicBlockToLoop(ClonedBB, *LI);
    return OldLoop;
  } else {
    NewLoop->addBasicBlockToLoop(ClonedBB, *LI);
    return nullptr;
  }
}

/// The function chooses which type of unroll (epilog or prolog) is more
/// profitabale.
/// Epilog unroll is more profitable when there is PHI that starts from
/// constant.  In this case epilog will leave PHI start from constant,
/// but prolog will convert it to non-constant.
///
/// loop:
///   PN = PHI [I, Latch], [CI, PreHeader]
///   I = foo(PN)
///   ...
///
/// Epilog unroll case.
/// loop:
///   PN = PHI [I2, Latch], [CI, PreHeader]
///   I1 = foo(PN)
///   I2 = foo(I1)
///   ...
/// Prolog unroll case.
///   NewPN = PHI [PrologI, Prolog], [CI, PreHeader]
/// loop:
///   PN = PHI [I2, Latch], [NewPN, PreHeader]
///   I1 = foo(PN)
///   I2 = foo(I1)
///   ...
///
static bool isEpilogProfitable(Loop *L) {
  BasicBlock *PreHeader = L->getLoopPreheader();
  BasicBlock *Header = L->getHeader();
  assert(PreHeader && Header);
  for (const PHINode &PN : Header->phis()) {
    if (isa<ConstantInt>(PN.getIncomingValueForBlock(PreHeader)))
      return true;
  }
  return false;
}

struct LoadValue {
  Instruction *DefI = nullptr;
  unsigned Generation = 0;
  LoadValue() = default;
  LoadValue(Instruction *Inst, unsigned Generation)
      : DefI(Inst), Generation(Generation) {}
};

class StackNode {
  ScopedHashTable<const SCEV *, LoadValue>::ScopeTy LoadScope;
  unsigned CurrentGeneration;
  unsigned ChildGeneration;
  DomTreeNode *Node;
  DomTreeNode::const_iterator ChildIter;
  DomTreeNode::const_iterator EndIter;
  bool Processed = false;

public:
  StackNode(ScopedHashTable<const SCEV *, LoadValue> &AvailableLoads,
            unsigned cg, DomTreeNode *N, DomTreeNode::const_iterator Child,
            DomTreeNode::const_iterator End)
      : LoadScope(AvailableLoads), CurrentGeneration(cg), ChildGeneration(cg),
        Node(N), ChildIter(Child), EndIter(End) {}
  // Accessors.
  unsigned currentGeneration() const { return CurrentGeneration; }
  unsigned childGeneration() const { return ChildGeneration; }
  void childGeneration(unsigned generation) { ChildGeneration = generation; }
  DomTreeNode *node() { return Node; }
  DomTreeNode::const_iterator childIter() const { return ChildIter; }

  DomTreeNode *nextChild() {
    DomTreeNode *Child = *ChildIter;
    ++ChildIter;
    return Child;
  }

  DomTreeNode::const_iterator end() const { return EndIter; }
  bool isProcessed() const { return Processed; }
  void process() { Processed = true; }
};

Value *getMatchingValue(LoadValue LV, LoadInst *LI, unsigned CurrentGeneration,
                        BatchAAResults &BAA,
                        function_ref<MemorySSA *()> GetMSSA) {
  if (!LV.DefI)
    return nullptr;
  if (LV.DefI->getType() != LI->getType())
    return nullptr;
  if (LV.Generation != CurrentGeneration) {
    MemorySSA *MSSA = GetMSSA();
    if (!MSSA)
      return nullptr;
    auto *EarlierMA = MSSA->getMemoryAccess(LV.DefI);
    MemoryAccess *LaterDef =
        MSSA->getWalker()->getClobberingMemoryAccess(LI, BAA);
    if (!MSSA->dominates(LaterDef, EarlierMA))
      return nullptr;
  }
  return LV.DefI;
}

void loadCSE(Loop *L, DominatorTree &DT, ScalarEvolution &SE, LoopInfo &LI,
             BatchAAResults &BAA, function_ref<MemorySSA *()> GetMSSA) {
  ScopedHashTable<const SCEV *, LoadValue> AvailableLoads;
  SmallVector<std::unique_ptr<StackNode>> NodesToProcess;
  DomTreeNode *HeaderD = DT.getNode(L->getHeader());
  NodesToProcess.emplace_back(new StackNode(AvailableLoads, 0, HeaderD,
                                            HeaderD->begin(), HeaderD->end()));

  unsigned CurrentGeneration = 0;
  while (!NodesToProcess.empty()) {
    StackNode *NodeToProcess = &*NodesToProcess.back();

    CurrentGeneration = NodeToProcess->currentGeneration();

    if (!NodeToProcess->isProcessed()) {
      // Process the node.

      // If this block has a single predecessor, then the predecessor is the
      // parent
      // of the domtree node and all of the live out memory values are still
      // current in this block.  If this block has multiple predecessors, then
      // they could have invalidated the live-out memory values of our parent
      // value.  For now, just be conservative and invalidate memory if this
      // block has multiple predecessors.
      if (!NodeToProcess->node()->getBlock()->getSinglePredecessor())
        ++CurrentGeneration;
      for (auto &I : make_early_inc_range(*NodeToProcess->node()->getBlock())) {

        auto *Load = dyn_cast<LoadInst>(&I);
        if (!Load || !Load->isSimple()) {
          if (I.mayWriteToMemory())
            CurrentGeneration++;
          continue;
        }

        const SCEV *PtrSCEV = SE.getSCEV(Load->getPointerOperand());
        LoadValue LV = AvailableLoads.lookup(PtrSCEV);
        if (Value *M =
                getMatchingValue(LV, Load, CurrentGeneration, BAA, GetMSSA)) {
          if (LI.replacementPreservesLCSSAForm(Load, M)) {
            Load->replaceAllUsesWith(M);
            Load->eraseFromParent();
          }
        } else {
          AvailableLoads.insert(PtrSCEV, LoadValue(Load, CurrentGeneration));
        }
      }
      NodeToProcess->childGeneration(CurrentGeneration);
      NodeToProcess->process();
    } else if (NodeToProcess->childIter() != NodeToProcess->end()) {
      // Push the next child onto the stack.
      DomTreeNode *Child = NodeToProcess->nextChild();
      if (!L->contains(Child->getBlock()))
        continue;
      NodesToProcess.emplace_back(
          new StackNode(AvailableLoads, NodeToProcess->childGeneration(), Child,
                        Child->begin(), Child->end()));
    } else {
      // It has been processed, and there are no more children to process,
      // so delete it and pop it off the stack.
      NodesToProcess.pop_back();
    }
  }
}

/// Perform some cleanup and simplifications on loops after unrolling. It is
/// useful to simplify the IV's in the new loop, as well as do a quick
/// simplify/dce pass of the instructions.
void llvm::simplifyLoopAfterUnroll(Loop *L, bool SimplifyIVs, LoopInfo *LI,
                                   ScalarEvolution *SE, DominatorTree *DT,
                                   AssumptionCache *AC,
                                   const TargetTransformInfo *TTI,
                                   AAResults *AA) {
  using namespace llvm::PatternMatch;

  // Simplify any new induction variables in the partially unrolled loop.
  if (SE && SimplifyIVs) {
    SmallVector<WeakTrackingVH, 16> DeadInsts;
    simplifyLoopIVs(L, SE, DT, LI, TTI, DeadInsts);

    // Aggressively clean up dead instructions that simplifyLoopIVs already
    // identified. Any remaining should be cleaned up below.
    while (!DeadInsts.empty()) {
      Value *V = DeadInsts.pop_back_val();
      if (Instruction *Inst = dyn_cast_or_null<Instruction>(V))
        RecursivelyDeleteTriviallyDeadInstructions(Inst);
    }

    if (AA) {
      std::unique_ptr<MemorySSA> MSSA = nullptr;
      BatchAAResults BAA(*AA);
      loadCSE(L, *DT, *SE, *LI, BAA, [L, AA, DT, &MSSA]() -> MemorySSA * {
        if (!MSSA)
          MSSA.reset(new MemorySSA(*L, AA, DT));
        return &*MSSA;
      });
    }
  }

  // At this point, the code is well formed.  Perform constprop, instsimplify,
  // and dce.
  const DataLayout &DL = L->getHeader()->getDataLayout();
  SmallVector<WeakTrackingVH, 16> DeadInsts;
  for (BasicBlock *BB : L->getBlocks()) {
    // Remove repeated debug instructions after loop unrolling.
    if (BB->getParent()->getSubprogram())
      RemoveRedundantDbgInstrs(BB);

    for (Instruction &Inst : llvm::make_early_inc_range(*BB)) {
      if (Value *V = simplifyInstruction(&Inst, {DL, nullptr, DT, AC}))
        if (LI->replacementPreservesLCSSAForm(&Inst, V))
          Inst.replaceAllUsesWith(V);
      if (isInstructionTriviallyDead(&Inst))
        DeadInsts.emplace_back(&Inst);

      // Fold ((add X, C1), C2) to (add X, C1+C2). This is very common in
      // unrolled loops, and handling this early allows following code to
      // identify the IV as a "simple recurrence" without first folding away
      // a long chain of adds.
      {
        Value *X;
        const APInt *C1, *C2;
        if (match(&Inst, m_Add(m_Add(m_Value(X), m_APInt(C1)), m_APInt(C2)))) {
          auto *InnerI = dyn_cast<Instruction>(Inst.getOperand(0));
          auto *InnerOBO = cast<OverflowingBinaryOperator>(Inst.getOperand(0));
          bool SignedOverflow;
          APInt NewC = C1->sadd_ov(*C2, SignedOverflow);
          Inst.setOperand(0, X);
          Inst.setOperand(1, ConstantInt::get(Inst.getType(), NewC));
          Inst.setHasNoUnsignedWrap(Inst.hasNoUnsignedWrap() &&
                                    InnerOBO->hasNoUnsignedWrap());
          Inst.setHasNoSignedWrap(Inst.hasNoSignedWrap() &&
                                  InnerOBO->hasNoSignedWrap() &&
                                  !SignedOverflow);
          if (InnerI && isInstructionTriviallyDead(InnerI))
            DeadInsts.emplace_back(InnerI);
        }
      }
    }
    // We can't do recursive deletion until we're done iterating, as we might
    // have a phi which (potentially indirectly) uses instructions later in
    // the block we're iterating through.
    RecursivelyDeleteTriviallyDeadInstructions(DeadInsts);
  }
}

// Loops containing convergent instructions that are uncontrolled or controlled
// from outside the loop must have a count that divides their TripMultiple.
LLVM_ATTRIBUTE_USED
static bool canHaveUnrollRemainder(const Loop *L) {
  if (getLoopConvergenceHeart(L))
    return false;

  // Check for uncontrolled convergent operations.
  for (auto &BB : L->blocks()) {
    for (auto &I : *BB) {
      if (isa<ConvergenceControlInst>(I))
        return true;
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (CB->isConvergent())
          return CB->getConvergenceControlToken();
    }
  }
  return true;
}

// If LoopUnroll has proven OriginalLoopProb is incorrect for some iterations
// of the original loop, adjust latch probabilities in the unrolled loop to
// maintain the original total frequency of the original loop body.
//
// OriginalLoopProb is practical but imprecise
// -------------------------------------------
//
// The latch branch weights that LLVM originally adds to a loop encode one latch
// probability, OriginalLoopProb, applied uniformly across the loop's infinite
// set of theoretically possible iterations.  While this uniform latch
// probability serves as a practical statistic summarizing the trip counts
// observed during profiling, it is imprecise.  Specifically, unless it is zero,
// it is impossible for it to be the actual probability observed at every
// individual iteration.  To see why, consider that the only way to actually
// observe at run time that the latch probability remains non-zero is to profile
// at least one loop execution that has an infinite number of iterations.  I do
// not know how to profile an infinite number of loop iterations, and most loops
// I work with are always finite.
//
// LoopUnroll proves OriginalLoopProb is incorrect
// ------------------------------------------------
//
// LoopUnroll reorganizes the original loop so that loop iterations are no
// longer all implemented by the same code, and then it analyzes some of those
// loop iteration implementations independently of others.  In particular, it
// converts some of their conditional latches to unconditional.  That is, by
// examining code structure without any profile data, LoopUnroll proves that the
// actual latch probability at the end of such an iteration is either 1 or 0.
// When an individual iteration's actual latch probability is 1 or 0, that means
// it always behaves the same, so it is impossible to observe it as having any
// other probability.  The original uniform latch probability is rarely 1 or 0
// because, when applied to all possible iterations, that would yield an
// estimated trip count of infinity or 1, respectively.
//
// Thus, the new probabilities of 1 or 0 are proven corrections to
// OriginalLoopProb for individual iterations in the original loop.  However,
// LoopUnroll often is able to perform these corrections for only some
// iterations, leaving other iterations with OriginalLoopProb, and thus
// corrupting the aggregate effect on the total frequency of the original loop
// body.
//
// Adjusting latch probabilities
// -----------------------------
//
// This function ensures that the total frequency of the original loop body,
// summed across all its occurrences in the unrolled loop after the
// aforementioned latch conversions, is the same as in the original loop.  To do
// so, it adjusts probabilities on the remaining conditional latches.  However,
// it cannot derive the new probabilities directly from the original uniform
// latch probability because the latter has been proven incorrect for some
// original loop iterations.
//
// There are often many sets of latch probabilities that can produce the
// original total loop body frequency.  If there are many remaining conditional
// latches and !UnrollUniformWeights, this function just quickly hacks a few of
// their probabilities to restore the original total loop body frequency.
// Otherwise, it tries harder to determine less arbitrary probabilities.
static void fixProbContradiction(UnrollLoopOptions ULO,
                                 BranchProbability OriginalLoopProb,
                                 bool CompletelyUnroll,
                                 std::vector<unsigned> &IterCounts,
                                 const std::vector<BasicBlock *> &CondLatches,
                                 std::vector<BasicBlock *> &CondLatchNexts) {
  // Runtime unrolling is handled later in LoopUnroll not here.
  //
  // There are two scenarios in which LoopUnroll sets ProbUpdateRequired to true
  // because it needs to update probabilities that were originally
  // OriginalLoopProb, but only in one scenario has LoopUnroll proven
  // OriginalLoopProb incorrect for iterations within the original loop:
  // - If ULO.Runtime, LoopUnroll adds new guards that enforce new reaching
  //   conditions for new loop iteration implementations (e.g., one unrolled
  //   loop iteration executes only if at least ULO.Count original loop
  //   iterations remain).  Those reaching conditions dictate how conditional
  //   latches can be converted to unconditional (e.g., within an unrolled loop
  //   iteration, there is no need to recheck the number of remaining original
  //   loop iterations).  None of this reorganization alters the set of possible
  //   original loop iteration counts or proves OriginalLoopProb incorrect for
  //   any of the original loop iterations.  Thus, LoopUnroll derives
  //   probabilities for the new guards and latches directly from
  //   OriginalLoopProb based on the probabilities that their reaching
  //   conditions would occur in the original loop.  Doing so maintains the
  //   total frequency of the original loop body.
  // - If !ULO.Runtime, LoopUnroll initially adds new loop iteration
  //   implementations, which have the same latch probabilities as in the
  //   original loop because there are no new guards that change their reaching
  //   conditions.  Sometimes, LoopUnroll is then done, and so does not set
  //   ProbUpdateRequired to true.  Other times, LoopUnroll then proves that
  //   some latches are unconditional, directly contradicting OriginalLoopProb
  //   for the corresponding original loop iterations.  That reduces the set of
  //   possible original loop iteration counts, possibly producing a finite set
  //   if it manages to eliminate the backedge.  LoopUnroll has to choose a new
  //   set of latch probabilities that produce the same total loop body
  //   frequency.
  //
  // This function addresses the second scenario only.
  if (ULO.Runtime)
    return;

  // If CondLatches.empty(), there are no latch branches with probabilities we
  // can adjust.  That should mean that the actual trip count is always exactly
  // the number of remaining unrolled iterations, and so OriginalLoopProb should
  // have yielded that trip count as the original loop body frequency.  Of
  // course, OriginalLoopProb could be based on inaccurate profile data, but
  // there is nothing we can do about that here.
  if (CondLatches.empty())
    return;

  // If the original latch probability is 1, the original frequency is infinity.
  // Leaving all remaining probabilities set to 1 might or might not get us
  // there (e.g., a completely unrolled loop cannot be infinite), but it is the
  // closest we can come.
  assert(!OriginalLoopProb.isUnknown() &&
         "Expected to have loop probability to fix");
  if (OriginalLoopProb.isOne())
    return;

  // FreqDesired is the frequency implied by the original loop probability.
  double FreqDesired = 1 / (1 - OriginalLoopProb.toDouble());

  // Get the probability at CondLatches[I].
  auto GetProb = [&](unsigned I) {
    BranchInst *B = cast<BranchInst>(CondLatches[I]->getTerminator());
    bool FirstTargetIsNext = B->getSuccessor(0) == CondLatchNexts[I];
    return getBranchProbability(B, FirstTargetIsNext).toDouble();
  };

  // Set the probability at CondLatches[I] to Prob.
  auto SetProb = [&](unsigned I, double Prob) {
    BranchInst *B = cast<BranchInst>(CondLatches[I]->getTerminator());
    bool FirstTargetIsNext = B->getSuccessor(0) == CondLatchNexts[I];
    bool Success = setBranchProbability(
        B, BranchProbability::getBranchProbability(Prob), FirstTargetIsNext);
    assert(Success && "Expected to be able to set branch probability");
  };

  // Set all probabilities in CondLatches to Prob.
  auto SetAllProbs = [&](double Prob) {
    for (unsigned I = 0, E = CondLatches.size(); I < E; ++I)
      SetProb(I, Prob);
  };

  // If UnrollUniformWeights or n <= 2, we choose the simplest probability model
  // we can think of: every remaining conditional branch instruction has the
  // same probability, Prob, of continuing to the next iteration.  This model
  // has several helpful properties:
  // - There is only one search parameter, Prob.
  // - We have no reason to think one latch branch's probability should be
  //   higher or lower than another, and so this model makes them all the same.
  //   In the worst cases, we thus avoid setting just some probabilities to 0 or
  //   1, which can unrealistically make some code appear unreachable.  There
  //   are cases where they *all* must become 0 or 1 to achieve the total
  //   frequency of original loop body, and our model does permit that.
  // - The frequency, FreqOne, of the original loop body in a single iteration
  //   of the unrolled loop is computed by a simple polynomial, where p=Prob,
  //   n=CondLatches.size(), and c_i=IterCounts[i]:
  //
  //     FreqOne = Sum(i=0..n)(c_i * p^i)
  //
  // - If the backedge has been eliminated:
  //   - FreqOne is the total frequency of the original loop body in the
  //     unrolled loop.
  //   - If Prob == 1, the total frequency of the original loop body is exactly
  //     the number of remaining loop iterations, as expected because every
  //     remaining loop iteration always then executes.
  // - If the backedge remains:
  //   - Sum(i=0..inf)(FreqOne * p^(n*i)) = FreqOne / (1 - p^n) is the total
  //     frequency of the original loop body in the unrolled loop, regardless of
  //     whether the backedge is conditional or unconditional.
  //   - As Prob approaches 1, the total frequency of the original loop body
  //     approaches infinity, as expected because the loop approaches never
  //     exiting.
  // - For n <= 2, we can use simple formulas to solve the above polynomial
  //   equations exactly for p without performing a search.
  // - For n > 2, evaluating each point in the search space, using ComputeFreq
  //   below, requires about as few instructions as we could hope for.  That is,
  //   the probability is constant across the conditional branches, so the only
  //   computation is across conditional branches and any backedge, as required
  //   for any model for Prob.
  // - Prob == 1 produces the maximum possible total frequency for the original
  //   loop body, as described above.  Prob == 0 produces the minimum, 0.
  //   Increasing or decreasing Prob monotonically increases or decreases the
  //   frequency, respectively.  Thus, for every possible frequency, there
  //   exists some Prob that can produce it, and we can easily use bisection to
  //   search the problem space.

  // When iterating for a solution, we stop early if we find probabilities
  // that produce a Freq whose difference from FreqDesired is small
  // (FreqPrec).  Otherwise, we expect to compute a solution at least that
  // accurate (but surely far more accurate).
  const double FreqPrec = 1e-6;

  // Compute the new frequency produced by using Prob throughout CondLatches.
  auto ComputeFreq = [&](double Prob) {
    double ProbReaching = 1;        // p^0
    double FreqOne = IterCounts[0]; // c_0*p^0
    for (unsigned I = 0, E = CondLatches.size(); I < E; ++I) {
      ProbReaching *= Prob;                        // p^(I+1)
      FreqOne += IterCounts[I + 1] * ProbReaching; // c_(I+1)*p^(I+1)
    }
    double ProbReachingBackedge = CompletelyUnroll ? 0 : ProbReaching;
    assert(FreqOne > 0 && "Expected at least one iteration before first latch");
    if (ProbReachingBackedge == 1)
      return std::numeric_limits<double>::infinity();
    return FreqOne / (1 - ProbReachingBackedge);
  };

  // Compute the probability that, used at CondLaches[0] where
  // CondLatches.size() == 1, gets as close as possible to FreqDesired.
  auto ComputeProbForLinear = [&]() {
    // The polynomial is linear (0 = A*p + B), so just solve it.
    double A = IterCounts[1] + (CompletelyUnroll ? 0 : FreqDesired);
    double B = IterCounts[0] - FreqDesired;
    assert(A > 0 && "Expected iterations after last conditional latch");
    double Prob = -B / A;
    // If it computes an invalid Prob, FreqDesired is impossibly low or high.
    // Otherwise, Prob should produce nearly FreqDesired.
    assert((Prob < 0 || Prob > 1 ||
            fabs(ComputeFreq(Prob) - FreqDesired) < FreqPrec) &&
           "Expected accurate frequency when linear case is possible");
    Prob = std::max(Prob, 0.);
    Prob = std::min(Prob, 1.);
    return Prob;
  };

  // Compute the probability that, used throughout CondLatches where
  // CondLatches.size() == 2, gets as close as possible to FreqDesired.
  auto ComputeProbForQuadratic = [&]() {
    // The polynomial is quadratic (0 = A*p^2 + B*p + C), so just solve it.
    double A = IterCounts[2] + (CompletelyUnroll ? 0 : FreqDesired);
    double B = IterCounts[1];
    double C = IterCounts[0] - FreqDesired;
    assert(A > 0 && "Expected iterations after last conditional latch");
    double Prob = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);
    // If it computes an invalid Prob, FreqDesired is impossibly low or high.
    // Otherwise, Prob should produce nearly FreqDesired.
    assert((Prob < 0 || Prob > 1 ||
            fabs(ComputeFreq(Prob) - FreqDesired) < FreqPrec) &&
           "Expected accurate frequency when quadratic case is possible");
    Prob = std::max(Prob, 0.);
    Prob = std::min(Prob, 1.);
    return Prob;
  };

  // Compute the probability required at CondLatches[ComputeIdx] to get as close
  // as possible to FreqDesired without replacing probabilities elsewhere in
  // CondLatches.  Return {Prob, Freq} where 0 <= Prob <= 1 and Freq is the new
  // frequency.
  auto ComputeProb = [&](unsigned ComputeIdx) -> std::pair<double, double> {
    assert(ComputeIdx < CondLatches.size());

    // Accumulate the frequency from before ComputeIdx into FreqBeforeCompute,
    // and accumulate the rest in Freq without yet multiplying the latter by any
    // probability for ComputeIdx (i.e., treat it as 1 for now).
    double ProbReaching = 1;     // p^0
    double Freq = IterCounts[0]; // c_0*p^0
    double FreqBeforeCompute;
    for (unsigned I = 0, E = CondLatches.size(); I < E; ++I) {
      // Get the branch probability for CondLatches[I].
      double Prob;
      if (I == ComputeIdx) {
        FreqBeforeCompute = Freq;
        Freq = 0;
        Prob = 1;
      } else {
        Prob = GetProb(I);
      }
      ProbReaching *= Prob;                     // p^(I+1)
      Freq += IterCounts[I + 1] * ProbReaching; // c_(I+1)*p^(I+1)
    }

    // Compute the required probability, and limit it to a valid probability (0
    // <= p <= 1).  See the Freq formula below for how to derive the ProbCompute
    // formula.
    double ProbReachingBackedge = CompletelyUnroll ? 0 : ProbReaching;
    double ProbComputeNumerator = FreqDesired - FreqBeforeCompute;
    double ProbComputeDenominator = Freq + FreqDesired * ProbReachingBackedge;
    double ProbCompute;
    if (ProbComputeNumerator <= 0) {
      // FreqBeforeCompute has already reached or surpassed FreqDesired, so add
      // no more frequency.  It is possible that ProbComputeDenominator == 0
      // here because some latch probability (maybe the original) was set to
      // zero, so this check avoids setting ProbCompute=1 (in the else if below)
      // and division by zero where the numerator <= 0 (in the else below).
      ProbCompute = 0;
    } else if (ProbComputeDenominator == 0) {
      // Analytically, this case seems impossible.  It would occur if either:
      // - Both Freq and FreqDesired are zero.  But the latter would cause
      //   ProbComputeNumerator < 0, which we catch above, and FreqDesired
      //   should always be >= 1 anyway.
      // - There are no iterations after CondLatches[ComputeIdx], not even via
      //   a backedge, so that both Freq and ProbReachingBackedge are zero.
      //   But iterations should exist after even the last conditional latch.
      // - Some latch probability (maybe the original) was set to zero so that
      //   both Freq and ProbReachingBackedge are zero.  But that should not
      //   have happened because, according to the above ProbComputeNumerator
      //   check, we have not yet reached FreqDesired (which, if the original
      //   latch probability is zero, is just 1 and thus always reached or
      //   surpassed).
      //
      // Numerically, perhaps this case is possible.  We interpret it to mean we
      // need more frequency (ProbComputeNumerator > 0) but have no way to get
      // any (ProbComputeDenominator is analytically too small to distinguish it
      // from 0 in floating point), suggesting infinite probability is needed,
      // but 1 is the maximum valid probability and thus the best we can do.
      //
      // TODO: Cover this case in the test suite if you can.
      ProbCompute = 1;
    } else {
      ProbCompute = ProbComputeNumerator / ProbComputeDenominator;
      ProbCompute = std::max(ProbCompute, 0.);
      ProbCompute = std::min(ProbCompute, 1.);
    }

    // Compute the resulting total frequency.
    if (ProbReachingBackedge * ProbCompute == 1) {
      // Analytically, this case seems impossible.  It requires that there is a
      // backedge and that FreqDesired == infinity so that every conditional
      // latch's probability had to be set to 1.  But FreqDesired == infinity
      // means OriginalLoopProb.isOne(), which we guarded against earlier.
      //
      // Numerically, perhaps this case is possible.  We interpret it to mean
      // that analytically the probability has to be so near 1 that, in floating
      // point, the frequency is computed as infinite.
      //
      // TODO: Cover this case in the test suite if you can.
      Freq = std::numeric_limits<double>::infinity();
    } else {
      assert(FreqBeforeCompute > 0 &&
             "Expected at least one iteration before first latch");
      // In this equation, if we replace the left-hand side with FreqDesired and
      // then solve for ProbCompute, we get the ProbCompute formula above.
      Freq = (FreqBeforeCompute + Freq * ProbCompute) /
             (1 - ProbReachingBackedge * ProbCompute);
    }
    return {ProbCompute, Freq};
  };

  // Determine and set branch weights.
  //
  // Prob < 0 and Prob > 1 cannot be represented as branch weights.  We might
  // compute such a Prob if FreqDesired is impossible (e.g., due to inaccurate
  // profile data) for the maximum trip count we have determined when completely
  // unrolling.  In that case, so just go with whichever is closest.
  if (CondLatches.size() == 1) {
    SetAllProbs(ComputeProbForLinear());
  } else if (CondLatches.size() == 2) {
    SetAllProbs(ComputeProbForQuadratic());
  } else if (!UnrollUniformWeights) {
    // The polynomial is too complex for a simple formula, and the quick and
    // dirty fix has been selected.  Adjust probabilities starting from the
    // first latch, which has the most influence on the total frequency, so
    // starting there should minimize the number of latches that have to be
    // visited.  We do have to iterate because the first latch alone might not
    // be enough.  For example, we might need to set all probabilities to 1 if
    // the frequency is the unroll factor.
    for (unsigned I = 0; I != CondLatches.size(); ++I) {
      double Prob, Freq;
      std::tie(Prob, Freq) = ComputeProb(I);
      SetProb(I, Prob);
      if (fabs(Freq - FreqDesired) < FreqPrec)
        break;
    }
  } else {
    // The polynomial is too complex for a simple formula, and uniform branch
    // weights have been selected, so bisect.
    double ProbMin, ProbMax, ProbPrev;
    auto TryProb = [&](double Prob) {
      ProbPrev = Prob;
      double FreqDelta = ComputeFreq(Prob) - FreqDesired;
      if (fabs(FreqDelta) < FreqPrec)
        return 0;
      if (FreqDelta < 0) {
        ProbMin = Prob;
        return -1;
      }
      ProbMax = Prob;
      return 1;
    };
    // If Prob == 0 is too small and Prob == 1 is too large, bisect between
    // them.  To place a hard upper limit on the search time, stop bisecting
    // when Prob stops changing (ProbDelta) by much (ProbPrec).
    if (TryProb(0.) < 0 && TryProb(1.) > 0) {
      const double ProbPrec = 1e-12;
      double Prob, ProbDelta;
      do {
        Prob = (ProbMin + ProbMax) / 2;
        ProbDelta = Prob - ProbPrev;
      } while (TryProb(Prob) != 0 && fabs(ProbDelta) > ProbPrec);
    }
    SetAllProbs(ProbPrev);
  }

  // FIXME: We have not considered non-latch loop exits:
  // - Their original probabilities are not considered in our calculation of
  //   FreqDesired.
  // - Their probabilities are not considered in our probability model used to
  //   determine new probabilities for remaining conditional branches.
  // - If they are conditional and LoopUnroll converts them to unconditional,
  //   LoopUnroll has proven their original probabilities are incorrect for some
  //   original loop iterations, but that does not cause ProbUpdateRequired to
  //   be set to true.
  //
  // To adjust FreqDesired and our probability model correctly for a non-latch
  // loop exit, we would need to compute the original probability that the exit
  // is reached from the loop header (in contrast, we currently assume that
  // probability is 1 in the case of a latch exit) and the probability that the
  // exit is taken if it is conditional (use the branch's old or new weights for
  // FreqDesired or the probability model, respectively).  Does computing the
  // reaching probability require a CFG traversal, or is there some existing
  // library that can do it?  Prior discussions suggest some such libraries are
  // difficult to use within LoopUnroll:
  // <https://github.com/llvm/llvm-project/pull/164799#issuecomment-3438681519>.
  // For now, we just let our corrected probabilities be less accurate in that
  // scenario.  Alternatively, we could refuse to correct probabilities at all
  // in that scenario, but that seems worse.
}

/// Unroll the given loop by Count. The loop must be in LCSSA form.  Unrolling
/// can only fail when the loop's latch block is not terminated by a conditional
/// branch instruction. However, if the trip count (and multiple) are not known,
/// loop unrolling will mostly produce more code that is no faster.
///
/// If Runtime is true then UnrollLoop will try to insert a prologue or
/// epilogue that ensures the latch has a trip multiple of Count. UnrollLoop
/// will not runtime-unroll the loop if computing the run-time trip count will
/// be expensive and AllowExpensiveTripCount is false.
///
/// The LoopInfo Analysis that is passed will be kept consistent.
///
/// This utility preserves LoopInfo. It will also preserve ScalarEvolution and
/// DominatorTree if they are non-null.
///
/// If RemainderLoop is non-null, it will receive the remainder loop (if
/// required and not fully unrolled).
LoopUnrollResult
llvm::UnrollLoop(Loop *L, UnrollLoopOptions ULO, LoopInfo *LI,
                 ScalarEvolution *SE, DominatorTree *DT, AssumptionCache *AC,
                 const TargetTransformInfo *TTI, OptimizationRemarkEmitter *ORE,
                 bool PreserveLCSSA, Loop **RemainderLoop, AAResults *AA) {
  assert(DT && "DomTree is required");

  if (!L->getLoopPreheader()) {
    LLVM_DEBUG(dbgs() << "  Can't unroll; loop preheader-insertion failed.\n");
    return LoopUnrollResult::Unmodified;
  }

  if (!L->getLoopLatch()) {
    LLVM_DEBUG(dbgs() << "  Can't unroll; loop exit-block-insertion failed.\n");
    return LoopUnrollResult::Unmodified;
  }

  // Loops with indirectbr cannot be cloned.
  if (!L->isSafeToClone()) {
    LLVM_DEBUG(dbgs() << "  Can't unroll; Loop body cannot be cloned.\n");
    return LoopUnrollResult::Unmodified;
  }

  if (L->getHeader()->hasAddressTaken()) {
    // The loop-rotate pass can be helpful to avoid this in many cases.
    LLVM_DEBUG(
        dbgs() << "  Won't unroll loop: address of header block is taken.\n");
    return LoopUnrollResult::Unmodified;
  }

  assert(ULO.Count > 0);

  // All these values should be taken only after peeling because they might have
  // changed.
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Header = L->getHeader();
  BasicBlock *LatchBlock = L->getLoopLatch();
  SmallVector<BasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  std::vector<BasicBlock *> OriginalLoopBlocks = L->getBlocks();

  const unsigned MaxTripCount = SE->getSmallConstantMaxTripCount(L);
  const bool MaxOrZero = SE->isBackedgeTakenCountMaxOrZero(L);
  std::optional<unsigned> OriginalTripCount =
      llvm::getLoopEstimatedTripCount(L);
  BranchProbability OriginalLoopProb = llvm::getLoopProbability(L);

  // Effectively "DCE" unrolled iterations that are beyond the max tripcount
  // and will never be executed.
  if (MaxTripCount && ULO.Count > MaxTripCount)
    ULO.Count = MaxTripCount;

  struct ExitInfo {
    unsigned TripCount;
    unsigned TripMultiple;
    unsigned BreakoutTrip;
    bool ExitOnTrue;
    BasicBlock *FirstExitingBlock = nullptr;
    SmallVector<BasicBlock *> ExitingBlocks;
  };
  DenseMap<BasicBlock *, ExitInfo> ExitInfos;
  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  for (auto *ExitingBlock : ExitingBlocks) {
    // The folding code is not prepared to deal with non-branch instructions
    // right now.
    auto *BI = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
    if (!BI)
      continue;

    ExitInfo &Info = ExitInfos[ExitingBlock];
    Info.TripCount = SE->getSmallConstantTripCount(L, ExitingBlock);
    Info.TripMultiple = SE->getSmallConstantTripMultiple(L, ExitingBlock);
    if (Info.TripCount != 0) {
      Info.BreakoutTrip = Info.TripCount % ULO.Count;
      Info.TripMultiple = 0;
    } else {
      Info.BreakoutTrip = Info.TripMultiple =
          (unsigned)std::gcd(ULO.Count, Info.TripMultiple);
    }
    Info.ExitOnTrue = !L->contains(BI->getSuccessor(0));
    Info.ExitingBlocks.push_back(ExitingBlock);
    LLVM_DEBUG(dbgs() << "  Exiting block %" << ExitingBlock->getName()
                      << ": TripCount=" << Info.TripCount
                      << ", TripMultiple=" << Info.TripMultiple
                      << ", BreakoutTrip=" << Info.BreakoutTrip << "\n");
  }

  // Are we eliminating the loop control altogether?  Note that we can know
  // we're eliminating the backedge without knowing exactly which iteration
  // of the unrolled body exits.
  const bool CompletelyUnroll = ULO.Count == MaxTripCount;

  const bool PreserveOnlyFirst = CompletelyUnroll && MaxOrZero;

  // There's no point in performing runtime unrolling if this unroll count
  // results in a full unroll.
  if (CompletelyUnroll)
    ULO.Runtime = false;

  // Go through all exits of L and see if there are any phi-nodes there. We just
  // conservatively assume that they're inserted to preserve LCSSA form, which
  // means that complete unrolling might break this form. We need to either fix
  // it in-place after the transformation, or entirely rebuild LCSSA. TODO: For
  // now we just recompute LCSSA for the outer loop, but it should be possible
  // to fix it in-place.
  bool NeedToFixLCSSA =
      PreserveLCSSA && CompletelyUnroll &&
      any_of(ExitBlocks,
             [](const BasicBlock *BB) { return isa<PHINode>(BB->begin()); });

  // The current loop unroll pass can unroll loops that have
  // (1) single latch; and
  // (2a) latch is unconditional; or
  // (2b) latch is conditional and is an exiting block
  // FIXME: The implementation can be extended to work with more complicated
  // cases, e.g. loops with multiple latches.
  BranchInst *LatchBI = dyn_cast<BranchInst>(LatchBlock->getTerminator());

  // A conditional branch which exits the loop, which can be optimized to an
  // unconditional branch in the unrolled loop in some cases.
  bool LatchIsExiting = L->isLoopExiting(LatchBlock);
  if (!LatchBI || (LatchBI->isConditional() && !LatchIsExiting)) {
    LLVM_DEBUG(
        dbgs() << "Can't unroll; a conditional latch must exit the loop");
    return LoopUnrollResult::Unmodified;
  }

  assert((!ULO.Runtime || canHaveUnrollRemainder(L)) &&
         "Can't runtime unroll if loop contains a convergent operation.");

  bool EpilogProfitability =
      UnrollRuntimeEpilog.getNumOccurrences() ? UnrollRuntimeEpilog
                                              : isEpilogProfitable(L);

  if (ULO.Runtime &&
      !UnrollRuntimeLoopRemainder(
          L, ULO.Count, ULO.AllowExpensiveTripCount, EpilogProfitability,
          ULO.UnrollRemainder, ULO.ForgetAllSCEV, LI, SE, DT, AC, TTI,
          PreserveLCSSA, ULO.SCEVExpansionBudget, ULO.RuntimeUnrollMultiExit,
          RemainderLoop, OriginalTripCount, OriginalLoopProb)) {
    if (ULO.Force)
      ULO.Runtime = false;
    else {
      LLVM_DEBUG(dbgs() << "Won't unroll; remainder loop could not be "
                           "generated when assuming runtime trip count\n");
      return LoopUnrollResult::Unmodified;
    }
  }

  using namespace ore;
  // Report the unrolling decision.
  if (CompletelyUnroll) {
    LLVM_DEBUG(dbgs() << "COMPLETELY UNROLLING loop %" << Header->getName()
                      << " with trip count " << ULO.Count << "!\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemark(DEBUG_TYPE, "FullyUnrolled", L->getStartLoc(),
                                  L->getHeader())
               << "completely unrolled loop with "
               << NV("UnrollCount", ULO.Count) << " iterations";
      });
  } else {
    LLVM_DEBUG(dbgs() << "UNROLLING loop %" << Header->getName() << " by "
                      << ULO.Count);
    if (ULO.Runtime)
      LLVM_DEBUG(dbgs() << " with run-time trip count");
    LLVM_DEBUG(dbgs() << "!\n");

    if (ORE)
      ORE->emit([&]() {
        OptimizationRemark Diag(DEBUG_TYPE, "PartialUnrolled", L->getStartLoc(),
                                L->getHeader());
        Diag << "unrolled loop by a factor of " << NV("UnrollCount", ULO.Count);
        if (ULO.Runtime)
          Diag << " with run-time trip count";
        return Diag;
      });
  }

  // We are going to make changes to this loop. SCEV may be keeping cached info
  // about it, in particular about backedge taken count. The changes we make
  // are guaranteed to invalidate this information for our loop. It is tempting
  // to only invalidate the loop being unrolled, but it is incorrect as long as
  // all exiting branches from all inner loops have impact on the outer loops,
  // and if something changes inside them then any of outer loops may also
  // change. When we forget outermost loop, we also forget all contained loops
  // and this is what we need here.
  if (SE) {
    if (ULO.ForgetAllSCEV)
      SE->forgetAllLoops();
    else {
      SE->forgetTopmostLoop(L);
      SE->forgetBlockAndLoopDispositions();
    }
  }

  if (!LatchIsExiting)
    ++NumUnrolledNotLatch;

  // For the first iteration of the loop, we should use the precloned values for
  // PHI nodes.  Insert associations now.
  ValueToValueMapTy LastValueMap;
  std::vector<PHINode*> OrigPHINode;
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    OrigPHINode.push_back(cast<PHINode>(I));
  }

  // Collect phi nodes for reductions for which we can introduce multiple
  // parallel reduction phis and compute the final reduction result after the
  // loop. This requires a single exit block after unrolling. This is ensured by
  // restricting to single-block loops where the unrolled iterations are known
  // to not exit.
  DenseMap<PHINode *, RecurrenceDescriptor> Reductions;
  bool CanAddAdditionalAccumulators =
      (UnrollAddParallelReductions.getNumOccurrences() > 0
           ? UnrollAddParallelReductions
           : ULO.AddAdditionalAccumulators) &&
      !CompletelyUnroll && L->getNumBlocks() == 1 &&
      (ULO.Runtime ||
       (ExitInfos.contains(Header) && ((ExitInfos[Header].TripCount != 0 &&
                                        ExitInfos[Header].BreakoutTrip == 0))));

  // Limit parallelizing reductions to unroll counts of 4 or less for now.
  // TODO: The number of parallel reductions should depend on the number of
  // execution units. We also don't have to add a parallel reduction phi per
  // unrolled iteration, but could for example add a parallel phi for every 2
  // unrolled iterations.
  if (CanAddAdditionalAccumulators && ULO.Count <= 4) {
    for (PHINode &Phi : Header->phis()) {
      auto RdxDesc = canParallelizeReductionWhenUnrolling(Phi, L, SE);
      if (!RdxDesc)
        continue;

      // Only handle duplicate phis for a single reduction for now.
      // TODO: Handle any number of reductions
      if (!Reductions.empty())
        continue;

      Reductions[&Phi] = *RdxDesc;
    }
  }

  std::vector<BasicBlock *> Headers;
  std::vector<BasicBlock *> Latches;
  Headers.push_back(Header);
  Latches.push_back(LatchBlock);

  // The current on-the-fly SSA update requires blocks to be processed in
  // reverse postorder so that LastValueMap contains the correct value at each
  // exit.
  LoopBlocksDFS DFS(L);
  DFS.perform(LI);

  // Stash the DFS iterators before adding blocks to the loop.
  LoopBlocksDFS::RPOIterator BlockBegin = DFS.beginRPO();
  LoopBlocksDFS::RPOIterator BlockEnd = DFS.endRPO();

  std::vector<BasicBlock*> UnrolledLoopBlocks = L->getBlocks();

  // Loop Unrolling might create new loops. While we do preserve LoopInfo, we
  // might break loop-simplified form for these loops (as they, e.g., would
  // share the same exit blocks). We'll keep track of loops for which we can
  // break this so that later we can re-simplify them.
  SmallSetVector<Loop *, 4> LoopsToSimplify;
  LoopsToSimplify.insert_range(*L);

  // When a FSDiscriminator is enabled, we don't need to add the multiply
  // factors to the discriminators.
  if (Header->getParent()->shouldEmitDebugInfoForProfiling() &&
      !EnableFSDiscriminator)
    for (BasicBlock *BB : L->getBlocks())
      for (Instruction &I : *BB)
        if (!I.isDebugOrPseudoInst())
          if (const DILocation *DIL = I.getDebugLoc()) {
            auto NewDIL = DIL->cloneByMultiplyingDuplicationFactor(ULO.Count);
            if (NewDIL)
              I.setDebugLoc(*NewDIL);
            else
              LLVM_DEBUG(dbgs()
                         << "Failed to create new discriminator: "
                         << DIL->getFilename() << " Line: " << DIL->getLine());
          }

  // Identify what noalias metadata is inside the loop: if it is inside the
  // loop, the associated metadata must be cloned for each iteration.
  SmallVector<MDNode *, 6> LoopLocalNoAliasDeclScopes;
  identifyNoAliasScopesToClone(L->getBlocks(), LoopLocalNoAliasDeclScopes);

  // We place the unrolled iterations immediately after the original loop
  // latch.  This is a reasonable default placement if we don't have block
  // frequencies, and if we do, well the layout will be adjusted later.
  auto BlockInsertPt = std::next(LatchBlock->getIterator());
  SmallVector<Instruction *> PartialReductions;
  for (unsigned It = 1; It != ULO.Count; ++It) {
    SmallVector<BasicBlock *, 8> NewBlocks;
    SmallDenseMap<const Loop *, Loop *, 4> NewLoops;
    NewLoops[L] = L;

    for (LoopBlocksDFS::RPOIterator BB = BlockBegin; BB != BlockEnd; ++BB) {
      ValueToValueMapTy VMap;
      BasicBlock *New = CloneBasicBlock(*BB, VMap, "." + Twine(It));
      Header->getParent()->insert(BlockInsertPt, New);

      assert((*BB != Header || LI->getLoopFor(*BB) == L) &&
             "Header should not be in a sub-loop");
      // Tell LI about New.
      const Loop *OldLoop = addClonedBlockToLoopInfo(*BB, New, LI, NewLoops);
      if (OldLoop)
        LoopsToSimplify.insert(NewLoops[OldLoop]);

      if (*BB == Header) {
        // Loop over all of the PHI nodes in the block, changing them to use
        // the incoming values from the previous block.
        for (PHINode *OrigPHI : OrigPHINode) {
          PHINode *NewPHI = cast<PHINode>(VMap[OrigPHI]);
          Value *InVal = NewPHI->getIncomingValueForBlock(LatchBlock);

          // Use cloned phis as parallel phis for partial reductions, which will
          // get combined to the final reduction result after the loop.
          if (Reductions.contains(OrigPHI)) {
            // Collect partial  reduction results.
            if (PartialReductions.empty())
              PartialReductions.push_back(cast<Instruction>(InVal));
            PartialReductions.push_back(cast<Instruction>(VMap[InVal]));

            // Update the start value for the cloned phis to use the identity
            // value for the reduction.
            const RecurrenceDescriptor &RdxDesc = Reductions[OrigPHI];
            NewPHI->setIncomingValueForBlock(
                L->getLoopPreheader(),
                getRecurrenceIdentity(RdxDesc.getRecurrenceKind(),
                                      OrigPHI->getType(),
                                      RdxDesc.getFastMathFlags()));

            // Update NewPHI to use the cloned value for the iteration and move
            // to header.
            NewPHI->replaceUsesOfWith(InVal, VMap[InVal]);
            NewPHI->moveBefore(OrigPHI->getIterator());
            continue;
          }

          if (Instruction *InValI = dyn_cast<Instruction>(InVal))
            if (It > 1 && L->contains(InValI))
              InVal = LastValueMap[InValI];
          VMap[OrigPHI] = InVal;
          NewPHI->eraseFromParent();
        }

        // Eliminate copies of the loop heart intrinsic, if any.
        if (ULO.Heart) {
          auto it = VMap.find(ULO.Heart);
          assert(it != VMap.end());
          Instruction *heartCopy = cast<Instruction>(it->second);
          heartCopy->eraseFromParent();
          VMap.erase(it);
        }
      }

      // Remap source location atom instance. Do this now, rather than
      // when we remap instructions, because remap is called once we've
      // cloned all blocks (all the clones would get the same atom
      // number).
      if (!VMap.AtomMap.empty())
        for (Instruction &I : *New)
          RemapSourceAtom(&I, VMap);

      // Update our running map of newest clones
      LastValueMap[*BB] = New;
      for (ValueToValueMapTy::iterator VI = VMap.begin(), VE = VMap.end();
           VI != VE; ++VI)
        LastValueMap[VI->first] = VI->second;

      // Add phi entries for newly created values to all exit blocks.
      for (BasicBlock *Succ : successors(*BB)) {
        if (L->contains(Succ))
          continue;
        for (PHINode &PHI : Succ->phis()) {
          Value *Incoming = PHI.getIncomingValueForBlock(*BB);
          ValueToValueMapTy::iterator It = LastValueMap.find(Incoming);
          if (It != LastValueMap.end())
            Incoming = It->second;
          PHI.addIncoming(Incoming, New);
          SE->forgetLcssaPhiWithNewPredecessor(L, &PHI);
        }
      }
      // Keep track of new headers and latches as we create them, so that
      // we can insert the proper branches later.
      if (*BB == Header)
        Headers.push_back(New);
      if (*BB == LatchBlock)
        Latches.push_back(New);

      // Keep track of the exiting block and its successor block contained in
      // the loop for the current iteration.
      auto ExitInfoIt = ExitInfos.find(*BB);
      if (ExitInfoIt != ExitInfos.end())
        ExitInfoIt->second.ExitingBlocks.push_back(New);

      NewBlocks.push_back(New);
      UnrolledLoopBlocks.push_back(New);

      // Update DomTree: since we just copy the loop body, and each copy has a
      // dedicated entry block (copy of the header block), this header's copy
      // dominates all copied blocks. That means, dominance relations in the
      // copied body are the same as in the original body.
      if (*BB == Header)
        DT->addNewBlock(New, Latches[It - 1]);
      else {
        auto BBDomNode = DT->getNode(*BB);
        auto BBIDom = BBDomNode->getIDom();
        BasicBlock *OriginalBBIDom = BBIDom->getBlock();
        DT->addNewBlock(
            New, cast<BasicBlock>(LastValueMap[cast<Value>(OriginalBBIDom)]));
      }
    }

    // Remap all instructions in the most recent iteration.
    // Key Instructions: Nothing to do - we've already remapped the atoms.
    remapInstructionsInBlocks(NewBlocks, LastValueMap);
    for (BasicBlock *NewBlock : NewBlocks)
      for (Instruction &I : *NewBlock)
        if (auto *II = dyn_cast<AssumeInst>(&I))
          AC->registerAssumption(II);

    {
      // Identify what other metadata depends on the cloned version. After
      // cloning, replace the metadata with the corrected version for both
      // memory instructions and noalias intrinsics.
      std::string ext = (Twine("It") + Twine(It)).str();
      cloneAndAdaptNoAliasScopes(LoopLocalNoAliasDeclScopes, NewBlocks,
                                 Header->getContext(), ext);
    }
  }

  // Loop over the PHI nodes in the original block, setting incoming values.
  for (PHINode *PN : OrigPHINode) {
    if (CompletelyUnroll) {
      PN->replaceAllUsesWith(PN->getIncomingValueForBlock(Preheader));
      PN->eraseFromParent();
    } else if (ULO.Count > 1) {
      if (Reductions.contains(PN))
        continue;

      Value *InVal = PN->removeIncomingValue(LatchBlock, false);
      // If this value was defined in the loop, take the value defined by the
      // last iteration of the loop.
      if (Instruction *InValI = dyn_cast<Instruction>(InVal)) {
        if (L->contains(InValI))
          InVal = LastValueMap[InVal];
      }
      assert(Latches.back() == LastValueMap[LatchBlock] && "bad last latch");
      PN->addIncoming(InVal, Latches.back());
    }
  }

  // Connect latches of the unrolled iterations to the headers of the next
  // iteration. Currently they point to the header of the same iteration.
  for (unsigned i = 0, e = Latches.size(); i != e; ++i) {
    unsigned j = (i + 1) % e;
    Latches[i]->getTerminator()->replaceSuccessorWith(Headers[i], Headers[j]);
  }

  // Remove loop metadata copied from the original loop latch to branches that
  // are no longer latches.
  for (unsigned I = 0, E = Latches.size() - (CompletelyUnroll ? 0 : 1); I < E;
       ++I)
    Latches[I]->getTerminator()->setMetadata(LLVMContext::MD_loop, nullptr);

  // Update dominators of blocks we might reach through exits.
  // Immediate dominator of such block might change, because we add more
  // routes which can lead to the exit: we can now reach it from the copied
  // iterations too.
  if (ULO.Count > 1) {
    for (auto *BB : OriginalLoopBlocks) {
      auto *BBDomNode = DT->getNode(BB);
      SmallVector<BasicBlock *, 16> ChildrenToUpdate;
      for (auto *ChildDomNode : BBDomNode->children()) {
        auto *ChildBB = ChildDomNode->getBlock();
        if (!L->contains(ChildBB))
          ChildrenToUpdate.push_back(ChildBB);
      }
      // The new idom of the block will be the nearest common dominator
      // of all copies of the previous idom. This is equivalent to the
      // nearest common dominator of the previous idom and the first latch,
      // which dominates all copies of the previous idom.
      BasicBlock *NewIDom = DT->findNearestCommonDominator(BB, LatchBlock);
      for (auto *ChildBB : ChildrenToUpdate)
        DT->changeImmediateDominator(ChildBB, NewIDom);
    }
  }

  assert(!UnrollVerifyDomtree ||
         DT->verify(DominatorTree::VerificationLevel::Fast));

  SmallVector<DominatorTree::UpdateType> DTUpdates;
  auto SetDest = [&](BasicBlock *Src, bool WillExit, bool ExitOnTrue) {
    auto *Term = cast<BranchInst>(Src->getTerminator());
    const unsigned Idx = ExitOnTrue ^ WillExit;
    BasicBlock *Dest = Term->getSuccessor(Idx);
    BasicBlock *DeadSucc = Term->getSuccessor(1-Idx);

    // Remove predecessors from all non-Dest successors.
    DeadSucc->removePredecessor(Src, /* KeepOneInputPHIs */ true);

    // Replace the conditional branch with an unconditional one.
    auto *BI = BranchInst::Create(Dest, Term->getIterator());
    BI->setDebugLoc(Term->getDebugLoc());
    Term->eraseFromParent();

    DTUpdates.emplace_back(DominatorTree::Delete, Src, DeadSucc);
  };

  auto WillExit = [&](const ExitInfo &Info, unsigned i, unsigned j,
                      bool IsLatch) -> std::optional<bool> {
    if (CompletelyUnroll) {
      if (PreserveOnlyFirst) {
        if (i == 0)
          return std::nullopt;
        return j == 0;
      }
      // Complete (but possibly inexact) unrolling
      if (j == 0)
        return true;
      if (Info.TripCount && j != Info.TripCount)
        return false;
      return std::nullopt;
    }

    if (ULO.Runtime) {
      // If runtime unrolling inserts a prologue, information about non-latch
      // exits may be stale.
      if (IsLatch && j != 0)
        return false;
      return std::nullopt;
    }

    if (j != Info.BreakoutTrip &&
        (Info.TripMultiple == 0 || j % Info.TripMultiple != 0)) {
      // If we know the trip count or a multiple of it, we can safely use an
      // unconditional branch for some iterations.
      return false;
    }
    return std::nullopt;
  };

  // Fold branches for iterations where we know that they will exit or not
  // exit.  In the case of an iteration's latch, if we thus find
  // *OriginalLoopProb is incorrect, set ProbUpdateRequired to true.
  bool ProbUpdateRequired = false;
  for (auto &Pair : ExitInfos) {
    ExitInfo &Info = Pair.second;
    for (unsigned i = 0, e = Info.ExitingBlocks.size(); i != e; ++i) {
      // The branch destination.
      unsigned j = (i + 1) % e;
      bool IsLatch = Pair.first == LatchBlock;
      std::optional<bool> KnownWillExit = WillExit(Info, i, j, IsLatch);
      if (!KnownWillExit) {
        if (!Info.FirstExitingBlock)
          Info.FirstExitingBlock = Info.ExitingBlocks[i];
        continue;
      }

      // We don't fold known-exiting branches for non-latch exits here,
      // because this ensures that both all loop blocks and all exit blocks
      // remain reachable in the CFG.
      // TODO: We could fold these branches, but it would require much more
      // sophisticated updates to LoopInfo.
      if (*KnownWillExit && !IsLatch) {
        if (!Info.FirstExitingBlock)
          Info.FirstExitingBlock = Info.ExitingBlocks[i];
        continue;
      }

      // For a latch, record any OriginalLoopProb contradiction.
      if (!OriginalLoopProb.isUnknown() && IsLatch) {
        BranchProbability ActualProb = *KnownWillExit
                                           ? BranchProbability::getZero()
                                           : BranchProbability::getOne();
        ProbUpdateRequired |= OriginalLoopProb != ActualProb;
      }

      SetDest(Info.ExitingBlocks[i], *KnownWillExit, Info.ExitOnTrue);
    }
  }

  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  DomTreeUpdater *DTUToUse = &DTU;
  if (ExitingBlocks.size() == 1 && ExitInfos.size() == 1) {
    // Manually update the DT if there's a single exiting node. In that case
    // there's a single exit node and it is sufficient to update the nodes
    // immediately dominated by the original exiting block. They will become
    // dominated by the first exiting block that leaves the loop after
    // unrolling. Note that the CFG inside the loop does not change, so there's
    // no need to update the DT inside the unrolled loop.
    DTUToUse = nullptr;
    auto &[OriginalExit, Info] = *ExitInfos.begin();
    if (!Info.FirstExitingBlock)
      Info.FirstExitingBlock = Info.ExitingBlocks.back();
    for (auto *C : to_vector(DT->getNode(OriginalExit)->children())) {
      if (L->contains(C->getBlock()))
        continue;
      C->setIDom(DT->getNode(Info.FirstExitingBlock));
    }
  } else {
    DTU.applyUpdates(DTUpdates);
  }

  // When completely unrolling, the last latch becomes unreachable.
  if (!LatchIsExiting && CompletelyUnroll) {
    // There is no need to update the DT here, because there must be a unique
    // latch. Hence if the latch is not exiting it must directly branch back to
    // the original loop header and does not dominate any nodes.
    assert(LatchBlock->getSingleSuccessor() && "Loop with multiple latches?");
    changeToUnreachable(Latches.back()->getTerminator(), PreserveLCSSA);
  }

  // After merging adjacent blocks in Latches below:
  // - CondLatches will list the blocks from Latches that are still terminated
  //   with conditional branches.
  // - For 1 <= I < CondLatches.size(), IterCounts[I] will store the number of
  //   the original loop iterations through which control flows from
  //   CondLatches[I-1] to CondLatches[I].
  // - For I == 0 or I == CondLatches.size(), IterCounts[I] will store the
  //   number of the original loop iterations through which control can flow
  //   before CondLatches.front() or after CondLatches.back(), respectively,
  //   without taking the unrolled loop's backedge, if any.
  // - CondLatchNexts[I] will store the CondLatches[I] branch target for the
  //   next of the original loop's iterations (as opposed to the exit target).
  assert(ULO.Count == Latches.size() &&
         "Expected one latch block per unrolled iteration");
  std::vector<unsigned> IterCounts(1, 0);
  std::vector<BasicBlock *> CondLatches;
  std::vector<BasicBlock *> CondLatchNexts;
  IterCounts.reserve(Latches.size() + 1);
  CondLatches.reserve(Latches.size());
  CondLatchNexts.reserve(Latches.size());

  // Merge adjacent basic blocks, if possible.
  for (unsigned I = 0, E = Latches.size(); I < E; ++I) {
    ++IterCounts.back();
    BasicBlock *Latch = Latches[I];
    BranchInst *Term = dyn_cast<BranchInst>(Latch->getTerminator());
    assert((Term ||
            (CompletelyUnroll && !LatchIsExiting && Latch == Latches.back())) &&
           "Need a branch as terminator, except when fully unrolling with "
           "unconditional latch");
    if (!Term)
      continue;
    if (Term->isUnconditional()) {
      BasicBlock *Dest = Term->getSuccessor(0);
      BasicBlock *Fold = Dest->getUniquePredecessor();
      if (MergeBlockIntoPredecessor(Dest, /*DTU=*/DTUToUse, LI,
                                    /*MSSAU=*/nullptr, /*MemDep=*/nullptr,
                                    /*PredecessorWithTwoSuccessors=*/false,
                                    DTUToUse ? nullptr : DT)) {
        // Dest has been folded into Fold. Update our worklists accordingly.
        llvm::replace(Latches, Dest, Fold);
        llvm::erase(UnrolledLoopBlocks, Dest);
      }
    } else {
      IterCounts.push_back(0);
      CondLatches.push_back(Latch);
      CondLatchNexts.push_back(Headers[(I + 1) % E]);
    }
  }

  // Fix probabilities we contradicted above.
  if (ProbUpdateRequired) {
    fixProbContradiction(ULO, OriginalLoopProb, CompletelyUnroll, IterCounts,
                         CondLatches, CondLatchNexts);
  }

  // If there are partial reductions, create code in the exit block to compute
  // the final result and update users of the final result.
  if (!PartialReductions.empty()) {
    BasicBlock *ExitBlock = L->getExitBlock();
    assert(ExitBlock &&
           "Can only introduce parallel reduction phis with single exit block");
    assert(Reductions.size() == 1 &&
           "currently only a single reduction is supported");
    Value *FinalRdxValue = PartialReductions.back();
    Value *RdxResult = nullptr;
    for (PHINode &Phi : ExitBlock->phis()) {
      if (Phi.getIncomingValueForBlock(L->getLoopLatch()) != FinalRdxValue)
        continue;
      if (!RdxResult) {
        RdxResult = PartialReductions.front();
        IRBuilder Builder(ExitBlock, ExitBlock->getFirstNonPHIIt());
        Builder.setFastMathFlags(Reductions.begin()->second.getFastMathFlags());
        RecurKind RK = Reductions.begin()->second.getRecurrenceKind();
        for (Instruction *RdxPart : drop_begin(PartialReductions)) {
          RdxResult = Builder.CreateBinOp(
              (Instruction::BinaryOps)RecurrenceDescriptor::getOpcode(RK),
              RdxPart, RdxResult, "bin.rdx");
        }
        NeedToFixLCSSA = true;
        for (Instruction *RdxPart : PartialReductions)
          RdxPart->dropPoisonGeneratingFlags();
      }

      Phi.replaceAllUsesWith(RdxResult);
    }
  }

  if (DTUToUse) {
    // Apply updates to the DomTree.
    DT = &DTU.getDomTree();
  }
  assert(!UnrollVerifyDomtree ||
         DT->verify(DominatorTree::VerificationLevel::Fast));

  // At this point, the code is well formed.  We now simplify the unrolled loop,
  // doing constant propagation and dead code elimination as we go.
  simplifyLoopAfterUnroll(L, !CompletelyUnroll && ULO.Count > 1, LI, SE, DT, AC,
                          TTI, AA);

  NumCompletelyUnrolled += CompletelyUnroll;
  ++NumUnrolled;

  Loop *OuterL = L->getParentLoop();
  // Update LoopInfo if the loop is completely removed.
  if (CompletelyUnroll) {
    LI->erase(L);
    // We shouldn't try to use `L` anymore.
    L = nullptr;
  } else {
    // Update metadata for the loop's branch weights and estimated trip count:
    // - If ULO.Runtime, UnrollRuntimeLoopRemainder sets the guard branch
    //   weights, latch branch weights, and estimated trip count of the
    //   remainder loop it creates.  It also sets the branch weights for the
    //   unrolled loop guard it creates.  The branch weights for the unrolled
    //   loop latch are adjusted below.  FIXME: Handle prologue loops.
    // - Otherwise, if unrolled loop iteration latches become unconditional,
    //   branch weights are adjusted by the fixProbContradiction call above.
    // - Otherwise, the original loop's branch weights are correct for the
    //   unrolled loop, so do not adjust them.
    // - In all cases, the unrolled loop's estimated trip count is set below.
    //
    // As an example of the last case, consider what happens if the unroll count
    // is 4 for a loop with an estimated trip count of 10 when we do not create
    // a remainder loop and all iterations' latches remain conditional.  Each
    // unrolled iteration's latch still has the same probability of exiting the
    // loop as it did when in the original loop, and thus it should still have
    // the same branch weights.  Each unrolled iteration's non-zero probability
    // of exiting already appropriately reduces the probability of reaching the
    // remaining iterations just as it did in the original loop.  Trying to also
    // adjust the branch weights of the final unrolled iteration's latch (i.e.,
    // the backedge for the unrolled loop as a whole) to reflect its new trip
    // count of 3 will erroneously further reduce its block frequencies.
    // However, in case an analysis later needs to estimate the trip count of
    // the unrolled loop as a whole without considering the branch weights for
    // each unrolled iteration's latch within it, we store the new trip count as
    // separate metadata.
    if (!OriginalLoopProb.isUnknown() && ULO.Runtime && EpilogProfitability) {
      assert((CondLatches.size() == 1 &&
              (ProbUpdateRequired || OriginalLoopProb.isOne())) &&
             "Expected ULO.Runtime to give unrolled loop 1 conditional latch, "
             "the backedge, requiring a probability update unless infinite");
      // Where p is always the probability of executing at least 1 more
      // iteration, the probability for at least n more iterations is p^n.
      setLoopProbability(L, OriginalLoopProb.pow(ULO.Count));
    }
    if (OriginalTripCount) {
      unsigned NewTripCount = *OriginalTripCount / ULO.Count;
      if (!ULO.Runtime && *OriginalTripCount % ULO.Count)
        ++NewTripCount;
      setLoopEstimatedTripCount(L, NewTripCount);
    }
  }

  // LoopInfo should not be valid, confirm that.
  if (UnrollVerifyLoopInfo)
    LI->verify(*DT);

  // After complete unrolling most of the blocks should be contained in OuterL.
  // However, some of them might happen to be out of OuterL (e.g. if they
  // precede a loop exit). In this case we might need to insert PHI nodes in
  // order to preserve LCSSA form.
  // We don't need to check this if we already know that we need to fix LCSSA
  // form.
  // TODO: For now we just recompute LCSSA for the outer loop in this case, but
  // it should be possible to fix it in-place.
  if (PreserveLCSSA && OuterL && CompletelyUnroll && !NeedToFixLCSSA)
    NeedToFixLCSSA |= ::needToInsertPhisForLCSSA(OuterL, UnrolledLoopBlocks, LI);

  // Make sure that loop-simplify form is preserved. We want to simplify
  // at least one layer outside of the loop that was unrolled so that any
  // changes to the parent loop exposed by the unrolling are considered.
  if (OuterL) {
    // OuterL includes all loops for which we can break loop-simplify, so
    // it's sufficient to simplify only it (it'll recursively simplify inner
    // loops too).
    if (NeedToFixLCSSA) {
      // LCSSA must be performed on the outermost affected loop. The unrolled
      // loop's last loop latch is guaranteed to be in the outermost loop
      // after LoopInfo's been updated by LoopInfo::erase.
      Loop *LatchLoop = LI->getLoopFor(Latches.back());
      Loop *FixLCSSALoop = OuterL;
      if (!FixLCSSALoop->contains(LatchLoop))
        while (FixLCSSALoop->getParentLoop() != LatchLoop)
          FixLCSSALoop = FixLCSSALoop->getParentLoop();

      formLCSSARecursively(*FixLCSSALoop, *DT, LI, SE);
    } else if (PreserveLCSSA) {
      assert(OuterL->isLCSSAForm(*DT) &&
             "Loops should be in LCSSA form after loop-unroll.");
    }

    // TODO: That potentially might be compile-time expensive. We should try
    // to fix the loop-simplified form incrementally.
    simplifyLoop(OuterL, DT, LI, SE, AC, nullptr, PreserveLCSSA);
  } else {
    // Simplify loops for which we might've broken loop-simplify form.
    for (Loop *SubLoop : LoopsToSimplify)
      simplifyLoop(SubLoop, DT, LI, SE, AC, nullptr, PreserveLCSSA);
  }

  return CompletelyUnroll ? LoopUnrollResult::FullyUnrolled
                          : LoopUnrollResult::PartiallyUnrolled;
}

/// Given an llvm.loop loop id metadata node, returns the loop hint metadata
/// node with the given name (for example, "llvm.loop.unroll.count"). If no
/// such metadata node exists, then nullptr is returned.
MDNode *llvm::GetUnrollMetadata(MDNode *LoopID, StringRef Name) {
  // First operand should refer to the loop id itself.
  assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

  for (const MDOperand &MDO : llvm::drop_begin(LoopID->operands())) {
    MDNode *MD = dyn_cast<MDNode>(MDO);
    if (!MD)
      continue;

    MDString *S = dyn_cast<MDString>(MD->getOperand(0));
    if (!S)
      continue;

    if (Name == S->getString())
      return MD;
  }
  return nullptr;
}

std::optional<RecurrenceDescriptor>
llvm::canParallelizeReductionWhenUnrolling(PHINode &Phi, Loop *L,
                                           ScalarEvolution *SE) {
  RecurrenceDescriptor RdxDesc;
  if (!RecurrenceDescriptor::isReductionPHI(&Phi, L, RdxDesc,
                                            /*DemandedBits=*/nullptr,
                                            /*AC=*/nullptr, /*DT=*/nullptr, SE))
    return std::nullopt;
  if (RdxDesc.hasUsesOutsideReductionChain())
    return std::nullopt;
  RecurKind RK = RdxDesc.getRecurrenceKind();
  // Skip unsupported reductions.
  // TODO: Handle additional reductions, including FP and min-max
  // reductions.
  if (RecurrenceDescriptor::isAnyOfRecurrenceKind(RK) ||
      RecurrenceDescriptor::isFindRecurrenceKind(RK) ||
      RecurrenceDescriptor::isMinMaxRecurrenceKind(RK))
    return std::nullopt;

  if (RdxDesc.hasExactFPMath())
    return std::nullopt;

  if (RdxDesc.IntermediateStore)
    return std::nullopt;

  // Don't unroll reductions with constant ops; those can be folded to a
  // single induction update.
  if (any_of(cast<Instruction>(Phi.getIncomingValueForBlock(L->getLoopLatch()))
                 ->operands(),
             IsaPred<Constant>))
    return std::nullopt;

  BasicBlock *Latch = L->getLoopLatch();
  if (!Latch ||
      !is_contained(
          cast<Instruction>(Phi.getIncomingValueForBlock(Latch))->operands(),
          &Phi))
    return std::nullopt;

  return RdxDesc;
}
