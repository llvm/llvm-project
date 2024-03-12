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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
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
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <assert.h>
#include <numeric>
#include <type_traits>
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

static cl::opt<bool>
UnrollSimplifyReductions("unroll-simplify-reductions", cl::init(true),
                         cl::Hidden, cl::desc("Try to simplify reductions "
                                              "after unrolling a loop."));


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

/// This function tries to break apart simple reduction loops like the one
/// below:
///
/// loop:
///   PN = PHI [SUM2, loop], ...
///   X = ...
///   SUM1 = ADD (X, PN)
///   Y = ...
///   SUM2 = ADD (Y, SUM1)
///   br loop
///
/// into independent sums of the form:
///
/// loop:
///   PN1 = PHI [SUM1, loop], ...
///   PN2 = PHI [SUM2, loop], ...
///   X = ...
///   SUM1 = ADD (X, PN1)
///   Y = ...
///   SUM2 = ADD (Y, PN2)
///   <Reductions>
///   br loop
///
/// where <Reductions> are new instructions inserted to compute the final
/// values of the reduction from the partial sums we introduced, in this case:
///
/// <Reductions> =
///   PN.red = ADD (PN1, PN2)
///   SUM1.red = ADD (SUM1, PN2)
///   SUM2.red = ADD (SUM1, SUM2)
///
/// In practice in most cases only one or two of the reduced values are
/// required outside the loop so most of the reduction instructions do not
/// need to be added into the loop. Moreover, these instructions can be sunk
/// from the loop which happens in later passes.
///
/// This is a very common pattern in unrolled loops that compute dot products
/// (for example) and breaking apart the reduction chains can help greatly with
/// vectorisation.
static bool trySimplifyReductions(Instruction &I) {
  // Check if I is a PHINode (potentially the start of a reduction chain).
  // Note: For simplicity we only consider loops that consists of a single
  // basic block that branches to itself.
  BasicBlock *BB = I.getParent();
  PHINode *PN = dyn_cast<PHINode>(&I);
  if (!PN || PN->getBasicBlockIndex(BB) == -1)
    return false;

  // Attempt to construct a list of instructions that are chained together
  // (i.e. that perform a reduction).
  SmallVector<BinaryOperator *, 16> Ops;
  for (Instruction *Cur = PN, *Next = nullptr; /* true */;
       Cur = Next, Next = nullptr) {
    // Try to find the next element in the reduction chain.
    for (auto *U : Cur->users()) {
      auto *Candidate = dyn_cast<Instruction>(U);
      if (Candidate && Candidate->getParent() == BB) {
        // If we've already found a candidate element for the chain and we find
        // *another* candidate we bail out as this means the intermediate
        // values of the reduction are needed within the loop, and so there is
        // no point in breaking the reduction apart.
        if (Next)
          return false;
        Next = Candidate;
      }
    }
    // If we've reached the start, i.e. the next element in the chain would be
    // the PN we started with, we are done.
    if (Next == PN)
      break;
    // Else, check if we found a candidate at all and if so if it is a binary
    // operator.
    if (!Next || !isa<BinaryOperator>(Next))
      return false;
    // If everything checks out, add the new element to the chain.
    Ops.push_back(cast<BinaryOperator>(Next));
  }

  // Ensure the reduction comprises at least two instructions, otherwise this
  // is a trivial reduction of a single element that doesn't need to be
  // simplified.
  if (Ops.size() < 2)
    return false;

  LLVM_DEBUG(dbgs() << "Candidate reduction of length " << Ops.size()
                    << " found at " << I << ".\n");

  // Ensure all instructions perform the same operation and that the operation
  // is associative and commutative so that we can break the chain apart and
  // reassociate the Ops.
  Instruction::BinaryOps const Opcode = Ops[0]->getOpcode();
  for (auto const *Op : Ops)
    if (Op->getOpcode() != Opcode || !Op->isAssociative() ||
        !Op->isCommutative())
      return false;

  // Define the neutral element of the reduction or bail out if we don't have
  // one defined.
  // TODO: This could be generalised to other operations (e.g. MUL's).
  Value *NeutralElem = nullptr;
  switch (Opcode) {
  case Instruction::BinaryOps::Add:
  case Instruction::BinaryOps::Or:
  case Instruction::BinaryOps::Xor:
  case Instruction::BinaryOps::FAdd:
    NeutralElem = Constant::getNullValue(PN->getType());
    break;
  case Instruction::BinaryOps::And:
    NeutralElem = Constant::getAllOnesValue(PN->getType());
    break;
  case Instruction::BinaryOps::Mul:
  case Instruction::BinaryOps::FMul:
  default:
    return false;
  }
  assert(NeutralElem && "Neutral element of reduction undefined.");

  // --------------------------------------------------------------------- //
  // At this point Ops is a list of chained binary operations performing a //
  // reduction that we know we can break apart.                            //
  // --------------------------------------------------------------------- //

  // For shorthand, let N be the length of the chain.
  unsigned const N = Ops.size();
  LLVM_DEBUG(dbgs() << "Simplifying reduction of length " << N << ".\n");

  // Create new phi nodes for all but the first element in the chain.
  SmallVector<PHINode *, 16> Phis{PN};
  for (unsigned i = 1; i < N; i++) {
    PHINode *NewPN = PHINode::Create(PN->getType(), PN->getNumIncomingValues(),
                                     PN->getName());
    // Copy incoming blocks from the first/original PN to the new Phi and set
    // their incoming values to the neutral element of the reduction.
    for (auto *IncomingBB : PN->blocks())
      NewPN->addIncoming(NeutralElem, IncomingBB);
    NewPN->insertAfter(Phis.back());
    Phis.push_back(NewPN);
  }

  // Set the chained operands of the Ops to the Phis and the incoming values of
  // the Phis (for this BB) to the Ops.
  for (unsigned i = 0; i < N; i++) {
    PHINode *Phi = Phis[i];
    Instruction *Op = Ops[i];

    // Find the index of the operand of Op to replace. The first Op reads its
    // value from the first Phi node. The other Ops read their value from the
    // previous Op.
    Value *OperandToReplace = i == 0 ? cast<Value>(PN) : Ops[i-1];
    unsigned OperandIdx = Op->getOperand(0) == OperandToReplace ? 0 : 1;
    assert(Op->getOperand(OperandIdx) == OperandToReplace &&
           "Operand mismatch. Perhaps a malformed chain?");

    // Set the operand of Op to Phi and the incoming value of Phi for BB to Op.
    Op->setOperand(OperandIdx, Phi);
    Phi->setIncomingValueForBlock(BB, Op);
  }

  // Replace old uses of PN and Ops outside this BB with the updated totals.
  // The "old" total corresponding to PN now corresponds to the sum of all
  // Phis. Similarly, the old totals in Ops correspond to the sum of the
  // partial results in the new Ops up to the index of the Op we want to
  // compute, plus the sum of the Phis from that index onwards.
  //
  // More rigorously, the totals can be computed as follows.
  // 1. Let k be an index in the list of length N+1 below of the variables we
  //    want to compute the new totals for:
  //      { PN, Ops[0], Ops[1], ... }
  // 2. Let Sum(k) denote the new total to compute for the k-th variable in the
  //    list above. Then,
  //      Sum(0) = Sum(PN) = \sum_{0 <= i < N} Phis[i],
  //      Sum(1) = Sum(Ops[0]) = \sum_{0 <= i < 1} Ops[i] +
  //                             \sum_{1 <= i < N} Phis[i],
  //      ...
  //      Sum(N) = Sum(Ops[N-1]) = \sum_{0 <= i < N} Ops[i].
  // 3. More generally,
  //      Sum(k) = Sum(PN) if k == 0 else Sum(Ops[k-1])
  //             = \sum_{0 <= i < k} Ops[i] +
  //               \sum_{k <= i < N} Phis[i],
  //      for 0 <= k <= N.
  // 4. Finally, if we name the sums in Ops and Phis separately, i.e.
  //      SOps(k) = \sum_{0 <= i < k} Ops[i],
  //      SPhis(k) = \sum_{k <= i < N} Phis[i],
  //    then
  //      Sum(k) = SOps(k) + SPhis(k), 0 <= k <= N.
  // .

  // Helper function to create a new binary op.
  // Note: We copy the flags from Ops[0]. Could this be too permissive?
  auto CreateBinOp = [&](Value *V1, Value *V2) {
    auto Name = PN->getName() + ".red";
    return BinaryOperator::CreateWithCopiedFlags(Opcode, V1, V2, Ops[0], Name,
                                                 &BB->back());
  };

  // Compute the partial sums of the Ops:
  //   SOps[k] = \sum_{0 <= i < k} Ops[i], 0 <= k <= N.
  // For 1 <= k <= N we have:
  //   SOps[k] = Ops[k-1] + \sum_{0 <= i < k-1} Ops[i]
  //           = Ops[k-1] + SOps[k-1],
  // so if we compute SOps in order (i.e. from 0 to N) we can reuse partial
  // results.
  SmallVector<Value *, 16> SOps(N+1);
  SOps[0] = nullptr; // alternatively we could use NeutralElem
  SOps[1] = Ops.front();
  for (unsigned k = 2; k <= N; k++)
    SOps[k] = CreateBinOp(SOps[k-1], Ops[k-1]);

  // Compute the partial sums of the Phis:
  //   SPhis[k] = \sum_{k <= i < N} Phis[i], 0 <= k <= N.
  // Similarly, for 0 <= k <= N-1 we have:
  //   SPhis[k] = Phis[k] + \sum_{k+1 <= i < N} Phis[i]
  //            = Phis[k] + SPhis[k+1],
  // so if we compute SPhis in reverse (i.e. from N down to 0) we can reuse the
  // partial sums computed thus far.
  SmallVector<Value *, 16> SPhis(N+1);
  SPhis[N] = nullptr; // alternatively we could use NeutralElem
  SPhis[N-1] = Phis.back();
  for (signed k = N-2; k >= 0; k--)
    SPhis[k] = CreateBinOp(SPhis[k+1], Phis[k]);

  // Finally, compute the total sums for PN and Ops from:
  //   Sums[k] = SOps[k] + SPhis[k], 0 <= k <= N.
  // These sums might be dead so we had them to a weak tracking vector for
  // cleanup after.
  SmallVector<WeakTrackingVH, 16> Sums(N+1);
  for (unsigned k = 0; k <= N; k++) {
    // Pick the Op we want to compute the new total for.
    Value *Op = k == 0 ? cast<Value>(PN) : Ops[k-1];

    Value *SOp = SOps[k], *SPhi = SPhis[k];
    if (SOp && SPhi)
      Sums[k] = CreateBinOp(SOp, SPhi);
    else if (SOp)
      Sums[k] = SOp;
    else
      Sums[k] = SPhi;

    // Replace uses of the old total with the new total.
    Op->replaceUsesOutsideBlock(Sums[k], BB);
  }

  // Drop dead totals. In case the totals *are* used they could and should be
  // sunk, but this happens in later passes so we don't bother doing it here.
  RecursivelyDeleteTriviallyDeadInstructionsPermissive(Sums);

  return true;
}

/// Perform some cleanup and simplifications on loops after unrolling. It is
/// useful to simplify the IV's in the new loop, as well as do a quick
/// simplify/dce pass of the instructions.
void llvm::simplifyLoopAfterUnroll(Loop *L, bool SimplifyIVs, LoopInfo *LI,
                                   ScalarEvolution *SE, DominatorTree *DT,
                                   AssumptionCache *AC,
                                   const TargetTransformInfo *TTI) {
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
  }

  // At this point, the code is well formed.  Perform constprop, instsimplify,
  // and dce.
  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();
  SmallVector<WeakTrackingVH, 16> DeadInsts;
  for (BasicBlock *BB : L->getBlocks()) {
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
    // Try to simplify reductions (e.g. chains of floating-point adds) into
    // independent operations (see more at trySimplifyReductions). This is a
    // very common pattern in unrolled loops that compute dot products (for
    // example).
    //
    // We do this outside the loop over the instructions above to let
    // instsimplify kick in before trying to apply this transform.
    if (UnrollSimplifyReductions)
      for (PHINode &PN : BB->phis())
        trySimplifyReductions(PN);
  }
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
LoopUnrollResult llvm::UnrollLoop(Loop *L, UnrollLoopOptions ULO, LoopInfo *LI,
                                  ScalarEvolution *SE, DominatorTree *DT,
                                  AssumptionCache *AC,
                                  const TargetTransformInfo *TTI,
                                  OptimizationRemarkEmitter *ORE,
                                  bool PreserveLCSSA, Loop **RemainderLoop) {
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
  unsigned EstimatedLoopInvocationWeight = 0;
  std::optional<unsigned> OriginalTripCount =
      llvm::getLoopEstimatedTripCount(L, &EstimatedLoopInvocationWeight);

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

    ExitInfo &Info = ExitInfos.try_emplace(ExitingBlock).first->second;
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

  // Loops containing convergent instructions cannot use runtime unrolling,
  // as the prologue/epilogue may add additional control-dependencies to
  // convergent operations.
  LLVM_DEBUG(
      {
        bool HasConvergent = false;
        for (auto &BB : L->blocks())
          for (auto &I : *BB)
            if (auto *CB = dyn_cast<CallBase>(&I))
              HasConvergent |= CB->isConvergent();
        assert((!HasConvergent || !ULO.Runtime) &&
               "Can't runtime unroll if loop contains a convergent operation.");
      });

  bool EpilogProfitability =
      UnrollRuntimeEpilog.getNumOccurrences() ? UnrollRuntimeEpilog
                                              : isEpilogProfitable(L);

  if (ULO.Runtime &&
      !UnrollRuntimeLoopRemainder(L, ULO.Count, ULO.AllowExpensiveTripCount,
                                  EpilogProfitability, ULO.UnrollRemainder,
                                  ULO.ForgetAllSCEV, LI, SE, DT, AC, TTI,
                                  PreserveLCSSA, RemainderLoop)) {
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
  for (Loop *SubLoop : *L)
    LoopsToSimplify.insert(SubLoop);

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

      if (*BB == Header)
        // Loop over all of the PHI nodes in the block, changing them to use
        // the incoming values from the previous block.
        for (PHINode *OrigPHI : OrigPHINode) {
          PHINode *NewPHI = cast<PHINode>(VMap[OrigPHI]);
          Value *InVal = NewPHI->getIncomingValueForBlock(LatchBlock);
          if (Instruction *InValI = dyn_cast<Instruction>(InVal))
            if (It > 1 && L->contains(InValI))
              InVal = LastValueMap[InValI];
          VMap[OrigPHI] = InVal;
          NewPHI->eraseFromParent();
        }

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
          SE->forgetValue(&PHI);
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

    // Remap all instructions in the most recent iteration
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
    BranchInst::Create(Dest, Term->getIterator());
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
  // exit.
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

  // Merge adjacent basic blocks, if possible.
  for (BasicBlock *Latch : Latches) {
    BranchInst *Term = dyn_cast<BranchInst>(Latch->getTerminator());
    assert((Term ||
            (CompletelyUnroll && !LatchIsExiting && Latch == Latches.back())) &&
           "Need a branch as terminator, except when fully unrolling with "
           "unconditional latch");
    if (Term && Term->isUnconditional()) {
      BasicBlock *Dest = Term->getSuccessor(0);
      BasicBlock *Fold = Dest->getUniquePredecessor();
      if (MergeBlockIntoPredecessor(Dest, /*DTU=*/DTUToUse, LI,
                                    /*MSSAU=*/nullptr, /*MemDep=*/nullptr,
                                    /*PredecessorWithTwoSuccessors=*/false,
                                    DTUToUse ? nullptr : DT)) {
        // Dest has been folded into Fold. Update our worklists accordingly.
        std::replace(Latches.begin(), Latches.end(), Dest, Fold);
        llvm::erase(UnrolledLoopBlocks, Dest);
      }
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
                          TTI);

  NumCompletelyUnrolled += CompletelyUnroll;
  ++NumUnrolled;

  Loop *OuterL = L->getParentLoop();
  // Update LoopInfo if the loop is completely removed.
  if (CompletelyUnroll) {
    LI->erase(L);
    // We shouldn't try to use `L` anymore.
    L = nullptr;
  } else if (OriginalTripCount) {
    // Update the trip count. Note that the remainder has already logic
    // computing it in `UnrollRuntimeLoopRemainder`.
    setLoopEstimatedTripCount(L, *OriginalTripCount / ULO.Count,
                              EstimatedLoopInvocationWeight);
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

  for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; ++i) {
    MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
    if (!MD)
      continue;

    MDString *S = dyn_cast<MDString>(MD->getOperand(0));
    if (!S)
      continue;

    if (Name.equals(S->getString()))
      return MD;
  }
  return nullptr;
}
