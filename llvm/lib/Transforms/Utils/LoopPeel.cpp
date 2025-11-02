//===- LoopPeel.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Loop Peeling Utilities.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LoopPeel.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionPatternMatch.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;
using namespace llvm::SCEVPatternMatch;

#define DEBUG_TYPE "loop-peel"

STATISTIC(NumPeeled, "Number of loops peeled");
STATISTIC(NumPeeledEnd, "Number of loops peeled from end");

static cl::opt<unsigned> UnrollPeelCount(
    "unroll-peel-count", cl::Hidden,
    cl::desc("Set the unroll peeling count, for testing purposes"));

static cl::opt<bool>
    UnrollAllowPeeling("unroll-allow-peeling", cl::init(true), cl::Hidden,
                       cl::desc("Allows loops to be peeled when the dynamic "
                                "trip count is known to be low."));

static cl::opt<bool>
    UnrollAllowLoopNestsPeeling("unroll-allow-loop-nests-peeling",
                                cl::init(false), cl::Hidden,
                                cl::desc("Allows loop nests to be peeled."));

static cl::opt<unsigned> UnrollPeelMaxCount(
    "unroll-peel-max-count", cl::init(7), cl::Hidden,
    cl::desc("Max average trip count which will cause loop peeling."));

static cl::opt<unsigned> UnrollForcePeelCount(
    "unroll-force-peel-count", cl::init(0), cl::Hidden,
    cl::desc("Force a peel count regardless of profiling information."));

static cl::opt<bool> DisableAdvancedPeeling(
    "disable-advanced-peeling", cl::init(false), cl::Hidden,
    cl::desc(
        "Disable advance peeling. Issues for convergent targets (D134803)."));

static cl::opt<bool> EnablePeelingForIV(
    "enable-peeling-for-iv", cl::init(false), cl::Hidden,
    cl::desc("Enable peeling to convert Phi nodes into IVs"));

static const char *PeeledCountMetaData = "llvm.loop.peeled.count";

// Check whether we are capable of peeling this loop.
bool llvm::canPeel(const Loop *L) {
  // Make sure the loop is in simplified form
  if (!L->isLoopSimplifyForm())
    return false;
  if (!DisableAdvancedPeeling)
    return true;

  SmallVector<BasicBlock *, 4> Exits;
  L->getUniqueNonLatchExitBlocks(Exits);
  // The latch must either be the only exiting block or all non-latch exit
  // blocks have either a deopt or unreachable terminator or compose a chain of
  // blocks where the last one is either deopt or unreachable terminated. Both
  // deopt and unreachable terminators are a strong indication they are not
  // taken. Note that this is a profitability check, not a legality check. Also
  // note that LoopPeeling currently can only update the branch weights of latch
  // blocks and branch weights to blocks with deopt or unreachable do not need
  // updating.
  return llvm::all_of(Exits, IsBlockFollowedByDeoptOrUnreachable);
}

namespace {

// As a loop is peeled, it may be the case that Phi nodes become
// loop-invariant (ie, known because there is only one choice).
// For example, consider the following function:
//   void g(int);
//   void binary() {
//     int x = 0;
//     int y = 0;
//     int a = 0;
//     for(int i = 0; i <100000; ++i) {
//       g(x);
//       x = y;
//       g(a);
//       y = a + 1;
//       a = 5;
//     }
//   }
// Peeling 3 iterations is beneficial because the values for x, y and a
// become known.  The IR for this loop looks something like the following:
//
//   %i = phi i32 [ 0, %entry ], [ %inc, %if.end ]
//   %a = phi i32 [ 0, %entry ], [ 5, %if.end ]
//   %y = phi i32 [ 0, %entry ], [ %add, %if.end ]
//   %x = phi i32 [ 0, %entry ], [ %y, %if.end ]
//   ...
//   tail call void @_Z1gi(i32 signext %x)
//   tail call void @_Z1gi(i32 signext %a)
//   %add = add nuw nsw i32 %a, 1
//   %inc = add nuw nsw i32 %i, 1
//   %exitcond = icmp eq i32 %inc, 100000
//   br i1 %exitcond, label %for.cond.cleanup, label %for.body
//
// The arguments for the calls to g will become known after 3 iterations
// of the loop, because the phi nodes values become known after 3 iterations
// of the loop (ie, they are known on the 4th iteration, so peel 3 iterations).
// The first iteration has g(0), g(0); the second has g(0), g(5); the
// third has g(1), g(5) and the fourth (and all subsequent) have g(6), g(5).
// Now consider the phi nodes:
//   %a is a phi with constants so it is determined after iteration 1.
//   %y is a phi based on a constant and %a so it is determined on
//     the iteration after %a is determined, so iteration 2.
//   %x is a phi based on a constant and %y so it is determined on
//     the iteration after %y, so iteration 3.
//   %i is based on itself (and is an induction variable) so it is
//     never determined.
// This means that peeling off 3 iterations will result in being able to
// remove the phi nodes for %a, %y, and %x.  The arguments for the
// corresponding calls to g are determined and the code for computing
// x, y, and a can be removed.
//
// Similarly, there are cases where peeling makes Phi nodes loop-inductions
// (i.e., the value is increased or decreased by a fixed amount on every
// iteration). For example, consider the following function.
//
//   #define N 100
//   void f(int a[], int b[]) {
//     int im = N - 1;
//     for (int i = 0; i < N; i++) {
//       a[i] = b[i] + b[im];
//       im = i;
//     }
//   }
//
// The IR of the loop will look something like the following.
//
//   %i = phi i32 [ 0, %entry ], [ %i.next, %for.body ]
//   %im = phi i32 [ 99, %entry ], [ %i, %for.body ]
//   ...
//   %i.next = add nuw nsw i32 %i, 1
//   ...
//
// In this case, %im becomes a loop-induction variable by peeling 1 iteration,
// because %i is a loop-induction one. The peeling count can be determined by
// the same algorithm with loop-invariant case. Such peeling is profitable for
// loop-vectorization.
//
// The PhiAnalyzer class calculates how many times a loop should be
// peeled based on the above analysis of the phi nodes in the loop while
// respecting the maximum specified.
class PhiAnalyzer {
public:
  PhiAnalyzer(const Loop &L, unsigned MaxIterations, bool PeelForIV);

  // Calculate the sufficient minimum number of iterations of the loop to peel
  // such that phi instructions become determined (subject to allowable limits)
  std::optional<unsigned> calculateIterationsToPeel();

protected:
  enum class PeelCounterType {
    Invariant,
    Induction,
  };

  using PeelCounterValue = std::pair<unsigned, PeelCounterType>;
  using PeelCounter = std::optional<PeelCounterValue>;
  const PeelCounter Unknown = std::nullopt;

  // Add 1 respecting Unknown and return Unknown if result over MaxIterations
  PeelCounter addOne(PeelCounter PC) const {
    if (PC == Unknown)
      return Unknown;
    auto [Val, Ty] = *PC;
    return (Val + 1 <= MaxIterations) ? PeelCounter({Val + 1, Ty}) : Unknown;
  }

  // Return a value representing zero for the given counter type.
  PeelCounter makeZero(PeelCounterType Ty) const {
    return PeelCounter({0, Ty});
  }

  // Calculate the number of iterations after which the given value becomes an
  // invariant or an induction.
  PeelCounter calculate(const Value &);

  // Auxiliary function to calculate the number of iterations for a comparison
  // instruction or a binary operator.
  PeelCounter mergeTwoCounters(const Instruction &CmpOrBinaryOp,
                               const PeelCounterValue &LHS,
                               const PeelCounterValue &RHS) const;

  // Returns true if the \p Phi is an induction in the target loop. This is a
  // lightweight check and possible to detect an IV in some cases.
  bool isInductionPHI(const PHINode *Phi) const;

  const Loop &L;
  const unsigned MaxIterations;
  const bool PeelForIV;

  // Map of Values to number of iterations to invariance or induction
  SmallDenseMap<const Value *, PeelCounter> IterationsToInvarianceOrInduction;
};

PhiAnalyzer::PhiAnalyzer(const Loop &L, unsigned MaxIterations, bool PeelForIV)
    : L(L), MaxIterations(MaxIterations), PeelForIV(PeelForIV) {
  assert(canPeel(&L) && "loop is not suitable for peeling");
  assert(MaxIterations > 0 && "no peeling is allowed?");
}

/// Test whether \p Phi is an induction variable. Although this can be
/// determined using SCEV analysis, it is expensive to compute here. Instead,
/// we perform cheaper checks that may not detect complex cases but are
/// sufficient for some situations.
bool PhiAnalyzer::isInductionPHI(const PHINode *Phi) const {
  // Currently we only support a loop that has single latch.
  BasicBlock *Latch = L.getLoopLatch();
  if (Latch == nullptr)
    return false;

  Value *Cur = Phi->getIncomingValueForBlock(Latch);
  SmallPtrSet<Value *, 4> Visited;
  bool VisitBinOp = false;

  // Starting from the incoming value of the Phi, we follow the use-def chain.
  // We consider Phi to be an IV if we can reach it again by traversing only
  // add, sub, or cast instructions.
  while (true) {
    if (Cur == Phi)
      break;

    // Avoid infinite loop.
    if (!Visited.insert(Cur).second)
      return false;

    auto *I = dyn_cast<Instruction>(Cur);
    if (!I || !L.contains(I))
      return false;

    if (auto *Cast = dyn_cast<CastInst>(I)) {
      Cur = Cast->getOperand(0);
    } else if (auto *BinOp = dyn_cast<BinaryOperator>(I)) {
      if (BinOp->getOpcode() != Instruction::Add &&
          BinOp->getOpcode() != Instruction::Sub)
        return false;
      if (!isa<ConstantInt>(BinOp->getOperand(1)))
        return false;

      VisitBinOp = true;
      Cur = BinOp->getOperand(0);
    } else {
      return false;
    }
  }

  // Ignore cases where no binary operations are visited.
  return VisitBinOp;
}

/// When either \p LHS or \p RHS is an IV, the result of \p CmpOrBinaryOp is
/// considered an IV only if it is an addition or a subtraction. Otherwise the
/// result can be a value that is neither a loop-invariant nor an IV.
///
/// If both \p LHS and \p RHS are loop-invariants, then the result of
/// \CmpOrBinaryOp is also a loop-invariant.
PhiAnalyzer::PeelCounter
PhiAnalyzer::mergeTwoCounters(const Instruction &CmpOrBinaryOp,
                              const PeelCounterValue &LHS,
                              const PeelCounterValue &RHS) const {
  auto &[LVal, LTy] = LHS;
  auto &[RVal, RTy] = RHS;
  unsigned NewVal = std::max(LVal, RVal);

  if (LTy == PeelCounterType::Induction || RTy == PeelCounterType::Induction) {
    if (const auto *BinOp = dyn_cast<BinaryOperator>(&CmpOrBinaryOp)) {
      if (BinOp->getOpcode() == Instruction::Add ||
          BinOp->getOpcode() == Instruction::Sub)
        return PeelCounter({NewVal, PeelCounterType::Induction});
    }
    return Unknown;
  }
  return PeelCounter({NewVal, PeelCounterType::Invariant});
}

// This function calculates the number of iterations after which the value
// becomes an invariant. The pre-calculated values are memorized in a map.
// N.B. This number will be Unknown or <= MaxIterations.
// The function is calculated according to the following definition:
// Given %x = phi <Inputs from above the loop>, ..., [%y, %back.edge].
//   F(%x) = G(%y) + 1 (N.B. [MaxIterations | Unknown] + 1 => Unknown)
//   G(%y) = 0 if %y is a loop invariant
//   G(%y) = G(%BackEdgeValue) if %y is a phi in the header block
//   G(%y) = TODO: if %y is an expression based on phis and loop invariants
//           The example looks like:
//           %x = phi(0, %a) <-- becomes invariant starting from 3rd iteration.
//           %y = phi(0, 5)
//           %a = %y + 1
//   G(%y) = Unknown otherwise (including phi not in header block)
PhiAnalyzer::PeelCounter PhiAnalyzer::calculate(const Value &V) {
  // If we already know the answer, take it from the map.
  // Otherwise, place Unknown to map to avoid infinite recursion. Such
  // cycles can never stop on an invariant.
  auto [I, Inserted] =
      IterationsToInvarianceOrInduction.try_emplace(&V, Unknown);
  if (!Inserted)
    return I->second;

  if (L.isLoopInvariant(&V))
    // Loop invariant so known at start.
    return (IterationsToInvarianceOrInduction[&V] =
                makeZero(PeelCounterType::Invariant));
  if (const PHINode *Phi = dyn_cast<PHINode>(&V)) {
    if (Phi->getParent() != L.getHeader()) {
      // Phi is not in header block so Unknown.
      assert(IterationsToInvarianceOrInduction[&V] == Unknown &&
             "unexpected value saved");
      return Unknown;
    }

    // If Phi is an induction, register it as a starting point.
    if (PeelForIV && isInductionPHI(Phi))
      return (IterationsToInvarianceOrInduction[&V] =
                  makeZero(PeelCounterType::Induction));

    // We need to analyze the input from the back edge and add 1.
    Value *Input = Phi->getIncomingValueForBlock(L.getLoopLatch());
    PeelCounter Iterations = calculate(*Input);
    assert(IterationsToInvarianceOrInduction[Input] == Iterations &&
           "unexpected value saved");
    return (IterationsToInvarianceOrInduction[Phi] = addOne(Iterations));
  }
  if (const Instruction *I = dyn_cast<Instruction>(&V)) {
    if (isa<CmpInst>(I) || I->isBinaryOp()) {
      // Binary instructions get the max of the operands.
      PeelCounter LHS = calculate(*I->getOperand(0));
      if (LHS == Unknown)
        return Unknown;
      PeelCounter RHS = calculate(*I->getOperand(1));
      if (RHS == Unknown)
        return Unknown;
      return (IterationsToInvarianceOrInduction[I] =
                  mergeTwoCounters(*I, *LHS, *RHS));
    }
    if (I->isCast())
      // Cast instructions get the value of the operand.
      return (IterationsToInvarianceOrInduction[I] =
                  calculate(*I->getOperand(0)));
  }
  // TODO: handle more expressions

  // Everything else is Unknown.
  assert(IterationsToInvarianceOrInduction[&V] == Unknown &&
         "unexpected value saved");
  return Unknown;
}

std::optional<unsigned> PhiAnalyzer::calculateIterationsToPeel() {
  unsigned Iterations = 0;
  for (auto &PHI : L.getHeader()->phis()) {
    PeelCounter ToInvarianceOrInduction = calculate(PHI);
    if (ToInvarianceOrInduction != Unknown) {
      unsigned Val = ToInvarianceOrInduction->first;
      assert(Val <= MaxIterations && "bad result in phi analysis");
      Iterations = std::max(Iterations, Val);
      if (Iterations == MaxIterations)
        break;
    }
  }
  assert((Iterations <= MaxIterations) && "bad result in phi analysis");
  return Iterations ? std::optional<unsigned>(Iterations) : std::nullopt;
}

} // unnamed namespace

// Try to find any invariant memory reads that will become dereferenceable in
// the remainder loop after peeling. The load must also be used (transitively)
// by an exit condition. Returns the number of iterations to peel off (at the
// moment either 0 or 1).
static unsigned peelToTurnInvariantLoadsDerefencebale(Loop &L,
                                                      DominatorTree &DT,
                                                      AssumptionCache *AC) {
  // Skip loops with a single exiting block, because there should be no benefit
  // for the heuristic below.
  if (L.getExitingBlock())
    return 0;

  // All non-latch exit blocks must have an UnreachableInst terminator.
  // Otherwise the heuristic below may not be profitable.
  SmallVector<BasicBlock *, 4> Exits;
  L.getUniqueNonLatchExitBlocks(Exits);
  if (any_of(Exits, [](const BasicBlock *BB) {
        return !isa<UnreachableInst>(BB->getTerminator());
      }))
    return 0;

  // Now look for invariant loads that dominate the latch and are not known to
  // be dereferenceable. If there are such loads and no writes, they will become
  // dereferenceable in the loop if the first iteration is peeled off. Also
  // collect the set of instructions controlled by such loads. Only peel if an
  // exit condition uses (transitively) such a load.
  BasicBlock *Header = L.getHeader();
  BasicBlock *Latch = L.getLoopLatch();
  SmallPtrSet<Value *, 8> LoadUsers;
  const DataLayout &DL = L.getHeader()->getDataLayout();
  for (BasicBlock *BB : L.blocks()) {
    for (Instruction &I : *BB) {
      if (I.mayWriteToMemory())
        return 0;

      if (LoadUsers.contains(&I))
        LoadUsers.insert_range(I.users());
      // Do not look for reads in the header; they can already be hoisted
      // without peeling.
      if (BB == Header)
        continue;
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        Value *Ptr = LI->getPointerOperand();
        if (DT.dominates(BB, Latch) && L.isLoopInvariant(Ptr) &&
            !isDereferenceablePointer(Ptr, LI->getType(), DL, LI, AC, &DT))
          LoadUsers.insert_range(I.users());
      }
    }
  }
  SmallVector<BasicBlock *> ExitingBlocks;
  L.getExitingBlocks(ExitingBlocks);
  if (any_of(ExitingBlocks, [&LoadUsers](BasicBlock *Exiting) {
        return LoadUsers.contains(Exiting->getTerminator());
      }))
    return 1;
  return 0;
}

bool llvm::canPeelLastIteration(const Loop &L, ScalarEvolution &SE) {
  const SCEV *BTC = SE.getBackedgeTakenCount(&L);
  if (isa<SCEVCouldNotCompute>(BTC))
    return false;

  // Check if the exit condition of the loop can be adjusted by the peeling
  // codegen. For now, it must
  // * exit via the latch,
  // * the exit condition must be a NE/EQ compare of an induction with step
  // of 1 and must only be used by the exiting branch.
  BasicBlock *Latch = L.getLoopLatch();
  Value *Inc;
  Value *Bound;
  CmpPredicate Pred;
  BasicBlock *Succ1;
  BasicBlock *Succ2;
  return Latch && Latch == L.getExitingBlock() &&
         match(Latch->getTerminator(),
               m_Br(m_OneUse(m_ICmp(Pred, m_Value(Inc), m_Value(Bound))),
                    m_BasicBlock(Succ1), m_BasicBlock(Succ2))) &&
         ((Pred == CmpInst::ICMP_EQ && Succ2 == L.getHeader()) ||
          (Pred == CmpInst::ICMP_NE && Succ1 == L.getHeader())) &&
         Bound->getType()->isIntegerTy() && 
         SE.isLoopInvariant(SE.getSCEV(Bound), &L) &&
         match(SE.getSCEV(Inc),
               m_scev_AffineAddRec(m_SCEV(), m_scev_One(), m_SpecificLoop(&L)));
}

/// Returns true if the last iteration can be peeled off and the condition (Pred
/// LeftAR, RightSCEV) is known at the last iteration and the inverse condition
/// is known at the second-to-last.
static bool shouldPeelLastIteration(Loop &L, CmpPredicate Pred,
                                    const SCEVAddRecExpr *LeftAR,
                                    const SCEV *RightSCEV, ScalarEvolution &SE,
                                    const TargetTransformInfo &TTI) {
  if (!canPeelLastIteration(L, SE))
    return false;

  const SCEV *BTC = SE.getBackedgeTakenCount(&L);
  SCEVExpander Expander(SE, L.getHeader()->getDataLayout(), "loop-peel");
  if (!SE.isKnownNonZero(BTC) &&
      Expander.isHighCostExpansion(BTC, &L, SCEVCheapExpansionBudget, &TTI,
                                   L.getLoopPredecessor()->getTerminator()))
    return false;

  auto Guards = ScalarEvolution::LoopGuards::collect(&L, SE);
  BTC = SE.applyLoopGuards(BTC, Guards);
  RightSCEV = SE.applyLoopGuards(RightSCEV, Guards);
  const SCEV *ValAtLastIter = LeftAR->evaluateAtIteration(BTC, SE);
  const SCEV *ValAtSecondToLastIter = LeftAR->evaluateAtIteration(
      SE.getMinusSCEV(BTC, SE.getOne(BTC->getType())), SE);

  return SE.isKnownPredicate(ICmpInst::getInversePredicate(Pred), ValAtLastIter,
                             RightSCEV) &&
         SE.isKnownPredicate(Pred, ValAtSecondToLastIter, RightSCEV);
}

// Return the number of iterations to peel off from the beginning and end of the
// loop respectively, that make conditions in the body true/false. For example,
// if we peel 2 iterations off the loop below, the condition i < 2 can be
// evaluated at compile time.
//
//  for (i = 0; i < n; i++)
//    if (i < 2)
//      ..
//    else
//      ..
//   }
static std::pair<unsigned, unsigned>
countToEliminateCompares(Loop &L, unsigned MaxPeelCount, ScalarEvolution &SE,
                         const TargetTransformInfo &TTI) {
  assert(L.isLoopSimplifyForm() && "Loop needs to be in loop simplify form");
  unsigned DesiredPeelCount = 0;
  unsigned DesiredPeelCountLast = 0;

  // Do not peel the entire loop.
  const SCEV *BE = SE.getConstantMaxBackedgeTakenCount(&L);
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(BE))
    MaxPeelCount =
        std::min((unsigned)SC->getAPInt().getLimitedValue() - 1, MaxPeelCount);

  // Increase PeelCount while (IterVal Pred BoundSCEV) condition is satisfied;
  // return true if inversed condition become known before reaching the
  // MaxPeelCount limit.
  auto PeelWhilePredicateIsKnown =
      [&](unsigned &PeelCount, const SCEV *&IterVal, const SCEV *BoundSCEV,
          const SCEV *Step, ICmpInst::Predicate Pred) {
        while (PeelCount < MaxPeelCount &&
               SE.isKnownPredicate(Pred, IterVal, BoundSCEV)) {
          IterVal = SE.getAddExpr(IterVal, Step);
          ++PeelCount;
        }
        return SE.isKnownPredicate(ICmpInst::getInversePredicate(Pred), IterVal,
                                   BoundSCEV);
      };

  const unsigned MaxDepth = 4;
  std::function<void(Value *, unsigned)> ComputePeelCount =
      [&](Value *Condition, unsigned Depth) -> void {
    if (!Condition->getType()->isIntegerTy() || Depth >= MaxDepth)
      return;

    Value *LeftVal, *RightVal;
    if (match(Condition, m_And(m_Value(LeftVal), m_Value(RightVal))) ||
        match(Condition, m_Or(m_Value(LeftVal), m_Value(RightVal)))) {
      ComputePeelCount(LeftVal, Depth + 1);
      ComputePeelCount(RightVal, Depth + 1);
      return;
    }

    CmpPredicate Pred;
    if (!match(Condition, m_ICmp(Pred, m_Value(LeftVal), m_Value(RightVal))))
      return;

    const SCEV *LeftSCEV = SE.getSCEV(LeftVal);
    const SCEV *RightSCEV = SE.getSCEV(RightVal);

    // Do not consider predicates that are known to be true or false
    // independently of the loop iteration.
    if (SE.evaluatePredicate(Pred, LeftSCEV, RightSCEV))
      return;

    // Check if we have a condition with one AddRec and one non AddRec
    // expression. Normalize LeftSCEV to be the AddRec.
    if (!isa<SCEVAddRecExpr>(LeftSCEV)) {
      if (isa<SCEVAddRecExpr>(RightSCEV)) {
        std::swap(LeftSCEV, RightSCEV);
        Pred = ICmpInst::getSwappedPredicate(Pred);
      } else
        return;
    }

    const SCEVAddRecExpr *LeftAR = cast<SCEVAddRecExpr>(LeftSCEV);

    // Avoid huge SCEV computations in the loop below, make sure we only
    // consider AddRecs of the loop we are trying to peel.
    if (!LeftAR->isAffine() || LeftAR->getLoop() != &L)
      return;
    if (!(ICmpInst::isEquality(Pred) && LeftAR->hasNoSelfWrap()) &&
        !SE.getMonotonicPredicateType(LeftAR, Pred))
      return;

    // Check if extending the current DesiredPeelCount lets us evaluate Pred
    // or !Pred in the loop body statically.
    unsigned NewPeelCount = DesiredPeelCount;

    const SCEV *IterVal = LeftAR->evaluateAtIteration(
        SE.getConstant(LeftSCEV->getType(), NewPeelCount), SE);

    // If the original condition is not known, get the negated predicate
    // (which holds on the else branch) and check if it is known. This allows
    // us to peel of iterations that make the original condition false.
    if (!SE.isKnownPredicate(Pred, IterVal, RightSCEV))
      Pred = ICmpInst::getInversePredicate(Pred);

    const SCEV *Step = LeftAR->getStepRecurrence(SE);
    if (!PeelWhilePredicateIsKnown(NewPeelCount, IterVal, RightSCEV, Step,
                                   Pred)) {
      if (shouldPeelLastIteration(L, Pred, LeftAR, RightSCEV, SE, TTI))
        DesiredPeelCountLast = 1;
      return;
    }

    // However, for equality comparisons, that isn't always sufficient to
    // eliminate the comparsion in loop body, we may need to peel one more
    // iteration. See if that makes !Pred become unknown again.
    const SCEV *NextIterVal = SE.getAddExpr(IterVal, Step);
    if (ICmpInst::isEquality(Pred) &&
        !SE.isKnownPredicate(ICmpInst::getInversePredicate(Pred), NextIterVal,
                             RightSCEV) &&
        !SE.isKnownPredicate(Pred, IterVal, RightSCEV) &&
        SE.isKnownPredicate(Pred, NextIterVal, RightSCEV)) {
      if (NewPeelCount >= MaxPeelCount)
        return; // Need to peel one more iteration, but can't. Give up.
      ++NewPeelCount; // Great!
    }

    DesiredPeelCount = std::max(DesiredPeelCount, NewPeelCount);
    DesiredPeelCountLast = std::max(DesiredPeelCountLast, NewPeelCount);
  };

  auto ComputePeelCountMinMax = [&](MinMaxIntrinsic *MinMax) {
    if (!MinMax->getType()->isIntegerTy())
      return;
    Value *LHS = MinMax->getLHS(), *RHS = MinMax->getRHS();
    const SCEV *BoundSCEV, *IterSCEV;
    if (L.isLoopInvariant(LHS)) {
      BoundSCEV = SE.getSCEV(LHS);
      IterSCEV = SE.getSCEV(RHS);
    } else if (L.isLoopInvariant(RHS)) {
      BoundSCEV = SE.getSCEV(RHS);
      IterSCEV = SE.getSCEV(LHS);
    } else
      return;
    const auto *AddRec = dyn_cast<SCEVAddRecExpr>(IterSCEV);
    // For simplicity, we support only affine recurrences.
    if (!AddRec || !AddRec->isAffine() || AddRec->getLoop() != &L)
      return;
    const SCEV *Step = AddRec->getStepRecurrence(SE);
    bool IsSigned = MinMax->isSigned();
    // To minimize number of peeled iterations, we use strict relational
    // predicates here.
    ICmpInst::Predicate Pred;
    if (SE.isKnownPositive(Step))
      Pred = IsSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
    else if (SE.isKnownNegative(Step))
      Pred = IsSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
    else
      return;
    // Check that AddRec is not wrapping.
    if (!(IsSigned ? AddRec->hasNoSignedWrap() : AddRec->hasNoUnsignedWrap()))
      return;
    unsigned NewPeelCount = DesiredPeelCount;
    const SCEV *IterVal = AddRec->evaluateAtIteration(
        SE.getConstant(AddRec->getType(), NewPeelCount), SE);
    if (!PeelWhilePredicateIsKnown(NewPeelCount, IterVal, BoundSCEV, Step,
                                   Pred)) {
      if (shouldPeelLastIteration(L, Pred, AddRec, BoundSCEV, SE, TTI))
        DesiredPeelCountLast = 1;
      return;
    }
    DesiredPeelCount = NewPeelCount;
  };

  for (BasicBlock *BB : L.blocks()) {
    for (Instruction &I : *BB) {
      if (SelectInst *SI = dyn_cast<SelectInst>(&I))
        ComputePeelCount(SI->getCondition(), 0);
      if (MinMaxIntrinsic *MinMax = dyn_cast<MinMaxIntrinsic>(&I))
        ComputePeelCountMinMax(MinMax);
    }

    auto *BI = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BI || BI->isUnconditional())
      continue;

    // Ignore loop exit condition.
    if (L.getLoopLatch() == BB)
      continue;

    ComputePeelCount(BI->getCondition(), 0);
  }

  return {DesiredPeelCount, DesiredPeelCountLast};
}

/// This "heuristic" exactly matches implicit behavior which used to exist
/// inside getLoopEstimatedTripCount.  It was added here to keep an
/// improvement inside that API from causing peeling to become more aggressive.
/// This should probably be removed.
static bool violatesLegacyMultiExitLoopCheck(Loop *L) {
  BasicBlock *Latch = L->getLoopLatch();
  if (!Latch)
    return true;

  BranchInst *LatchBR = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!LatchBR || LatchBR->getNumSuccessors() != 2 || !L->isLoopExiting(Latch))
    return true;

  assert((LatchBR->getSuccessor(0) == L->getHeader() ||
          LatchBR->getSuccessor(1) == L->getHeader()) &&
         "At least one edge out of the latch must go to the header");

  SmallVector<BasicBlock *, 4> ExitBlocks;
  L->getUniqueNonLatchExitBlocks(ExitBlocks);
  return any_of(ExitBlocks, [](const BasicBlock *EB) {
      return !EB->getTerminatingDeoptimizeCall();
    });
}


// Return the number of iterations we want to peel off.
void llvm::computePeelCount(Loop *L, unsigned LoopSize,
                            TargetTransformInfo::PeelingPreferences &PP,
                            unsigned TripCount, DominatorTree &DT,
                            ScalarEvolution &SE, const TargetTransformInfo &TTI,
                            AssumptionCache *AC, unsigned Threshold) {
  assert(LoopSize > 0 && "Zero loop size is not allowed!");
  // Save the PP.PeelCount value set by the target in
  // TTI.getPeelingPreferences or by the flag -unroll-peel-count.
  unsigned TargetPeelCount = PP.PeelCount;
  PP.PeelCount = 0;
  PP.PeelLast = false;
  if (!canPeel(L))
    return;

  // Only try to peel innermost loops by default.
  // The constraint can be relaxed by the target in TTI.getPeelingPreferences
  // or by the flag -unroll-allow-loop-nests-peeling.
  if (!PP.AllowLoopNestsPeeling && !L->isInnermost())
    return;

  // If the user provided a peel count, use that.
  bool UserPeelCount = UnrollForcePeelCount.getNumOccurrences() > 0;
  if (UserPeelCount) {
    LLVM_DEBUG(dbgs() << "Force-peeling first " << UnrollForcePeelCount
                      << " iterations.\n");
    PP.PeelCount = UnrollForcePeelCount;
    PP.PeelProfiledIterations = true;
    return;
  }

  // Skip peeling if it's disabled.
  if (!PP.AllowPeeling)
    return;

  // Check that we can peel at least one iteration.
  if (2 * LoopSize > Threshold)
    return;

  unsigned AlreadyPeeled = 0;
  if (auto Peeled = getOptionalIntLoopAttribute(L, PeeledCountMetaData))
    AlreadyPeeled = *Peeled;
  // Stop if we already peeled off the maximum number of iterations.
  if (AlreadyPeeled >= UnrollPeelMaxCount)
    return;

  // Pay respect to limitations implied by loop size and the max peel count.
  unsigned MaxPeelCount = UnrollPeelMaxCount;
  MaxPeelCount = std::min(MaxPeelCount, Threshold / LoopSize - 1);

  // Start the max computation with the PP.PeelCount value set by the target
  // in TTI.getPeelingPreferences or by the flag -unroll-peel-count.
  unsigned DesiredPeelCount = TargetPeelCount;

  // Here we try to get rid of Phis which become invariants or inductions after
  // 1, 2, ..., N iterations of the loop. For this we compute the number for
  // iterations after which every Phi is guaranteed to become an invariant or an
  // induction, and try to peel the maximum number of iterations among these
  // values, thus turning all those Phis into invariants or inductions.
  if (MaxPeelCount > DesiredPeelCount) {
    // Check how many iterations are useful for resolving Phis
    auto NumPeels = PhiAnalyzer(*L, MaxPeelCount, EnablePeelingForIV)
                        .calculateIterationsToPeel();
    if (NumPeels)
      DesiredPeelCount = std::max(DesiredPeelCount, *NumPeels);
  }

  const auto &[CountToEliminateCmps, CountToEliminateCmpsLast] =
      countToEliminateCompares(*L, MaxPeelCount, SE, TTI);
  DesiredPeelCount = std::max(DesiredPeelCount, CountToEliminateCmps);

  if (DesiredPeelCount == 0)
    DesiredPeelCount = peelToTurnInvariantLoadsDerefencebale(*L, DT, AC);

  if (DesiredPeelCount > 0) {
    DesiredPeelCount = std::min(DesiredPeelCount, MaxPeelCount);
    // Consider max peel count limitation.
    assert(DesiredPeelCount > 0 && "Wrong loop size estimation?");
    if (DesiredPeelCount + AlreadyPeeled <= UnrollPeelMaxCount) {
      LLVM_DEBUG(dbgs() << "Peel " << DesiredPeelCount
                        << " iteration(s) to turn"
                        << " some Phis into invariants or inductions.\n");
      PP.PeelCount = DesiredPeelCount;
      PP.PeelProfiledIterations = false;
      PP.PeelLast = false;
      return;
    }
  }

  if (CountToEliminateCmpsLast > 0) {
    unsigned DesiredPeelCountLast =
        std::min(CountToEliminateCmpsLast, MaxPeelCount);
    // Consider max peel count limitation.
    assert(DesiredPeelCountLast > 0 && "Wrong loop size estimation?");
    if (DesiredPeelCountLast + AlreadyPeeled <= UnrollPeelMaxCount) {
      LLVM_DEBUG(dbgs() << "Peel " << DesiredPeelCount
                        << " iteration(s) to turn"
                        << " some Phis into invariants.\n");
      PP.PeelCount = DesiredPeelCountLast;
      PP.PeelProfiledIterations = false;
      PP.PeelLast = true;
      return;
    }
  }

  // Bail if we know the statically calculated trip count.
  // In this case we rather prefer partial unrolling.
  if (TripCount)
    return;

  // Do not apply profile base peeling if it is disabled.
  if (!PP.PeelProfiledIterations)
    return;
  // If we don't know the trip count, but have reason to believe the average
  // trip count is low, peeling should be beneficial, since we will usually
  // hit the peeled section.
  // We only do this in the presence of profile information, since otherwise
  // our estimates of the trip count are not reliable enough.
  if (L->getHeader()->getParent()->hasProfileData()) {
    if (violatesLegacyMultiExitLoopCheck(L))
      return;
    std::optional<unsigned> EstimatedTripCount = getLoopEstimatedTripCount(L);
    if (!EstimatedTripCount)
      return;

    LLVM_DEBUG(dbgs() << "Profile-based estimated trip count is "
                      << *EstimatedTripCount << "\n");

    if (*EstimatedTripCount + AlreadyPeeled <= MaxPeelCount) {
      unsigned PeelCount = *EstimatedTripCount;
      LLVM_DEBUG(dbgs() << "Peeling first " << PeelCount << " iterations.\n");
      PP.PeelCount = PeelCount;
      return;
    }
    LLVM_DEBUG(dbgs() << "Already peel count: " << AlreadyPeeled << "\n");
    LLVM_DEBUG(dbgs() << "Max peel count: " << UnrollPeelMaxCount << "\n");
    LLVM_DEBUG(dbgs() << "Loop cost: " << LoopSize << "\n");
    LLVM_DEBUG(dbgs() << "Max peel cost: " << Threshold << "\n");
    LLVM_DEBUG(dbgs() << "Max peel count by cost: "
                      << (Threshold / LoopSize - 1) << "\n");
  }
}

/// Clones the body of the loop L, putting it between \p InsertTop and \p
/// InsertBot.
/// \param IterNumber The serial number of the iteration currently being
/// peeled off.
/// \param PeelLast Peel off the last iterations from \p L.
/// \param ExitEdges The exit edges of the original loop.
/// \param[out] NewBlocks A list of the blocks in the newly created clone
/// \param[out] VMap The value map between the loop and the new clone.
/// \param LoopBlocks A helper for DFS-traversal of the loop.
/// \param LVMap A value-map that maps instructions from the original loop to
/// instructions in the last peeled-off iteration.
static void cloneLoopBlocks(
    Loop *L, unsigned IterNumber, bool PeelLast, BasicBlock *InsertTop,
    BasicBlock *InsertBot, BasicBlock *OrigPreHeader,
    SmallVectorImpl<std::pair<BasicBlock *, BasicBlock *>> &ExitEdges,
    SmallVectorImpl<BasicBlock *> &NewBlocks, LoopBlocksDFS &LoopBlocks,
    ValueToValueMapTy &VMap, ValueToValueMapTy &LVMap, DominatorTree *DT,
    LoopInfo *LI, ArrayRef<MDNode *> LoopLocalNoAliasDeclScopes,
    ScalarEvolution &SE) {
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  BasicBlock *PreHeader = L->getLoopPreheader();

  Function *F = Header->getParent();
  LoopBlocksDFS::RPOIterator BlockBegin = LoopBlocks.beginRPO();
  LoopBlocksDFS::RPOIterator BlockEnd = LoopBlocks.endRPO();
  Loop *ParentLoop = L->getParentLoop();

  // For each block in the original loop, create a new copy,
  // and update the value map with the newly created values.
  for (LoopBlocksDFS::RPOIterator BB = BlockBegin; BB != BlockEnd; ++BB) {
    BasicBlock *NewBB = CloneBasicBlock(*BB, VMap, ".peel", F);
    NewBlocks.push_back(NewBB);

    // If an original block is an immediate child of the loop L, its copy
    // is a child of a ParentLoop after peeling. If a block is a child of
    // a nested loop, it is handled in the cloneLoop() call below.
    if (ParentLoop && LI->getLoopFor(*BB) == L)
      ParentLoop->addBasicBlockToLoop(NewBB, *LI);

    VMap[*BB] = NewBB;

    // If dominator tree is available, insert nodes to represent cloned blocks.
    if (DT) {
      if (Header == *BB)
        DT->addNewBlock(NewBB, InsertTop);
      else {
        DomTreeNode *IDom = DT->getNode(*BB)->getIDom();
        // VMap must contain entry for IDom, as the iteration order is RPO.
        DT->addNewBlock(NewBB, cast<BasicBlock>(VMap[IDom->getBlock()]));
      }
    }
  }

  {
    // Identify what other metadata depends on the cloned version. After
    // cloning, replace the metadata with the corrected version for both
    // memory instructions and noalias intrinsics.
    std::string Ext = (Twine("Peel") + Twine(IterNumber)).str();
    cloneAndAdaptNoAliasScopes(LoopLocalNoAliasDeclScopes, NewBlocks,
                               Header->getContext(), Ext);
  }

  // Recursively create the new Loop objects for nested loops, if any,
  // to preserve LoopInfo.
  for (Loop *ChildLoop : *L) {
    cloneLoop(ChildLoop, ParentLoop, VMap, LI, nullptr);
  }

  // Hook-up the control flow for the newly inserted blocks.
  // The new header is hooked up directly to the "top", which is either
  // the original loop preheader (for the first iteration) or the previous
  // iteration's exiting block (for every other iteration)
  InsertTop->getTerminator()->setSuccessor(0, cast<BasicBlock>(VMap[Header]));

  // Similarly, for the latch:
  // The original exiting edge is still hooked up to the loop exit.
  BasicBlock *NewLatch = cast<BasicBlock>(VMap[Latch]);
  if (PeelLast) {
    // This is the last iteration and we definitely will go to the exit. Just
    // set both successors to InsertBot and let the branch be simplified later.
    assert(IterNumber == 0 && "Only peeling a single iteration implemented.");
    auto *LatchTerm = cast<BranchInst>(NewLatch->getTerminator());
    LatchTerm->setSuccessor(0, InsertBot);
    LatchTerm->setSuccessor(1, InsertBot);
  } else {
    auto *LatchTerm = cast<Instruction>(NewLatch->getTerminator());
    // The backedge now goes to the "bottom", which is either the loop's real
    // header (for the last peeled iteration) or the copied header of the next
    // iteration (for every other iteration)
    for (unsigned idx = 0, e = LatchTerm->getNumSuccessors(); idx < e; ++idx) {
      if (LatchTerm->getSuccessor(idx) == Header) {
        LatchTerm->setSuccessor(idx, InsertBot);
        break;
      }
    }
  }
  if (DT)
    DT->changeImmediateDominator(InsertBot, NewLatch);

  // The new copy of the loop body starts with a bunch of PHI nodes
  // that pick an incoming value from either the preheader, or the previous
  // loop iteration. Since this copy is no longer part of the loop, we
  // resolve this statically:
  if (PeelLast) {
    // For the last iteration, we introduce new phis for each header phi in
    // InsertTop, using the incoming value from the preheader for the original
    // preheader (when skipping the main loop) and the incoming value from the
    // latch for the latch (when continuing from the main loop).
    IRBuilder<> B(InsertTop, InsertTop->getFirstNonPHIIt());
    for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
      PHINode *NewPHI = cast<PHINode>(VMap[&*I]);
      PHINode *PN = B.CreatePHI(NewPHI->getType(), 2);
      NewPHI->eraseFromParent();
      if (OrigPreHeader)
        PN->addIncoming(cast<PHINode>(&*I)->getIncomingValueForBlock(PreHeader),
                        OrigPreHeader);

      PN->addIncoming(cast<PHINode>(&*I)->getIncomingValueForBlock(Latch),
                      Latch);
      VMap[&*I] = PN;
    }
  } else {
    // For the first iteration, we use the value from the preheader directly.
    // For any other iteration, we replace the phi with the value generated by
    // the immediately preceding clone of the loop body (which represents
    // the previous iteration).
    for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
      PHINode *NewPHI = cast<PHINode>(VMap[&*I]);
      if (IterNumber == 0) {
        VMap[&*I] = NewPHI->getIncomingValueForBlock(PreHeader);
      } else {
        Value *LatchVal = NewPHI->getIncomingValueForBlock(Latch);
        Instruction *LatchInst = dyn_cast<Instruction>(LatchVal);
        if (LatchInst && L->contains(LatchInst))
          VMap[&*I] = LVMap[LatchInst];
        else
          VMap[&*I] = LatchVal;
      }
      NewPHI->eraseFromParent();
    }
  }

  // Fix up the outgoing values - we need to add a value for the iteration
  // we've just created. Note that this must happen *after* the incoming
  // values are adjusted, since the value going out of the latch may also be
  // a value coming into the header.
  for (auto Edge : ExitEdges)
    for (PHINode &PHI : Edge.second->phis()) {
      Value *LatchVal = PHI.getIncomingValueForBlock(Edge.first);
      Instruction *LatchInst = dyn_cast<Instruction>(LatchVal);
      if (LatchInst && L->contains(LatchInst))
        LatchVal = VMap[LatchVal];
      PHI.addIncoming(LatchVal, cast<BasicBlock>(VMap[Edge.first]));
      SE.forgetLcssaPhiWithNewPredecessor(L, &PHI);
    }

  // LastValueMap is updated with the values for the current loop
  // which are used the next time this function is called.
  for (auto KV : VMap)
    LVMap[KV.first] = KV.second;
}

TargetTransformInfo::PeelingPreferences
llvm::gatherPeelingPreferences(Loop *L, ScalarEvolution &SE,
                               const TargetTransformInfo &TTI,
                               std::optional<bool> UserAllowPeeling,
                               std::optional<bool> UserAllowProfileBasedPeeling,
                               bool UnrollingSpecficValues) {
  TargetTransformInfo::PeelingPreferences PP;

  // Set the default values.
  PP.PeelCount = 0;
  PP.AllowPeeling = true;
  PP.AllowLoopNestsPeeling = false;
  PP.PeelLast = false;
  PP.PeelProfiledIterations = true;

  // Get the target specifc values.
  TTI.getPeelingPreferences(L, SE, PP);

  // User specified values using cl::opt.
  if (UnrollingSpecficValues) {
    if (UnrollPeelCount.getNumOccurrences() > 0)
      PP.PeelCount = UnrollPeelCount;
    if (UnrollAllowPeeling.getNumOccurrences() > 0)
      PP.AllowPeeling = UnrollAllowPeeling;
    if (UnrollAllowLoopNestsPeeling.getNumOccurrences() > 0)
      PP.AllowLoopNestsPeeling = UnrollAllowLoopNestsPeeling;
  }

  // User specifed values provided by argument.
  if (UserAllowPeeling)
    PP.AllowPeeling = *UserAllowPeeling;
  if (UserAllowProfileBasedPeeling)
    PP.PeelProfiledIterations = *UserAllowProfileBasedPeeling;

  return PP;
}

/// Peel off the first \p PeelCount iterations of loop \p L.
///
/// Note that this does not peel them off as a single straight-line block.
/// Rather, each iteration is peeled off separately, and needs to check the
/// exit condition.
/// For loops that dynamically execute \p PeelCount iterations or less
/// this provides a benefit, since the peeled off iterations, which account
/// for the bulk of dynamic execution, can be further simplified by scalar
/// optimizations.
bool llvm::peelLoop(Loop *L, unsigned PeelCount, bool PeelLast, LoopInfo *LI,
                    ScalarEvolution *SE, DominatorTree &DT, AssumptionCache *AC,
                    bool PreserveLCSSA, ValueToValueMapTy &LVMap) {
  assert(PeelCount > 0 && "Attempt to peel out zero iterations?");
  assert(canPeel(L) && "Attempt to peel a loop which is not peelable?");
  assert((!PeelLast || (canPeelLastIteration(*L, *SE) && PeelCount == 1)) &&
         "when peeling the last iteration, the loop must be supported and can "
         "only peel a single iteration");

  LoopBlocksDFS LoopBlocks(L);
  LoopBlocks.perform(LI);

  BasicBlock *Header = L->getHeader();
  BasicBlock *PreHeader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();
  SmallVector<std::pair<BasicBlock *, BasicBlock *>, 4> ExitEdges;
  L->getExitEdges(ExitEdges);

  // Remember dominators of blocks we might reach through exits to change them
  // later. Immediate dominator of such block might change, because we add more
  // routes which can lead to the exit: we can reach it from the peeled
  // iterations too.
  DenseMap<BasicBlock *, BasicBlock *> NonLoopBlocksIDom;
  for (auto *BB : L->blocks()) {
    auto *BBDomNode = DT.getNode(BB);
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
    BasicBlock *NewIDom = DT.findNearestCommonDominator(BB, Latch);
    for (auto *ChildBB : ChildrenToUpdate)
      NonLoopBlocksIDom[ChildBB] = NewIDom;
  }

  Function *F = Header->getParent();

  // Set up all the necessary basic blocks.
  BasicBlock *InsertTop;
  BasicBlock *InsertBot;
  BasicBlock *NewPreHeader = nullptr;
  DenseMap<Instruction *, Value *> ExitValues;
  if (PeelLast) {
    // It is convenient to split the single exit block from the latch the
    // into 3 parts - two blocks to anchor the peeled copy of the loop body,
    // and a new final  exit block.

    // Peeling the last iteration transforms.
    //
    // PreHeader:
    // ...
    // Header:
    //   LoopBody
    //   If (cond) goto Header
    // Exit:
    //
    // into
    //
    // Header:
    //  LoopBody
    //  If (cond) goto Header
    // InsertTop:
    //   LoopBody
    //   If (!cond) goto InsertBot
    // InsertBot:
    // Exit:
    // ...
    BasicBlock *Exit = L->getExitBlock();
    for (PHINode &P : Exit->phis())
      ExitValues[&P] = P.getIncomingValueForBlock(Latch);

    const SCEV *BTC = SE->getBackedgeTakenCount(L);

    InsertTop = SplitEdge(Latch, Exit, &DT, LI);
    InsertBot = SplitBlock(InsertTop, InsertTop->getTerminator(), &DT, LI);

    InsertTop->setName(Exit->getName() + ".peel.begin");
    InsertBot->setName(Exit->getName() + ".peel.next");
    NewPreHeader = nullptr;

    // If the original loop may only execute a single iteration we need to
    // insert a trip count check and skip the original loop with the last
    // iteration peeled off if necessary.
    if (!SE->isKnownNonZero(BTC)) {
      NewPreHeader = SplitEdge(PreHeader, Header, &DT, LI);
      SCEVExpander Expander(*SE, Latch->getDataLayout(), "loop-peel");

      BranchInst *PreHeaderBR = cast<BranchInst>(PreHeader->getTerminator());
      Value *BTCValue =
          Expander.expandCodeFor(BTC, BTC->getType(), PreHeaderBR);
      IRBuilder<> B(PreHeaderBR);
      Value *Cond =
          B.CreateICmpNE(BTCValue, ConstantInt::get(BTCValue->getType(), 0));
      B.CreateCondBr(Cond, NewPreHeader, InsertTop);
      PreHeaderBR->eraseFromParent();

      // PreHeader now dominates InsertTop.
      DT.changeImmediateDominator(InsertTop, PreHeader);
    }
  } else {
    // It is convenient to split the preheader into 3 parts - two blocks to
    // anchor the peeled copy of the loop body, and a new preheader for the
    // "real" loop.

    // Peeling the first iteration transforms.
    //
    // PreHeader:
    // ...
    // Header:
    //   LoopBody
    //   If (cond) goto Header
    // Exit:
    //
    // into
    //
    // InsertTop:
    //   LoopBody
    //   If (!cond) goto Exit
    // InsertBot:
    // NewPreHeader:
    // ...
    // Header:
    //  LoopBody
    //  If (cond) goto Header
    // Exit:
    //
    // Each following iteration will split the current bottom anchor in two,
    // and put the new copy of the loop body between these two blocks. That
    // is, after peeling another iteration from the example above, we'll
    // split InsertBot, and get:
    //
    // InsertTop:
    //   LoopBody
    //   If (!cond) goto Exit
    // InsertBot:
    //   LoopBody
    //   If (!cond) goto Exit
    // InsertBot.next:
    // NewPreHeader:
    // ...
    // Header:
    //  LoopBody
    //  If (cond) goto Header
    // Exit:
    //
    InsertTop = SplitEdge(PreHeader, Header, &DT, LI);
    InsertBot = SplitBlock(InsertTop, InsertTop->getTerminator(), &DT, LI);
    NewPreHeader = SplitBlock(InsertBot, InsertBot->getTerminator(), &DT, LI);

    InsertTop->setName(Header->getName() + ".peel.begin");
    InsertBot->setName(Header->getName() + ".peel.next");
    NewPreHeader->setName(PreHeader->getName() + ".peel.newph");
  }

  Instruction *LatchTerm =
      cast<Instruction>(cast<BasicBlock>(Latch)->getTerminator());

  // Identify what noalias metadata is inside the loop: if it is inside the
  // loop, the associated metadata must be cloned for each iteration.
  SmallVector<MDNode *, 6> LoopLocalNoAliasDeclScopes;
  identifyNoAliasScopesToClone(L->getBlocks(), LoopLocalNoAliasDeclScopes);

  // For each peeled-off iteration, make a copy of the loop.
  ValueToValueMapTy VMap;
  for (unsigned Iter = 0; Iter < PeelCount; ++Iter) {
    SmallVector<BasicBlock *, 8> NewBlocks;

    cloneLoopBlocks(L, Iter, PeelLast, InsertTop, InsertBot,
                    NewPreHeader ? PreHeader : nullptr, ExitEdges, NewBlocks,
                    LoopBlocks, VMap, LVMap, &DT, LI,
                    LoopLocalNoAliasDeclScopes, *SE);

    // Remap to use values from the current iteration instead of the
    // previous one.
    remapInstructionsInBlocks(NewBlocks, VMap);

    if (Iter == 0) {
      if (PeelLast) {
        // Adjust the exit condition so the loop exits one iteration early.
        // For now we simply subtract one form the second operand of the
        // exit condition. This relies on the peel count computation to
        // check that this is actually legal. In particular, it ensures that
        // the first operand of the compare is an AddRec with step 1 and we
        // execute more than one iteration.
        auto *Cmp =
            cast<ICmpInst>(L->getLoopLatch()->getTerminator()->getOperand(0));
        IRBuilder B(Cmp);
        Cmp->setOperand(
            1, B.CreateSub(Cmp->getOperand(1),
                           ConstantInt::get(Cmp->getOperand(1)->getType(), 1)));
      } else {
        // Update IDoms of the blocks reachable through exits.
        for (auto BBIDom : NonLoopBlocksIDom)
          DT.changeImmediateDominator(BBIDom.first,
                                      cast<BasicBlock>(LVMap[BBIDom.second]));
      }
    }

#ifdef EXPENSIVE_CHECKS
    assert(DT.verify(DominatorTree::VerificationLevel::Fast));
#endif

    // Remove Loop metadata from the latch branch instruction
    // because it is not the Loop's latch branch anymore.
    auto *LatchTermCopy = cast<Instruction>(VMap[LatchTerm]);
    LatchTermCopy->setMetadata(LLVMContext::MD_loop, nullptr);

    InsertTop = InsertBot;
    InsertBot = SplitBlock(InsertBot, InsertBot->getTerminator(), &DT, LI);
    InsertBot->setName(Header->getName() + ".peel.next");

    F->splice(InsertTop->getIterator(), F, NewBlocks[0]->getIterator(),
              F->end());
  }

  if (PeelLast) {
    // Now adjust users of the original exit values by replacing them with the
    // exit value from the peeled iteration and remove them.
    for (const auto &[P, E] : ExitValues) {
      Instruction *ExitInst = dyn_cast<Instruction>(E);
      if (ExitInst && L->contains(ExitInst))
        P->replaceAllUsesWith(&*VMap[ExitInst]);
      else
        P->replaceAllUsesWith(E);
      P->eraseFromParent();
    }
    formLCSSA(*L, DT, LI, SE);
  } else {
    // Now adjust the phi nodes in the loop header to get their initial values
    // from the last peeled-off iteration instead of the preheader.
    for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
      PHINode *PHI = cast<PHINode>(I);
      Value *NewVal = PHI->getIncomingValueForBlock(Latch);
      Instruction *LatchInst = dyn_cast<Instruction>(NewVal);
      if (LatchInst && L->contains(LatchInst))
        NewVal = LVMap[LatchInst];

      PHI->setIncomingValueForBlock(NewPreHeader, NewVal);
    }
  }

  // Update Metadata for count of peeled off iterations.
  unsigned AlreadyPeeled = 0;
  if (auto Peeled = getOptionalIntLoopAttribute(L, PeeledCountMetaData))
    AlreadyPeeled = *Peeled;
  unsigned TotalPeeled = AlreadyPeeled + PeelCount;
  addStringMetadataToLoop(L, PeeledCountMetaData, TotalPeeled);

  // Update metadata for the estimated trip count.  The original branch weight
  // metadata is already correct for both the remaining loop and the peeled loop
  // iterations, so do not adjust it.
  //
  // For example, consider what happens when peeling 2 iterations from a loop
  // with an estimated trip count of 10 and inserting them before the remaining
  // loop.  Each of the peeled iterations and each iteration in the remaining
  // loop still has the same probability of exiting the *entire original* loop
  // as it did when in the original loop, and thus it should still have the same
  // branch weights.  The peeled iterations' non-zero probabilities of exiting
  // already appropriately reduce the probability of reaching the remaining
  // iterations just as they did in the original loop.  Trying to also adjust
  // the remaining loop's branch weights to reflect its new trip count of 8 will
  // erroneously further reduce its block frequencies.  However, in case an
  // analysis later needs to determine the trip count of the remaining loop
  // while examining it in isolation without considering the probability of
  // actually reaching it, we store the new trip count as separate metadata.
  if (auto EstimatedTripCount = getLoopEstimatedTripCount(L)) {
    unsigned EstimatedTripCountNew = *EstimatedTripCount;
    if (EstimatedTripCountNew < TotalPeeled)
      EstimatedTripCountNew = 0;
    else
      EstimatedTripCountNew -= TotalPeeled;
    setLoopEstimatedTripCount(L, EstimatedTripCountNew);
  }

  if (Loop *ParentLoop = L->getParentLoop())
    L = ParentLoop;

  // We modified the loop, update SE.
  SE->forgetTopmostLoop(L);
  SE->forgetBlockAndLoopDispositions();

#ifdef EXPENSIVE_CHECKS
  // Finally DomtTree must be correct.
  assert(DT.verify(DominatorTree::VerificationLevel::Fast));
#endif

  // FIXME: Incrementally update loop-simplify
  simplifyLoop(L, &DT, LI, SE, AC, nullptr, PreserveLCSSA);

  NumPeeled++;
  NumPeeledEnd += PeelLast;

  return true;
}
