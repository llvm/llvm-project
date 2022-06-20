//===-- ConstraintElimination.cpp - Eliminate conds using constraints. ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Eliminate conditions based on constraints collected from dominating
// conditions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/ConstraintElimination.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ConstraintSystem.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Scalar.h"

#include <string>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "constraint-elimination"

STATISTIC(NumCondsRemoved, "Number of instructions removed");
DEBUG_COUNTER(EliminatedCounter, "conds-eliminated",
              "Controls which conditions are eliminated");

static int64_t MaxConstraintValue = std::numeric_limits<int64_t>::max();
static int64_t MinSignedConstraintValue = std::numeric_limits<int64_t>::min();

namespace {

class ConstraintInfo;

struct StackEntry {
  unsigned NumIn;
  unsigned NumOut;
  Instruction *Condition;
  bool IsNot;
  bool IsSigned = false;
  /// Variables that can be removed from the system once the stack entry gets
  /// removed.
  SmallVector<Value *, 2> ValuesToRelease;

  StackEntry(unsigned NumIn, unsigned NumOut, CmpInst *Condition, bool IsNot,
             bool IsSigned, SmallVector<Value *, 2> ValuesToRelease)
      : NumIn(NumIn), NumOut(NumOut), Condition(Condition), IsNot(IsNot),
        IsSigned(IsSigned), ValuesToRelease(ValuesToRelease) {}
};

/// Struct to express a pre-condition of the form %Op0 Pred %Op1.
struct PreconditionTy {
  CmpInst::Predicate Pred;
  Value *Op0;
  Value *Op1;

  PreconditionTy(CmpInst::Predicate Pred, Value *Op0, Value *Op1)
      : Pred(Pred), Op0(Op0), Op1(Op1) {}
};

struct ConstraintTy {
  SmallVector<int64_t, 8> Coefficients;
  SmallVector<PreconditionTy, 2> Preconditions;

  bool IsSigned = false;
  bool IsEq = false;

  ConstraintTy() = default;

  ConstraintTy(SmallVector<int64_t, 8> Coefficients, bool IsSigned)
      : Coefficients(Coefficients), IsSigned(IsSigned) {}

  unsigned size() const { return Coefficients.size(); }

  unsigned empty() const { return Coefficients.empty(); }

  /// Returns true if any constraint has a non-zero coefficient for any of the
  /// newly added indices. Zero coefficients for new indices are removed. If it
  /// returns true, no new variable need to be added to the system.
  bool needsNewIndices(const DenseMap<Value *, unsigned> &NewIndices) {
    for (unsigned I = 0; I < NewIndices.size(); ++I) {
      int64_t Last = Coefficients.pop_back_val();
      if (Last != 0)
        return true;
    }
    return false;
  }

  /// Returns true if all preconditions for this list of constraints are
  /// satisfied given \p CS and the corresponding \p Value2Index mapping.
  bool isValid(const ConstraintInfo &Info) const;
};

/// Wrapper encapsulating separate constraint systems and corresponding value
/// mappings for both unsigned and signed information. Facts are added to and
/// conditions are checked against the corresponding system depending on the
/// signed-ness of their predicates. While the information is kept separate
/// based on signed-ness, certain conditions can be transferred between the two
/// systems.
class ConstraintInfo {
  DenseMap<Value *, unsigned> UnsignedValue2Index;
  DenseMap<Value *, unsigned> SignedValue2Index;

  ConstraintSystem UnsignedCS;
  ConstraintSystem SignedCS;

public:
  DenseMap<Value *, unsigned> &getValue2Index(bool Signed) {
    return Signed ? SignedValue2Index : UnsignedValue2Index;
  }
  const DenseMap<Value *, unsigned> &getValue2Index(bool Signed) const {
    return Signed ? SignedValue2Index : UnsignedValue2Index;
  }

  ConstraintSystem &getCS(bool Signed) {
    return Signed ? SignedCS : UnsignedCS;
  }
  const ConstraintSystem &getCS(bool Signed) const {
    return Signed ? SignedCS : UnsignedCS;
  }

  void popLastConstraint(bool Signed) { getCS(Signed).popLastConstraint(); }
  void popLastNVariables(bool Signed, unsigned N) {
    getCS(Signed).popLastNVariables(N);
  }
};

} // namespace

// Decomposes \p V into a vector of pairs of the form { c, X } where c * X. The
// sum of the pairs equals \p V.  The first pair is the constant-factor and X
// must be nullptr. If the expression cannot be decomposed, returns an empty
// vector.
static SmallVector<std::pair<int64_t, Value *>, 4>
decompose(Value *V, SmallVector<PreconditionTy, 4> &Preconditions,
          bool IsSigned) {

  auto CanUseSExt = [](ConstantInt *CI) {
    const APInt &Val = CI->getValue();
    return Val.sgt(MinSignedConstraintValue) && Val.slt(MaxConstraintValue);
  };
  // Decompose \p V used with a signed predicate.
  if (IsSigned) {
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
      if (CanUseSExt(CI))
        return {{CI->getSExtValue(), nullptr}};
    }

    return {{0, nullptr}, {1, V}};
  }

  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->uge(MaxConstraintValue))
      return {};
    return {{CI->getZExtValue(), nullptr}};
  }
  auto *GEP = dyn_cast<GetElementPtrInst>(V);
  if (GEP && GEP->getNumOperands() == 2 && GEP->isInBounds()) {
    Value *Op0, *Op1;
    ConstantInt *CI;

    // If the index is zero-extended, it is guaranteed to be positive.
    if (match(GEP->getOperand(GEP->getNumOperands() - 1),
              m_ZExt(m_Value(Op0)))) {
      if (match(Op0, m_NUWShl(m_Value(Op1), m_ConstantInt(CI))) &&
          CanUseSExt(CI))
        return {{0, nullptr},
                {1, GEP->getPointerOperand()},
                {std::pow(int64_t(2), CI->getSExtValue()), Op1}};
      if (match(Op0, m_NSWAdd(m_Value(Op1), m_ConstantInt(CI))) &&
          CanUseSExt(CI))
        return {{CI->getSExtValue(), nullptr},
                {1, GEP->getPointerOperand()},
                {1, Op1}};
      return {{0, nullptr}, {1, GEP->getPointerOperand()}, {1, Op0}};
    }

    if (match(GEP->getOperand(GEP->getNumOperands() - 1), m_ConstantInt(CI)) &&
        !CI->isNegative() && CanUseSExt(CI))
      return {{CI->getSExtValue(), nullptr}, {1, GEP->getPointerOperand()}};

    SmallVector<std::pair<int64_t, Value *>, 4> Result;
    if (match(GEP->getOperand(GEP->getNumOperands() - 1),
              m_NUWShl(m_Value(Op0), m_ConstantInt(CI))) &&
        CanUseSExt(CI))
      Result = {{0, nullptr},
                {1, GEP->getPointerOperand()},
                {std::pow(int64_t(2), CI->getSExtValue()), Op0}};
    else if (match(GEP->getOperand(GEP->getNumOperands() - 1),
                   m_NSWAdd(m_Value(Op0), m_ConstantInt(CI))) &&
             CanUseSExt(CI))
      Result = {{CI->getSExtValue(), nullptr},
                {1, GEP->getPointerOperand()},
                {1, Op0}};
    else {
      Op0 = GEP->getOperand(GEP->getNumOperands() - 1);
      Result = {{0, nullptr}, {1, GEP->getPointerOperand()}, {1, Op0}};
    }
    // If Op0 is signed non-negative, the GEP is increasing monotonically and
    // can be de-composed.
    Preconditions.emplace_back(CmpInst::ICMP_SGE, Op0,
                               ConstantInt::get(Op0->getType(), 0));
    return Result;
  }

  Value *Op0;
  if (match(V, m_ZExt(m_Value(Op0))))
    V = Op0;

  Value *Op1;
  ConstantInt *CI;
  if (match(V, m_NUWAdd(m_Value(Op0), m_ConstantInt(CI))) &&
      !CI->uge(MaxConstraintValue))
    return {{CI->getZExtValue(), nullptr}, {1, Op0}};
  if (match(V, m_Add(m_Value(Op0), m_ConstantInt(CI))) && CI->isNegative() &&
      CanUseSExt(CI)) {
    Preconditions.emplace_back(
        CmpInst::ICMP_UGE, Op0,
        ConstantInt::get(Op0->getType(), CI->getSExtValue() * -1));
    return {{CI->getSExtValue(), nullptr}, {1, Op0}};
  }
  if (match(V, m_NUWAdd(m_Value(Op0), m_Value(Op1))))
    return {{0, nullptr}, {1, Op0}, {1, Op1}};

  if (match(V, m_NUWSub(m_Value(Op0), m_ConstantInt(CI))) && CanUseSExt(CI))
    return {{-1 * CI->getSExtValue(), nullptr}, {1, Op0}};
  if (match(V, m_NUWSub(m_Value(Op0), m_Value(Op1))))
    return {{0, nullptr}, {1, Op0}, {-1, Op1}};

  return {{0, nullptr}, {1, V}};
}

/// Turn a condition \p CmpI into a vector of constraints, using indices from \p
/// Value2Index. Additional indices for newly discovered values are added to \p
/// NewIndices.
static ConstraintTy
getConstraint(CmpInst::Predicate Pred, Value *Op0, Value *Op1,
              const DenseMap<Value *, unsigned> &Value2Index,
              DenseMap<Value *, unsigned> &NewIndices) {
  bool IsEq = false;
  // Try to convert Pred to one of ULE/SLT/SLE/SLT.
  switch (Pred) {
  case CmpInst::ICMP_UGT:
  case CmpInst::ICMP_UGE:
  case CmpInst::ICMP_SGT:
  case CmpInst::ICMP_SGE: {
    Pred = CmpInst::getSwappedPredicate(Pred);
    std::swap(Op0, Op1);
    break;
  }
  case CmpInst::ICMP_EQ:
    if (match(Op1, m_Zero())) {
      Pred = CmpInst::ICMP_ULE;
    } else {
      IsEq = true;
      Pred = CmpInst::ICMP_ULE;
    }
    break;
  case CmpInst::ICMP_NE:
    if (!match(Op1, m_Zero()))
      return {};
    Pred = CmpInst::getSwappedPredicate(CmpInst::ICMP_UGT);
    std::swap(Op0, Op1);
    break;
  default:
    break;
  }

  // Only ULE and ULT predicates are supported at the moment.
  if (Pred != CmpInst::ICMP_ULE && Pred != CmpInst::ICMP_ULT &&
      Pred != CmpInst::ICMP_SLE && Pred != CmpInst::ICMP_SLT)
    return {};

  SmallVector<PreconditionTy, 4> Preconditions;
  bool IsSigned = CmpInst::isSigned(Pred);
  auto ADec = decompose(Op0->stripPointerCastsSameRepresentation(),
                        Preconditions, IsSigned);
  auto BDec = decompose(Op1->stripPointerCastsSameRepresentation(),
                        Preconditions, IsSigned);
  // Skip if decomposing either of the values failed.
  if (ADec.empty() || BDec.empty())
    return {};

  int64_t Offset1 = ADec[0].first;
  int64_t Offset2 = BDec[0].first;
  Offset1 *= -1;

  // Create iterator ranges that skip the constant-factor.
  auto VariablesA = llvm::drop_begin(ADec);
  auto VariablesB = llvm::drop_begin(BDec);

  // First try to look up \p V in Value2Index and NewIndices. Otherwise add a
  // new entry to NewIndices.
  auto GetOrAddIndex = [&Value2Index, &NewIndices](Value *V) -> unsigned {
    auto V2I = Value2Index.find(V);
    if (V2I != Value2Index.end())
      return V2I->second;
    auto Insert =
        NewIndices.insert({V, Value2Index.size() + NewIndices.size() + 1});
    return Insert.first->second;
  };

  // Make sure all variables have entries in Value2Index or NewIndices.
  for (const auto &KV :
       concat<std::pair<int64_t, Value *>>(VariablesA, VariablesB))
    GetOrAddIndex(KV.second);

  // Build result constraint, by first adding all coefficients from A and then
  // subtracting all coefficients from B.
  ConstraintTy Res(
      SmallVector<int64_t, 8>(Value2Index.size() + NewIndices.size() + 1, 0),
      IsSigned);
  Res.IsEq = IsEq;
  auto &R = Res.Coefficients;
  for (const auto &KV : VariablesA)
    R[GetOrAddIndex(KV.second)] += KV.first;

  for (const auto &KV : VariablesB)
    R[GetOrAddIndex(KV.second)] -= KV.first;

  int64_t OffsetSum;
  if (AddOverflow(Offset1, Offset2, OffsetSum))
    return {};
  if (Pred == (IsSigned ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT))
    if (AddOverflow(OffsetSum, int64_t(-1), OffsetSum))
      return {};
  R[0] = OffsetSum;
  Res.Preconditions = std::move(Preconditions);
  return Res;
}

static ConstraintTy getConstraint(CmpInst *Cmp, ConstraintInfo &Info,
                                  DenseMap<Value *, unsigned> &NewIndices) {
  return getConstraint(
      Cmp->getPredicate(), Cmp->getOperand(0), Cmp->getOperand(1),
      Info.getValue2Index(CmpInst::isSigned(Cmp->getPredicate())), NewIndices);
}

bool ConstraintTy::isValid(const ConstraintInfo &Info) const {
  return Coefficients.size() > 0 &&
         all_of(Preconditions, [&Info](const PreconditionTy &C) {
           DenseMap<Value *, unsigned> NewIndices;
           auto R = getConstraint(
               C.Pred, C.Op0, C.Op1,
               Info.getValue2Index(CmpInst::isSigned(C.Pred)), NewIndices);
           // TODO: properly check NewIndices.
           return NewIndices.empty() && R.Preconditions.empty() && !R.IsEq &&
                  R.size() >= 1 &&
                  Info.getCS(CmpInst::isSigned(C.Pred))
                      .isConditionImplied(R.Coefficients);
         });
}

namespace {
/// Represents either a condition that holds on entry to a block or a basic
/// block, with their respective Dominator DFS in and out numbers.
struct ConstraintOrBlock {
  unsigned NumIn;
  unsigned NumOut;
  bool IsBlock;
  bool Not;
  union {
    BasicBlock *BB;
    CmpInst *Condition;
  };

  ConstraintOrBlock(DomTreeNode *DTN)
      : NumIn(DTN->getDFSNumIn()), NumOut(DTN->getDFSNumOut()), IsBlock(true),
        BB(DTN->getBlock()) {}
  ConstraintOrBlock(DomTreeNode *DTN, CmpInst *Condition, bool Not)
      : NumIn(DTN->getDFSNumIn()), NumOut(DTN->getDFSNumOut()), IsBlock(false),
        Not(Not), Condition(Condition) {}
};

/// Keep state required to build worklist.
struct State {
  DominatorTree &DT;
  SmallVector<ConstraintOrBlock, 64> WorkList;

  State(DominatorTree &DT) : DT(DT) {}

  /// Process block \p BB and add known facts to work-list.
  void addInfoFor(BasicBlock &BB);

  /// Returns true if we can add a known condition from BB to its successor
  /// block Succ. Each predecessor of Succ can either be BB or be dominated
  /// by Succ (e.g. the case when adding a condition from a pre-header to a
  /// loop header).
  bool canAddSuccessor(BasicBlock &BB, BasicBlock *Succ) const {
    if (BB.getSingleSuccessor()) {
      assert(BB.getSingleSuccessor() == Succ);
      return DT.properlyDominates(&BB, Succ);
    }
    return any_of(successors(&BB),
                  [Succ](const BasicBlock *S) { return S != Succ; }) &&
           all_of(predecessors(Succ), [&BB, Succ, this](BasicBlock *Pred) {
             return Pred == &BB || DT.dominates(Succ, Pred);
           });
  }
};

} // namespace

#ifndef NDEBUG
static void dumpWithNames(ConstraintTy &C,
                          DenseMap<Value *, unsigned> &Value2Index) {
  SmallVector<std::string> Names(Value2Index.size(), "");
  for (auto &KV : Value2Index) {
    Names[KV.second - 1] = std::string("%") + KV.first->getName().str();
  }
  ConstraintSystem CS;
  CS.addVariableRowFill(C.Coefficients);
  CS.dump(Names);
}
#endif

void State::addInfoFor(BasicBlock &BB) {
  WorkList.emplace_back(DT.getNode(&BB));

  // True as long as long as the current instruction is guaranteed to execute.
  bool GuaranteedToExecute = true;
  // Scan BB for assume calls.
  // TODO: also use this scan to queue conditions to simplify, so we can
  // interleave facts from assumes and conditions to simplify in a single
  // basic block. And to skip another traversal of each basic block when
  // simplifying.
  for (Instruction &I : BB) {
    Value *Cond;
    // For now, just handle assumes with a single compare as condition.
    if (match(&I, m_Intrinsic<Intrinsic::assume>(m_Value(Cond))) &&
        isa<ICmpInst>(Cond)) {
      if (GuaranteedToExecute) {
        // The assume is guaranteed to execute when BB is entered, hence Cond
        // holds on entry to BB.
        WorkList.emplace_back(DT.getNode(&BB), cast<ICmpInst>(Cond), false);
      } else {
        // Otherwise the condition only holds in the successors.
        for (BasicBlock *Succ : successors(&BB)) {
          if (!canAddSuccessor(BB, Succ))
            continue;
          WorkList.emplace_back(DT.getNode(Succ), cast<ICmpInst>(Cond), false);
        }
      }
    }
    GuaranteedToExecute &= isGuaranteedToTransferExecutionToSuccessor(&I);
  }

  auto *Br = dyn_cast<BranchInst>(BB.getTerminator());
  if (!Br || !Br->isConditional())
    return;

  // If the condition is an OR of 2 compares and the false successor only has
  // the current block as predecessor, queue both negated conditions for the
  // false successor.
  Value *Op0, *Op1;
  if (match(Br->getCondition(), m_LogicalOr(m_Value(Op0), m_Value(Op1))) &&
      isa<ICmpInst>(Op0) && isa<ICmpInst>(Op1)) {
    BasicBlock *FalseSuccessor = Br->getSuccessor(1);
    if (canAddSuccessor(BB, FalseSuccessor)) {
      WorkList.emplace_back(DT.getNode(FalseSuccessor), cast<ICmpInst>(Op0),
                            true);
      WorkList.emplace_back(DT.getNode(FalseSuccessor), cast<ICmpInst>(Op1),
                            true);
    }
    return;
  }

  // If the condition is an AND of 2 compares and the true successor only has
  // the current block as predecessor, queue both conditions for the true
  // successor.
  if (match(Br->getCondition(), m_LogicalAnd(m_Value(Op0), m_Value(Op1))) &&
      isa<ICmpInst>(Op0) && isa<ICmpInst>(Op1)) {
    BasicBlock *TrueSuccessor = Br->getSuccessor(0);
    if (canAddSuccessor(BB, TrueSuccessor)) {
      WorkList.emplace_back(DT.getNode(TrueSuccessor), cast<ICmpInst>(Op0),
                            false);
      WorkList.emplace_back(DT.getNode(TrueSuccessor), cast<ICmpInst>(Op1),
                            false);
    }
    return;
  }

  auto *CmpI = dyn_cast<ICmpInst>(Br->getCondition());
  if (!CmpI)
    return;
  if (canAddSuccessor(BB, Br->getSuccessor(0)))
    WorkList.emplace_back(DT.getNode(Br->getSuccessor(0)), CmpI, false);
  if (canAddSuccessor(BB, Br->getSuccessor(1)))
    WorkList.emplace_back(DT.getNode(Br->getSuccessor(1)), CmpI, true);
}

static void
tryToSimplifyOverflowMath(IntrinsicInst *II, ConstraintInfo &Info,
                          SmallVectorImpl<Instruction *> &ToRemove) {
  auto DoesConditionHold = [](CmpInst::Predicate Pred, Value *A, Value *B,
                              ConstraintInfo &Info) {
    DenseMap<Value *, unsigned> NewIndices;
    auto R = getConstraint(
        Pred, A, B, Info.getValue2Index(CmpInst::isSigned(Pred)), NewIndices);
    if (R.size() < 2 || R.needsNewIndices(NewIndices) || !R.isValid(Info))
      return false;

    auto &CSToUse = Info.getCS(CmpInst::isSigned(Pred));
    return CSToUse.isConditionImplied(R.Coefficients);
  };

  if (II->getIntrinsicID() == Intrinsic::ssub_with_overflow) {
    // If A s>= B && B s>= 0, ssub.with.overflow(a, b) should not overflow and
    // can be simplified to a regular sub.
    Value *A = II->getArgOperand(0);
    Value *B = II->getArgOperand(1);
    if (!DoesConditionHold(CmpInst::ICMP_SGE, A, B, Info) ||
        !DoesConditionHold(CmpInst::ICMP_SGE, B,
                           ConstantInt::get(A->getType(), 0), Info))
      return;

    IRBuilder<> Builder(II->getParent(), II->getIterator());
    Value *Sub = nullptr;
    for (User *U : make_early_inc_range(II->users())) {
      if (match(U, m_ExtractValue<0>(m_Value()))) {
        if (!Sub)
          Sub = Builder.CreateSub(A, B);
        U->replaceAllUsesWith(Sub);
      } else if (match(U, m_ExtractValue<1>(m_Value())))
        U->replaceAllUsesWith(Builder.getFalse());
      else
        continue;

      if (U->use_empty()) {
        auto *I = cast<Instruction>(U);
        ToRemove.push_back(I);
        I->setOperand(0, PoisonValue::get(II->getType()));
      }
    }

    if (II->use_empty())
      II->eraseFromParent();
  }
}

static bool eliminateConstraints(Function &F, DominatorTree &DT) {
  bool Changed = false;
  DT.updateDFSNumbers();

  ConstraintInfo Info;
  State S(DT);

  // First, collect conditions implied by branches and blocks with their
  // Dominator DFS in and out numbers.
  for (BasicBlock &BB : F) {
    if (!DT.getNode(&BB))
      continue;
    S.addInfoFor(BB);
  }

  // Next, sort worklist by dominance, so that dominating blocks and conditions
  // come before blocks and conditions dominated by them. If a block and a
  // condition have the same numbers, the condition comes before the block, as
  // it holds on entry to the block.
  sort(S.WorkList, [](const ConstraintOrBlock &A, const ConstraintOrBlock &B) {
    return std::tie(A.NumIn, A.IsBlock) < std::tie(B.NumIn, B.IsBlock);
  });

  SmallVector<Instruction *> ToRemove;

  // Finally, process ordered worklist and eliminate implied conditions.
  SmallVector<StackEntry, 16> DFSInStack;
  for (ConstraintOrBlock &CB : S.WorkList) {
    // First, pop entries from the stack that are out-of-scope for CB. Remove
    // the corresponding entry from the constraint system.
    while (!DFSInStack.empty()) {
      auto &E = DFSInStack.back();
      LLVM_DEBUG(dbgs() << "Top of stack : " << E.NumIn << " " << E.NumOut
                        << "\n");
      LLVM_DEBUG(dbgs() << "CB: " << CB.NumIn << " " << CB.NumOut << "\n");
      assert(E.NumIn <= CB.NumIn);
      if (CB.NumOut <= E.NumOut)
        break;
      LLVM_DEBUG(dbgs() << "Removing " << *E.Condition << " " << E.IsNot
                        << "\n");
      Info.popLastConstraint(E.IsSigned);
      // Remove variables in the system that went out of scope.
      auto &Mapping = Info.getValue2Index(E.IsSigned);
      for (Value *V : E.ValuesToRelease)
        Mapping.erase(V);
      Info.popLastNVariables(E.IsSigned, E.ValuesToRelease.size());
      DFSInStack.pop_back();
    }

    LLVM_DEBUG({
      dbgs() << "Processing ";
      if (CB.IsBlock)
        dbgs() << *CB.BB;
      else
        dbgs() << *CB.Condition;
      dbgs() << "\n";
    });

    // For a block, check if any CmpInsts become known based on the current set
    // of constraints.
    if (CB.IsBlock) {
      for (Instruction &I : make_early_inc_range(*CB.BB)) {
        if (auto *II = dyn_cast<WithOverflowInst>(&I)) {
          tryToSimplifyOverflowMath(II, Info, ToRemove);
          continue;
        }
        auto *Cmp = dyn_cast<ICmpInst>(&I);
        if (!Cmp)
          continue;

        DenseMap<Value *, unsigned> NewIndices;
        auto R = getConstraint(Cmp, Info, NewIndices);
        if (R.IsEq || R.empty() || R.needsNewIndices(NewIndices) ||
            !R.isValid(Info))
          continue;

        auto &CSToUse = Info.getCS(R.IsSigned);
        if (CSToUse.isConditionImplied(R.Coefficients)) {
          if (!DebugCounter::shouldExecute(EliminatedCounter))
            continue;

          LLVM_DEBUG(dbgs() << "Condition " << *Cmp
                            << " implied by dominating constraints\n");
          LLVM_DEBUG({
            for (auto &E : reverse(DFSInStack))
              dbgs() << "   C " << *E.Condition << " " << E.IsNot << "\n";
          });
          Cmp->replaceUsesWithIf(
              ConstantInt::getTrue(F.getParent()->getContext()), [](Use &U) {
                // Conditions in an assume trivially simplify to true. Skip uses
                // in assume calls to not destroy the available information.
                auto *II = dyn_cast<IntrinsicInst>(U.getUser());
                return !II || II->getIntrinsicID() != Intrinsic::assume;
              });
          NumCondsRemoved++;
          Changed = true;
        }
        if (CSToUse.isConditionImplied(
                ConstraintSystem::negate(R.Coefficients))) {
          if (!DebugCounter::shouldExecute(EliminatedCounter))
            continue;

          LLVM_DEBUG(dbgs() << "Condition !" << *Cmp
                            << " implied by dominating constraints\n");
          LLVM_DEBUG({
            for (auto &E : reverse(DFSInStack))
              dbgs() << "   C " << *E.Condition << " " << E.IsNot << "\n";
          });
          Cmp->replaceAllUsesWith(
              ConstantInt::getFalse(F.getParent()->getContext()));
          NumCondsRemoved++;
          Changed = true;
        }
      }
      continue;
    }

    // Set up a function to restore the predicate at the end of the scope if it
    // has been negated. Negate the predicate in-place, if required.
    auto *CI = dyn_cast<ICmpInst>(CB.Condition);
    auto PredicateRestorer = make_scope_exit([CI, &CB]() {
      if (CB.Not && CI)
        CI->setPredicate(CI->getInversePredicate());
    });
    if (CB.Not) {
      if (CI) {
        CI->setPredicate(CI->getInversePredicate());
      } else {
        LLVM_DEBUG(dbgs() << "Can only negate compares so far.\n");
        continue;
      }
    }

    // Otherwise, add the condition to the system and stack, if we can transform
    // it into a constraint.
    DenseMap<Value *, unsigned> NewIndices;
    auto R = getConstraint(CB.Condition, Info, NewIndices);
    if (!R.isValid(Info))
      continue;

    LLVM_DEBUG(dbgs() << "Adding " << *CB.Condition << " " << CB.Not << "\n");
    bool Added = false;
    assert(CmpInst::isSigned(CB.Condition->getPredicate()) == R.IsSigned &&
           "condition and constraint signs must match");
    auto &CSToUse = Info.getCS(R.IsSigned);
    if (R.Coefficients.empty())
      continue;

    Added |= CSToUse.addVariableRowFill(R.Coefficients);

    // If R has been added to the system, queue it for removal once it goes
    // out-of-scope.
    if (Added) {
      SmallVector<Value *, 2> ValuesToRelease;
      for (auto &KV : NewIndices) {
        Info.getValue2Index(R.IsSigned).insert(KV);
        ValuesToRelease.push_back(KV.first);
      }

      LLVM_DEBUG({
        dbgs() << "  constraint: ";
        dumpWithNames(R, Info.getValue2Index(R.IsSigned));
      });

      DFSInStack.emplace_back(CB.NumIn, CB.NumOut, CB.Condition, CB.Not,
                              R.IsSigned, ValuesToRelease);

      if (R.IsEq) {
        // Also add the inverted constraint for equality constraints.
        for (auto &Coeff : R.Coefficients)
          Coeff *= -1;
        CSToUse.addVariableRowFill(R.Coefficients);

        DFSInStack.emplace_back(CB.NumIn, CB.NumOut, CB.Condition, CB.Not,
                                R.IsSigned, SmallVector<Value *, 2>());
      }
    }
  }

#ifndef NDEBUG
  unsigned SignedEntries =
      count_if(DFSInStack, [](const StackEntry &E) { return E.IsSigned; });
  assert(Info.getCS(false).size() == DFSInStack.size() - SignedEntries &&
         "updates to CS and DFSInStack are out of sync");
  assert(Info.getCS(true).size() == SignedEntries &&
         "updates to CS and DFSInStack are out of sync");
#endif

  for (Instruction *I : ToRemove)
    I->eraseFromParent();
  return Changed;
}

PreservedAnalyses ConstraintEliminationPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  if (!eliminateConstraints(F, DT))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

namespace {

class ConstraintElimination : public FunctionPass {
public:
  static char ID;

  ConstraintElimination() : FunctionPass(ID) {
    initializeConstraintEliminationPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    return eliminateConstraints(F, DT);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
};

} // end anonymous namespace

char ConstraintElimination::ID = 0;

INITIALIZE_PASS_BEGIN(ConstraintElimination, "constraint-elimination",
                      "Constraint Elimination", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LazyValueInfoWrapperPass)
INITIALIZE_PASS_END(ConstraintElimination, "constraint-elimination",
                    "Constraint Elimination", false, false)

FunctionPass *llvm::createConstraintEliminationPass() {
  return new ConstraintElimination();
}
