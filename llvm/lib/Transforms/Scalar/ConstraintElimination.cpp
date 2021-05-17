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
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Transforms/Scalar.h"

#include <string>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "constraint-elimination"

STATISTIC(NumCondsRemoved, "Number of instructions removed");
DEBUG_COUNTER(EliminatedCounter, "conds-eliminated",
              "Controls which conditions are eliminated");

static int64_t MaxConstraintValue = std::numeric_limits<int64_t>::max();

// Decomposes \p V into a vector of pairs of the form { c, X } where c * X. The
// sum of the pairs equals \p V.  The first pair is the constant-factor and X
// must be nullptr. If the expression cannot be decomposed, returns an empty
// vector.
static SmallVector<std::pair<int64_t, Value *>, 4> decompose(Value *V) {
  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->isNegative() || CI->uge(MaxConstraintValue))
      return {};
    return {{CI->getSExtValue(), nullptr}};
  }
  auto *GEP = dyn_cast<GetElementPtrInst>(V);
  if (GEP && GEP->getNumOperands() == 2 && GEP->isInBounds()) {
    Value *Op0, *Op1;
    ConstantInt *CI;

    // If the index is zero-extended, it is guaranteed to be positive.
    if (match(GEP->getOperand(GEP->getNumOperands() - 1),
              m_ZExt(m_Value(Op0)))) {
      if (match(Op0, m_NUWShl(m_Value(Op1), m_ConstantInt(CI))))
        return {{0, nullptr},
                {1, GEP->getPointerOperand()},
                {std::pow(int64_t(2), CI->getSExtValue()), Op1}};
      if (match(Op0, m_NSWAdd(m_Value(Op1), m_ConstantInt(CI))))
        return {{CI->getSExtValue(), nullptr},
                {1, GEP->getPointerOperand()},
                {1, Op1}};
      return {{0, nullptr}, {1, GEP->getPointerOperand()}, {1, Op0}};
    }

    if (match(GEP->getOperand(GEP->getNumOperands() - 1), m_ConstantInt(CI)) &&
        !CI->isNegative())
      return {{CI->getSExtValue(), nullptr}, {1, GEP->getPointerOperand()}};

    SmallVector<std::pair<int64_t, Value *>, 4> Result;
    if (match(GEP->getOperand(GEP->getNumOperands() - 1),
              m_NUWShl(m_Value(Op0), m_ConstantInt(CI))))
      Result = {{0, nullptr},
                {1, GEP->getPointerOperand()},
                {std::pow(int64_t(2), CI->getSExtValue()), Op0}};
    else if (match(GEP->getOperand(GEP->getNumOperands() - 1),
                   m_NSWAdd(m_Value(Op0), m_ConstantInt(CI))))
      Result = {{CI->getSExtValue(), nullptr},
                {1, GEP->getPointerOperand()},
                {1, Op0}};
    else {
      Op0 = GEP->getOperand(GEP->getNumOperands() - 1);
      Result = {{0, nullptr}, {1, GEP->getPointerOperand()}, {1, Op0}};
    }
    return Result;
  }

  Value *Op0;
  if (match(V, m_ZExt(m_Value(Op0))))
    V = Op0;

  Value *Op1;
  ConstantInt *CI;
  if (match(V, m_NUWAdd(m_Value(Op0), m_ConstantInt(CI))))
    return {{CI->getSExtValue(), nullptr}, {1, Op0}};
  if (match(V, m_NUWAdd(m_Value(Op0), m_Value(Op1))))
    return {{0, nullptr}, {1, Op0}, {1, Op1}};

  if (match(V, m_NUWSub(m_Value(Op0), m_ConstantInt(CI))))
    return {{-1 * CI->getSExtValue(), nullptr}, {1, Op0}};
  if (match(V, m_NUWSub(m_Value(Op0), m_Value(Op1))))
    return {{0, nullptr}, {1, Op0}, {1, Op1}};

  return {{0, nullptr}, {1, V}};
}

struct ConstraintTy {
  SmallVector<int64_t, 8> Coefficients;

  ConstraintTy(SmallVector<int64_t, 8> Coefficients)
      : Coefficients(Coefficients) {}

  unsigned size() const { return Coefficients.size(); }
};

/// Turn a condition \p CmpI into a vector of constraints, using indices from \p
/// Value2Index. Additional indices for newly discovered values are added to \p
/// NewIndices.
static SmallVector<ConstraintTy, 4>
getConstraint(CmpInst::Predicate Pred, Value *Op0, Value *Op1,
              const DenseMap<Value *, unsigned> &Value2Index,
              DenseMap<Value *, unsigned> &NewIndices) {
  int64_t Offset1 = 0;
  int64_t Offset2 = 0;

  // First try to look up \p V in Value2Index and NewIndices. Otherwise add a
  // new entry to NewIndices.
  auto GetOrAddIndex = [&Value2Index, &NewIndices](Value *V) -> unsigned {
    auto V2I = Value2Index.find(V);
    if (V2I != Value2Index.end())
      return V2I->second;
    auto NewI = NewIndices.find(V);
    if (NewI != NewIndices.end())
      return NewI->second;
    auto Insert =
        NewIndices.insert({V, Value2Index.size() + NewIndices.size() + 1});
    return Insert.first->second;
  };

  if (Pred == CmpInst::ICMP_UGT || Pred == CmpInst::ICMP_UGE)
    return getConstraint(CmpInst::getSwappedPredicate(Pred), Op1, Op0,
                         Value2Index, NewIndices);

  if (Pred == CmpInst::ICMP_EQ) {
    auto A =
        getConstraint(CmpInst::ICMP_UGE, Op0, Op1, Value2Index, NewIndices);
    auto B =
        getConstraint(CmpInst::ICMP_ULE, Op0, Op1, Value2Index, NewIndices);
    append_range(A, B);
    return A;
  }

  if (Pred == CmpInst::ICMP_NE && match(Op1, m_Zero())) {
    return getConstraint(CmpInst::ICMP_UGT, Op0, Op1, Value2Index, NewIndices);
  }

  // Only ULE and ULT predicates are supported at the moment.
  if (Pred != CmpInst::ICMP_ULE && Pred != CmpInst::ICMP_ULT)
    return {};

  auto ADec = decompose(Op0->stripPointerCastsSameRepresentation());
  auto BDec = decompose(Op1->stripPointerCastsSameRepresentation());
  // Skip if decomposing either of the values failed.
  if (ADec.empty() || BDec.empty())
    return {};

  // Skip trivial constraints without any variables.
  if (ADec.size() == 1 && BDec.size() == 1)
    return {};

  Offset1 = ADec[0].first;
  Offset2 = BDec[0].first;
  Offset1 *= -1;

  // Create iterator ranges that skip the constant-factor.
  auto VariablesA = llvm::drop_begin(ADec);
  auto VariablesB = llvm::drop_begin(BDec);

  // Make sure all variables have entries in Value2Index or NewIndices.
  for (const auto &KV :
       concat<std::pair<int64_t, Value *>>(VariablesA, VariablesB))
    GetOrAddIndex(KV.second);

  // Build result constraint, by first adding all coefficients from A and then
  // subtracting all coefficients from B.
  SmallVector<int64_t, 8> R(Value2Index.size() + NewIndices.size() + 1, 0);
  for (const auto &KV : VariablesA)
    R[GetOrAddIndex(KV.second)] += KV.first;

  for (const auto &KV : VariablesB)
    R[GetOrAddIndex(KV.second)] -= KV.first;

  R[0] = Offset1 + Offset2 + (Pred == CmpInst::ICMP_ULT ? -1 : 0);
  return {R};
}

static SmallVector<ConstraintTy, 4>
getConstraint(CmpInst *Cmp, const DenseMap<Value *, unsigned> &Value2Index,
              DenseMap<Value *, unsigned> &NewIndices) {
  return getConstraint(Cmp->getPredicate(), Cmp->getOperand(0),
                       Cmp->getOperand(1), Value2Index, NewIndices);
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

struct StackEntry {
  unsigned NumIn;
  unsigned NumOut;
  CmpInst *Condition;
  bool IsNot;

  StackEntry(unsigned NumIn, unsigned NumOut, CmpInst *Condition, bool IsNot)
      : NumIn(NumIn), NumOut(NumOut), Condition(Condition), IsNot(IsNot) {}
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

static bool eliminateConstraints(Function &F, DominatorTree &DT) {
  bool Changed = false;
  DT.updateDFSNumbers();
  ConstraintSystem CS;

  SmallVector<ConstraintOrBlock, 64> WorkList;

  // First, collect conditions implied by branches and blocks with their
  // Dominator DFS in and out numbers.
  for (BasicBlock &BB : F) {
    if (!DT.getNode(&BB))
      continue;
    WorkList.emplace_back(DT.getNode(&BB));

    auto *Br = dyn_cast<BranchInst>(BB.getTerminator());
    if (!Br || !Br->isConditional())
      continue;

    // Returns true if we can add a known condition from BB to its successor
    // block Succ. Each predecessor of Succ can either be BB or be dominated by
    // Succ (e.g. the case when adding a condition from a pre-header to a loop
    // header).
    auto CanAdd = [&BB, &DT](BasicBlock *Succ) {
      return all_of(predecessors(Succ), [&BB, &DT, Succ](BasicBlock *Pred) {
        return Pred == &BB || DT.dominates(Succ, Pred);
      });
    };
    // If the condition is an OR of 2 compares and the false successor only has
    // the current block as predecessor, queue both negated conditions for the
    // false successor.
    Value *Op0, *Op1;
    if (match(Br->getCondition(), m_LogicalOr(m_Value(Op0), m_Value(Op1))) &&
        match(Op0, m_Cmp()) && match(Op1, m_Cmp())) {
      BasicBlock *FalseSuccessor = Br->getSuccessor(1);
      if (CanAdd(FalseSuccessor)) {
        WorkList.emplace_back(DT.getNode(FalseSuccessor), cast<CmpInst>(Op0),
                              true);
        WorkList.emplace_back(DT.getNode(FalseSuccessor), cast<CmpInst>(Op1),
                              true);
      }
      continue;
    }

    // If the condition is an AND of 2 compares and the true successor only has
    // the current block as predecessor, queue both conditions for the true
    // successor.
    if (match(Br->getCondition(), m_LogicalAnd(m_Value(Op0), m_Value(Op1))) &&
        match(Op0, m_Cmp()) && match(Op1, m_Cmp())) {
      BasicBlock *TrueSuccessor = Br->getSuccessor(0);
      if (CanAdd(TrueSuccessor)) {
        WorkList.emplace_back(DT.getNode(TrueSuccessor), cast<CmpInst>(Op0),
                              false);
        WorkList.emplace_back(DT.getNode(TrueSuccessor), cast<CmpInst>(Op1),
                              false);
      }
      continue;
    }

    auto *CmpI = dyn_cast<CmpInst>(Br->getCondition());
    if (!CmpI)
      continue;
    if (CanAdd(Br->getSuccessor(0)))
      WorkList.emplace_back(DT.getNode(Br->getSuccessor(0)), CmpI, false);
    if (CanAdd(Br->getSuccessor(1)))
      WorkList.emplace_back(DT.getNode(Br->getSuccessor(1)), CmpI, true);
  }

  // Next, sort worklist by dominance, so that dominating blocks and conditions
  // come before blocks and conditions dominated by them. If a block and a
  // condition have the same numbers, the condition comes before the block, as
  // it holds on entry to the block.
  sort(WorkList, [](const ConstraintOrBlock &A, const ConstraintOrBlock &B) {
    return std::tie(A.NumIn, A.IsBlock) < std::tie(B.NumIn, B.IsBlock);
  });

  // Finally, process ordered worklist and eliminate implied conditions.
  SmallVector<StackEntry, 16> DFSInStack;
  DenseMap<Value *, unsigned> Value2Index;
  for (ConstraintOrBlock &CB : WorkList) {
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
      DFSInStack.pop_back();
      CS.popLastConstraint();
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
      for (Instruction &I : *CB.BB) {
        auto *Cmp = dyn_cast<CmpInst>(&I);
        if (!Cmp)
          continue;

        DenseMap<Value *, unsigned> NewIndices;
        auto R = getConstraint(Cmp, Value2Index, NewIndices);
        if (R.size() != 1)
          continue;

        // Check if all coefficients of new indices are 0 after building the
        // constraint. Skip if any of the new indices has a non-null
        // coefficient.
        bool HasNewIndex = false;
        for (unsigned I = 0; I < NewIndices.size(); ++I) {
          int64_t Last = R[0].Coefficients.pop_back_val();
          if (Last != 0) {
            HasNewIndex = true;
            break;
          }
        }
        if (HasNewIndex || R[0].size() == 1)
          continue;

        if (CS.isConditionImplied(R[0].Coefficients)) {
          if (!DebugCounter::shouldExecute(EliminatedCounter))
            continue;

          LLVM_DEBUG(dbgs() << "Condition " << *Cmp
                            << " implied by dominating constraints\n");
          LLVM_DEBUG({
            for (auto &E : reverse(DFSInStack))
              dbgs() << "   C " << *E.Condition << " " << E.IsNot << "\n";
          });
          Cmp->replaceAllUsesWith(
              ConstantInt::getTrue(F.getParent()->getContext()));
          NumCondsRemoved++;
          Changed = true;
        }
        if (CS.isConditionImplied(
                ConstraintSystem::negate(R[0].Coefficients))) {
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
    auto *CI = dyn_cast<CmpInst>(CB.Condition);
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
    auto R = getConstraint(CB.Condition, Value2Index, NewIndices);
    if (R.empty())
      continue;

    for (auto &KV : NewIndices)
      Value2Index.insert(KV);

    LLVM_DEBUG(dbgs() << "Adding " << *CB.Condition << " " << CB.Not << "\n");
    bool Added = false;
    for (auto &C : R) {
      auto Coeffs = C.Coefficients;
      LLVM_DEBUG({
        dbgs() << "  constraint: ";
        dumpWithNames(C, Value2Index);
      });
      Added |= CS.addVariableRowFill(Coeffs);
      // If R has been added to the system, queue it for removal once it goes
      // out-of-scope.
      if (Added)
        DFSInStack.emplace_back(CB.NumIn, CB.NumOut, CB.Condition, CB.Not);
    }
  }

  assert(CS.size() == DFSInStack.size() &&
         "updates to CS and DFSInStack are out of sync");
  return Changed;
}

PreservedAnalyses ConstraintEliminationPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  if (!eliminateConstraints(F, DT))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<GlobalsAA>();
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
