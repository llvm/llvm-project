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
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Scalar.h"

#include <cmath>
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
  bool IsSigned = false;
  /// Variables that can be removed from the system once the stack entry gets
  /// removed.
  SmallVector<Value *, 2> ValuesToRelease;

  StackEntry(unsigned NumIn, unsigned NumOut, bool IsSigned,
             SmallVector<Value *, 2> ValuesToRelease)
      : NumIn(NumIn), NumOut(NumOut), IsSigned(IsSigned),
        ValuesToRelease(ValuesToRelease) {}
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

  SmallVector<SmallVector<int64_t, 8>> ExtraInfo;

  bool IsSigned = false;
  bool IsEq = false;

  ConstraintTy() = default;

  ConstraintTy(SmallVector<int64_t, 8> Coefficients, bool IsSigned)
      : Coefficients(Coefficients), IsSigned(IsSigned) {}

  unsigned size() const { return Coefficients.size(); }

  unsigned empty() const { return Coefficients.empty(); }

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

  const DataLayout &DL;

public:
  ConstraintInfo(const DataLayout &DL) : DL(DL) {}

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

  bool doesHold(CmpInst::Predicate Pred, Value *A, Value *B) const;

  void addFact(CmpInst::Predicate Pred, Value *A, Value *B, unsigned NumIn,
               unsigned NumOut, SmallVectorImpl<StackEntry> &DFSInStack);

  /// Turn a comparison of the form \p Op0 \p Pred \p Op1 into a vector of
  /// constraints, using indices from the corresponding constraint system.
  /// New variables that need to be added to the system are collected in
  /// \p NewVariables.
  ConstraintTy getConstraint(CmpInst::Predicate Pred, Value *Op0, Value *Op1,
                             SmallVectorImpl<Value *> &NewVariables) const;

  /// Turns a comparison of the form \p Op0 \p Pred \p Op1 into a vector of
  /// constraints using getConstraint. Returns an empty constraint if the result
  /// cannot be used to query the existing constraint system, e.g. because it
  /// would require adding new variables. Also tries to convert signed
  /// predicates to unsigned ones if possible to allow using the unsigned system
  /// which increases the effectiveness of the signed <-> unsigned transfer
  /// logic.
  ConstraintTy getConstraintForSolving(CmpInst::Predicate Pred, Value *Op0,
                                       Value *Op1) const;

  /// Try to add information from \p A \p Pred \p B to the unsigned/signed
  /// system if \p Pred is signed/unsigned.
  void transferToOtherSystem(CmpInst::Predicate Pred, Value *A, Value *B,
                             unsigned NumIn, unsigned NumOut,
                             SmallVectorImpl<StackEntry> &DFSInStack);
};

/// Represents a (Coefficient * Variable) entry after IR decomposition.
struct DecompEntry {
  int64_t Coefficient;
  Value *Variable;
  /// True if the variable is known positive in the current constraint.
  bool IsKnownPositive;

  DecompEntry(int64_t Coefficient, Value *Variable,
              bool IsKnownPositive = false)
      : Coefficient(Coefficient), Variable(Variable),
        IsKnownPositive(IsKnownPositive) {}
};

} // namespace

static SmallVector<DecompEntry, 4>
decompose(Value *V, SmallVector<PreconditionTy, 4> &Preconditions,
          bool IsSigned, const DataLayout &DL);

static bool canUseSExt(ConstantInt *CI) {
  const APInt &Val = CI->getValue();
  return Val.sgt(MinSignedConstraintValue) && Val.slt(MaxConstraintValue);
}

// A helper to multiply 2 signed integers where overflowing is allowed.
static int64_t multiplyWithOverflow(int64_t A, int64_t B) {
  int64_t Result;
  MulOverflow(A, B, Result);
  return Result;
}

// A helper to add 2 signed integers where overflowing is allowed.
static int64_t addWithOverflow(int64_t A, int64_t B) {
  int64_t Result;
  AddOverflow(A, B, Result);
  return Result;
}

static SmallVector<DecompEntry, 4>
decomposeGEP(GetElementPtrInst &GEP,
             SmallVector<PreconditionTy, 4> &Preconditions, bool IsSigned,
             const DataLayout &DL) {
  // Do not reason about pointers where the index size is larger than 64 bits,
  // as the coefficients used to encode constraints are 64 bit integers.
  unsigned AS =
      cast<PointerType>(GEP.getPointerOperand()->getType())->getAddressSpace();
  if (DL.getIndexSizeInBits(AS) > 64)
    return {};

  if (!GEP.isInBounds())
    return {{0, nullptr}, {1, &GEP}};

  // Handle the (gep (gep ....), C) case by incrementing the constant
  // coefficient of the inner GEP, if C is a constant.
  auto *InnerGEP = dyn_cast<GetElementPtrInst>(GEP.getPointerOperand());
  if (InnerGEP && GEP.getNumOperands() == 2 &&
      isa<ConstantInt>(GEP.getOperand(1))) {
    APInt Offset = cast<ConstantInt>(GEP.getOperand(1))->getValue();
    auto Result = decompose(InnerGEP, Preconditions, IsSigned, DL);

    auto GTI = gep_type_begin(GEP);
    // Bail out for scalable vectors for now.
    if (isa<ScalableVectorType>(GTI.getIndexedType()))
      return {};
    int64_t Scale = static_cast<int64_t>(
        DL.getTypeAllocSize(GTI.getIndexedType()).getFixedSize());

    Result[0].Coefficient =
        addWithOverflow(Result[0].Coefficient,
                        multiplyWithOverflow(Scale, Offset.getSExtValue()));
    if (Offset.isNegative()) {
      // Add pre-condition ensuring the GEP is increasing monotonically and
      // can be de-composed.
      Preconditions.emplace_back(
          CmpInst::ICMP_SGE, InnerGEP->getOperand(1),
          ConstantInt::get(InnerGEP->getOperand(1)->getType(),
                           -1 * Offset.getSExtValue()));
    }
    return Result;
  }

  SmallVector<DecompEntry, 4> Result = {{0, nullptr},
                                        {1, GEP.getPointerOperand()}};
  gep_type_iterator GTI = gep_type_begin(GEP);
  for (User::const_op_iterator I = GEP.op_begin() + 1, E = GEP.op_end(); I != E;
       ++I, ++GTI) {
    Value *Index = *I;

    // Bail out for scalable vectors for now.
    if (isa<ScalableVectorType>(GTI.getIndexedType()))
      return {};

    // Struct indices must be constants (and reference an existing field). Add
    // them to the constant factor.
    if (StructType *STy = GTI.getStructTypeOrNull()) {
      // For a struct, add the member offset.
      unsigned FieldNo = cast<ConstantInt>(Index)->getZExtValue();
      if (FieldNo == 0)
        continue;

      // Add offset to constant factor.
      Result[0].Coefficient = addWithOverflow(
          Result[0].Coefficient,
          int64_t(DL.getStructLayout(STy)->getElementOffset(FieldNo)));
      continue;
    }

    // For an array/pointer, add the element offset, explicitly scaled.
    unsigned Scale = DL.getTypeAllocSize(GTI.getIndexedType()).getFixedSize();

    auto IdxResult = decompose(Index, Preconditions, IsSigned, DL);
    if (IdxResult.empty()) {
      Result.emplace_back(Scale, Index);
    } else {
      for (auto &KV : IdxResult)
        KV.Coefficient = multiplyWithOverflow(KV.Coefficient, Scale);
      Result[0].Coefficient += IdxResult[0].Coefficient;
      append_range(Result, ArrayRef<DecompEntry>(IdxResult).drop_front());
    }
    // If Op0 is signed non-negative, the GEP is increasing monotonically and
    // can be de-composed.
    if (!isKnownNonNegative(Index, DL, /*Depth=*/MaxAnalysisRecursionDepth - 1))
      Preconditions.emplace_back(CmpInst::ICMP_SGE, Index,
                                 ConstantInt::get(Index->getType(), 0));
  }
  return Result;
}

// Decomposes \p V into a vector of entries of the form { Coefficient, Variable
// } where Coefficient * Variable. The sum of the pairs equals \p V.  The first
// pair is the constant-factor and X must be nullptr. If the expression cannot
// be decomposed, returns an empty vector.
static SmallVector<DecompEntry, 4>
decompose(Value *V, SmallVector<PreconditionTy, 4> &Preconditions,
          bool IsSigned, const DataLayout &DL) {

  auto MergeResults = [&Preconditions, IsSigned,
                       &DL](Value *A, Value *B,
                            bool IsSignedB) -> SmallVector<DecompEntry, 4> {
    auto ResA = decompose(A, Preconditions, IsSigned, DL);
    auto ResB = decompose(B, Preconditions, IsSignedB, DL);
    if (ResA.empty() || ResB.empty())
      return {};
    ResA[0].Coefficient += ResB[0].Coefficient;
    append_range(ResA, drop_begin(ResB));
    return ResA;
  };

  // Decompose \p V used with a signed predicate.
  if (IsSigned) {
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
      if (canUseSExt(CI))
        return {{CI->getSExtValue(), nullptr}};
    }
    Value *Op0;
    Value *Op1;
    if (match(V, m_NSWAdd(m_Value(Op0), m_Value(Op1))))
      return MergeResults(Op0, Op1, IsSigned);

    return {{0, nullptr}, {1, V}};
  }

  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->uge(MaxConstraintValue))
      return {};
    return {{int64_t(CI->getZExtValue()), nullptr}};
  }

  if (auto *GEP = dyn_cast<GetElementPtrInst>(V))
    return decomposeGEP(*GEP, Preconditions, IsSigned, DL);

  Value *Op0;
  bool IsKnownPositive = false;
  if (match(V, m_ZExt(m_Value(Op0)))) {
    IsKnownPositive = true;
    V = Op0;
  }

  Value *Op1;
  ConstantInt *CI;
  if (match(V, m_NUWAdd(m_Value(Op0), m_Value(Op1)))) {
    return MergeResults(Op0, Op1, IsSigned);
  }
  if (match(V, m_NSWAdd(m_Value(Op0), m_Value(Op1)))) {
    if (!isKnownNonNegative(Op0, DL, /*Depth=*/MaxAnalysisRecursionDepth - 1))
      Preconditions.emplace_back(CmpInst::ICMP_SGE, Op0,
                                 ConstantInt::get(Op0->getType(), 0));
    if (!isKnownNonNegative(Op1, DL, /*Depth=*/MaxAnalysisRecursionDepth - 1))
      Preconditions.emplace_back(CmpInst::ICMP_SGE, Op1,
                                 ConstantInt::get(Op1->getType(), 0));

    return MergeResults(Op0, Op1, IsSigned);
  }

  if (match(V, m_Add(m_Value(Op0), m_ConstantInt(CI))) && CI->isNegative() &&
      canUseSExt(CI)) {
    Preconditions.emplace_back(
        CmpInst::ICMP_UGE, Op0,
        ConstantInt::get(Op0->getType(), CI->getSExtValue() * -1));
    return MergeResults(Op0, CI, true);
  }

  if (match(V, m_NUWShl(m_Value(Op1), m_ConstantInt(CI))) && canUseSExt(CI)) {
    int64_t Mult = int64_t(std::pow(int64_t(2), CI->getSExtValue()));
    auto Result = decompose(Op1, Preconditions, IsSigned, DL);
    for (auto &KV : Result)
      KV.Coefficient *= Mult;
    return Result;
  }

  if (match(V, m_NUWMul(m_Value(Op1), m_ConstantInt(CI))) && canUseSExt(CI) &&
      (!CI->isNegative())) {
    auto Result = decompose(Op1, Preconditions, IsSigned, DL);
    for (auto &KV : Result)
      KV.Coefficient *= CI->getSExtValue();
    return Result;
  }

  if (match(V, m_NUWSub(m_Value(Op0), m_ConstantInt(CI))) && canUseSExt(CI))
    return {{-1 * CI->getSExtValue(), nullptr}, {1, Op0}};
  if (match(V, m_NUWSub(m_Value(Op0), m_Value(Op1))))
    return {{0, nullptr}, {1, Op0}, {-1, Op1}};

  return {{0, nullptr}, {1, V, IsKnownPositive}};
}

ConstraintTy
ConstraintInfo::getConstraint(CmpInst::Predicate Pred, Value *Op0, Value *Op1,
                              SmallVectorImpl<Value *> &NewVariables) const {
  assert(NewVariables.empty() && "NewVariables must be empty when passed in");
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
  auto &Value2Index = getValue2Index(IsSigned);
  auto ADec = decompose(Op0->stripPointerCastsSameRepresentation(),
                        Preconditions, IsSigned, DL);
  auto BDec = decompose(Op1->stripPointerCastsSameRepresentation(),
                        Preconditions, IsSigned, DL);
  // Skip if decomposing either of the values failed.
  if (ADec.empty() || BDec.empty())
    return {};

  int64_t Offset1 = ADec[0].Coefficient;
  int64_t Offset2 = BDec[0].Coefficient;
  Offset1 *= -1;

  // Create iterator ranges that skip the constant-factor.
  auto VariablesA = llvm::drop_begin(ADec);
  auto VariablesB = llvm::drop_begin(BDec);

  // First try to look up \p V in Value2Index and NewVariables. Otherwise add a
  // new entry to NewVariables.
  DenseMap<Value *, unsigned> NewIndexMap;
  auto GetOrAddIndex = [&Value2Index, &NewVariables,
                        &NewIndexMap](Value *V) -> unsigned {
    auto V2I = Value2Index.find(V);
    if (V2I != Value2Index.end())
      return V2I->second;
    auto Insert =
        NewIndexMap.insert({V, Value2Index.size() + NewVariables.size() + 1});
    if (Insert.second)
      NewVariables.push_back(V);
    return Insert.first->second;
  };

  // Make sure all variables have entries in Value2Index or NewVariables.
  for (const auto &KV : concat<DecompEntry>(VariablesA, VariablesB))
    GetOrAddIndex(KV.Variable);

  // Build result constraint, by first adding all coefficients from A and then
  // subtracting all coefficients from B.
  ConstraintTy Res(
      SmallVector<int64_t, 8>(Value2Index.size() + NewVariables.size() + 1, 0),
      IsSigned);
  // Collect variables that are known to be positive in all uses in the
  // constraint.
  DenseMap<Value *, bool> KnownPositiveVariables;
  Res.IsEq = IsEq;
  auto &R = Res.Coefficients;
  for (const auto &KV : VariablesA) {
    R[GetOrAddIndex(KV.Variable)] += KV.Coefficient;
    auto I = KnownPositiveVariables.insert({KV.Variable, KV.IsKnownPositive});
    I.first->second &= KV.IsKnownPositive;
  }

  for (const auto &KV : VariablesB) {
    R[GetOrAddIndex(KV.Variable)] -= KV.Coefficient;
    auto I = KnownPositiveVariables.insert({KV.Variable, KV.IsKnownPositive});
    I.first->second &= KV.IsKnownPositive;
  }

  int64_t OffsetSum;
  if (AddOverflow(Offset1, Offset2, OffsetSum))
    return {};
  if (Pred == (IsSigned ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT))
    if (AddOverflow(OffsetSum, int64_t(-1), OffsetSum))
      return {};
  R[0] = OffsetSum;
  Res.Preconditions = std::move(Preconditions);

  // Remove any (Coefficient, Variable) entry where the Coefficient is 0 for new
  // variables.
  while (!NewVariables.empty()) {
    int64_t Last = R.back();
    if (Last != 0)
      break;
    R.pop_back();
    Value *RemovedV = NewVariables.pop_back_val();
    NewIndexMap.erase(RemovedV);
  }

  // Add extra constraints for variables that are known positive.
  for (auto &KV : KnownPositiveVariables) {
    if (!KV.second || (Value2Index.find(KV.first) == Value2Index.end() &&
                       NewIndexMap.find(KV.first) == NewIndexMap.end()))
      continue;
    SmallVector<int64_t, 8> C(Value2Index.size() + NewVariables.size() + 1, 0);
    C[GetOrAddIndex(KV.first)] = -1;
    Res.ExtraInfo.push_back(C);
  }
  return Res;
}

ConstraintTy ConstraintInfo::getConstraintForSolving(CmpInst::Predicate Pred,
                                                     Value *Op0,
                                                     Value *Op1) const {
  // If both operands are known to be non-negative, change signed predicates to
  // unsigned ones. This increases the reasoning effectiveness in combination
  // with the signed <-> unsigned transfer logic.
  if (CmpInst::isSigned(Pred) &&
      isKnownNonNegative(Op0, DL, /*Depth=*/MaxAnalysisRecursionDepth - 1) &&
      isKnownNonNegative(Op1, DL, /*Depth=*/MaxAnalysisRecursionDepth - 1))
    Pred = CmpInst::getUnsignedPredicate(Pred);

  SmallVector<Value *> NewVariables;
  ConstraintTy R = getConstraint(Pred, Op0, Op1, NewVariables);
  if (R.IsEq || !NewVariables.empty())
    return {};
  return R;
}

bool ConstraintTy::isValid(const ConstraintInfo &Info) const {
  return Coefficients.size() > 0 &&
         all_of(Preconditions, [&Info](const PreconditionTy &C) {
           return Info.doesHold(C.Pred, C.Op0, C.Op1);
         });
}

bool ConstraintInfo::doesHold(CmpInst::Predicate Pred, Value *A,
                              Value *B) const {
  auto R = getConstraintForSolving(Pred, A, B);
  return R.Preconditions.empty() && !R.empty() &&
         getCS(R.IsSigned).isConditionImplied(R.Coefficients);
}

void ConstraintInfo::transferToOtherSystem(
    CmpInst::Predicate Pred, Value *A, Value *B, unsigned NumIn,
    unsigned NumOut, SmallVectorImpl<StackEntry> &DFSInStack) {
  // Check if we can combine facts from the signed and unsigned systems to
  // derive additional facts.
  if (!A->getType()->isIntegerTy())
    return;
  // FIXME: This currently depends on the order we add facts. Ideally we
  // would first add all known facts and only then try to add additional
  // facts.
  switch (Pred) {
  default:
    break;
  case CmpInst::ICMP_ULT:
    //  If B is a signed positive constant, A >=s 0 and A <s B.
    if (doesHold(CmpInst::ICMP_SGE, B, ConstantInt::get(B->getType(), 0))) {
      addFact(CmpInst::ICMP_SGE, A, ConstantInt::get(B->getType(), 0), NumIn,
              NumOut, DFSInStack);
      addFact(CmpInst::ICMP_SLT, A, B, NumIn, NumOut, DFSInStack);
    }
    break;
  case CmpInst::ICMP_SLT:
    if (doesHold(CmpInst::ICMP_SGE, A, ConstantInt::get(B->getType(), 0)))
      addFact(CmpInst::ICMP_ULT, A, B, NumIn, NumOut, DFSInStack);
    break;
  case CmpInst::ICMP_SGT:
    if (doesHold(CmpInst::ICMP_SGE, B, ConstantInt::get(B->getType(), -1)))
      addFact(CmpInst::ICMP_UGE, A, ConstantInt::get(B->getType(), 0), NumIn,
              NumOut, DFSInStack);
    break;
  case CmpInst::ICMP_SGE:
    if (doesHold(CmpInst::ICMP_SGE, B, ConstantInt::get(B->getType(), 0))) {
      addFact(CmpInst::ICMP_UGE, A, B, NumIn, NumOut, DFSInStack);
    }
    break;
  }
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
static void dumpWithNames(const ConstraintSystem &CS,
                          DenseMap<Value *, unsigned> &Value2Index) {
  SmallVector<std::string> Names(Value2Index.size(), "");
  for (auto &KV : Value2Index) {
    Names[KV.second - 1] = std::string("%") + KV.first->getName().str();
  }
  CS.dump(Names);
}

static void dumpWithNames(ArrayRef<int64_t> C,
                          DenseMap<Value *, unsigned> &Value2Index) {
  ConstraintSystem CS;
  CS.addVariableRowFill(C);
  dumpWithNames(CS, Value2Index);
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

  Value *Cond = Br->getCondition();

  // If the condition is a chain of ORs/AND and the successor only has the
  // current block as predecessor, queue conditions for the successor.
  Value *Op0, *Op1;
  if (match(Cond, m_LogicalOr(m_Value(Op0), m_Value(Op1))) ||
      match(Cond, m_LogicalAnd(m_Value(Op0), m_Value(Op1)))) {
    bool IsOr = match(Cond, m_LogicalOr());
    bool IsAnd = match(Cond, m_LogicalAnd());
    // If there's a select that matches both AND and OR, we need to commit to
    // one of the options. Arbitrarily pick OR.
    if (IsOr && IsAnd)
      IsAnd = false;

    BasicBlock *Successor = Br->getSuccessor(IsOr ? 1 : 0);
    if (canAddSuccessor(BB, Successor)) {
      SmallVector<Value *> CondWorkList;
      SmallPtrSet<Value *, 8> SeenCond;
      auto QueueValue = [&CondWorkList, &SeenCond](Value *V) {
        if (SeenCond.insert(V).second)
          CondWorkList.push_back(V);
      };
      QueueValue(Op1);
      QueueValue(Op0);
      while (!CondWorkList.empty()) {
        Value *Cur = CondWorkList.pop_back_val();
        if (auto *Cmp = dyn_cast<ICmpInst>(Cur)) {
          WorkList.emplace_back(DT.getNode(Successor), Cmp, IsOr);
          continue;
        }
        if (IsOr && match(Cur, m_LogicalOr(m_Value(Op0), m_Value(Op1)))) {
          QueueValue(Op1);
          QueueValue(Op0);
          continue;
        }
        if (IsAnd && match(Cur, m_LogicalAnd(m_Value(Op0), m_Value(Op1)))) {
          QueueValue(Op1);
          QueueValue(Op0);
          continue;
        }
      }
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

static bool checkAndReplaceCondition(CmpInst *Cmp, ConstraintInfo &Info) {
  LLVM_DEBUG(dbgs() << "Checking " << *Cmp << "\n");
  CmpInst::Predicate Pred = Cmp->getPredicate();
  Value *A = Cmp->getOperand(0);
  Value *B = Cmp->getOperand(1);

  auto R = Info.getConstraintForSolving(Pred, A, B);
  if (R.empty() || !R.isValid(Info)){
    LLVM_DEBUG(dbgs() << "   failed to decompose condition\n");
    return false;
  }

  auto &CSToUse = Info.getCS(R.IsSigned);

  // If there was extra information collected during decomposition, apply
  // it now and remove it immediately once we are done with reasoning
  // about the constraint.
  for (auto &Row : R.ExtraInfo)
    CSToUse.addVariableRow(Row);
  auto InfoRestorer = make_scope_exit([&]() {
    for (unsigned I = 0; I < R.ExtraInfo.size(); ++I)
      CSToUse.popLastConstraint();
  });

  bool Changed = false;
  LLVMContext &Ctx = Cmp->getModule()->getContext();
  if (CSToUse.isConditionImplied(R.Coefficients)) {
    if (!DebugCounter::shouldExecute(EliminatedCounter))
      return false;

    LLVM_DEBUG({
      dbgs() << "Condition " << *Cmp << " implied by dominating constraints\n";
      dumpWithNames(CSToUse, Info.getValue2Index(R.IsSigned));
    });
    Cmp->replaceUsesWithIf(ConstantInt::getTrue(Ctx), [](Use &U) {
      // Conditions in an assume trivially simplify to true. Skip uses
      // in assume calls to not destroy the available information.
      auto *II = dyn_cast<IntrinsicInst>(U.getUser());
      return !II || II->getIntrinsicID() != Intrinsic::assume;
    });
    NumCondsRemoved++;
    Changed = true;
  }
  if (CSToUse.isConditionImplied(ConstraintSystem::negate(R.Coefficients))) {
    if (!DebugCounter::shouldExecute(EliminatedCounter))
      return false;

    LLVM_DEBUG({
      dbgs() << "Condition !" << *Cmp << " implied by dominating constraints\n";
      dumpWithNames(CSToUse, Info.getValue2Index(R.IsSigned));
    });
    Cmp->replaceAllUsesWith(ConstantInt::getFalse(Ctx));
    NumCondsRemoved++;
    Changed = true;
  }
  return Changed;
}

void ConstraintInfo::addFact(CmpInst::Predicate Pred, Value *A, Value *B,
                             unsigned NumIn, unsigned NumOut,
                             SmallVectorImpl<StackEntry> &DFSInStack) {
  // If the constraint has a pre-condition, skip the constraint if it does not
  // hold.
  SmallVector<Value *> NewVariables;
  auto R = getConstraint(Pred, A, B, NewVariables);
  if (!R.isValid(*this))
    return;

  LLVM_DEBUG(dbgs() << "Adding '" << CmpInst::getPredicateName(Pred) << " ";
             A->printAsOperand(dbgs(), false); dbgs() << ", ";
             B->printAsOperand(dbgs(), false); dbgs() << "'\n");
  bool Added = false;
  auto &CSToUse = getCS(R.IsSigned);
  if (R.Coefficients.empty())
    return;

  Added |= CSToUse.addVariableRowFill(R.Coefficients);

  // If R has been added to the system, add the new variables and queue it for
  // removal once it goes out-of-scope.
  if (Added) {
    SmallVector<Value *, 2> ValuesToRelease;
    auto &Value2Index = getValue2Index(R.IsSigned);
    for (Value *V : NewVariables) {
      Value2Index.insert({V, Value2Index.size() + 1});
      ValuesToRelease.push_back(V);
    }

    LLVM_DEBUG({
      dbgs() << "  constraint: ";
      dumpWithNames(R.Coefficients, getValue2Index(R.IsSigned));
      dbgs() << "\n";
    });

    DFSInStack.emplace_back(NumIn, NumOut, R.IsSigned, ValuesToRelease);

    if (R.IsEq) {
      // Also add the inverted constraint for equality constraints.
      for (auto &Coeff : R.Coefficients)
        Coeff *= -1;
      CSToUse.addVariableRowFill(R.Coefficients);

      DFSInStack.emplace_back(NumIn, NumOut, R.IsSigned,
                              SmallVector<Value *, 2>());
    }
  }
}

static bool replaceSubOverflowUses(IntrinsicInst *II, Value *A, Value *B,
                                   SmallVectorImpl<Instruction *> &ToRemove) {
  bool Changed = false;
  IRBuilder<> Builder(II->getParent(), II->getIterator());
  Value *Sub = nullptr;
  for (User *U : make_early_inc_range(II->users())) {
    if (match(U, m_ExtractValue<0>(m_Value()))) {
      if (!Sub)
        Sub = Builder.CreateSub(A, B);
      U->replaceAllUsesWith(Sub);
      Changed = true;
    } else if (match(U, m_ExtractValue<1>(m_Value()))) {
      U->replaceAllUsesWith(Builder.getFalse());
      Changed = true;
    } else
      continue;

    if (U->use_empty()) {
      auto *I = cast<Instruction>(U);
      ToRemove.push_back(I);
      I->setOperand(0, PoisonValue::get(II->getType()));
      Changed = true;
    }
  }

  if (II->use_empty()) {
    II->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static bool
tryToSimplifyOverflowMath(IntrinsicInst *II, ConstraintInfo &Info,
                          SmallVectorImpl<Instruction *> &ToRemove) {
  auto DoesConditionHold = [](CmpInst::Predicate Pred, Value *A, Value *B,
                              ConstraintInfo &Info) {
    auto R = Info.getConstraintForSolving(Pred, A, B);
    if (R.size() < 2 || !R.isValid(Info))
      return false;

    auto &CSToUse = Info.getCS(R.IsSigned);
    return CSToUse.isConditionImplied(R.Coefficients);
  };

  bool Changed = false;
  if (II->getIntrinsicID() == Intrinsic::ssub_with_overflow) {
    // If A s>= B && B s>= 0, ssub.with.overflow(a, b) should not overflow and
    // can be simplified to a regular sub.
    Value *A = II->getArgOperand(0);
    Value *B = II->getArgOperand(1);
    if (!DoesConditionHold(CmpInst::ICMP_SGE, A, B, Info) ||
        !DoesConditionHold(CmpInst::ICMP_SGE, B,
                           ConstantInt::get(A->getType(), 0), Info))
      return false;
    Changed = replaceSubOverflowUses(II, A, B, ToRemove);
  }
  return Changed;
}

static bool eliminateConstraints(Function &F, DominatorTree &DT) {
  bool Changed = false;
  DT.updateDFSNumbers();

  ConstraintInfo Info(F.getParent()->getDataLayout());
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
  // it holds on entry to the block. Also make sure conditions with constant
  // operands come before conditions without constant operands. This increases
  // the effectiveness of the current signed <-> unsigned fact transfer logic.
  stable_sort(
      S.WorkList, [](const ConstraintOrBlock &A, const ConstraintOrBlock &B) {
        auto HasNoConstOp = [](const ConstraintOrBlock &B) {
          return !B.IsBlock && !isa<ConstantInt>(B.Condition->getOperand(0)) &&
                 !isa<ConstantInt>(B.Condition->getOperand(1));
        };
        bool NoConstOpA = HasNoConstOp(A);
        bool NoConstOpB = HasNoConstOp(B);
        return std::tie(A.NumIn, A.IsBlock, NoConstOpA) <
               std::tie(B.NumIn, B.IsBlock, NoConstOpB);
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
      LLVM_DEBUG({
        dbgs() << "Removing ";
        dumpWithNames(Info.getCS(E.IsSigned).getLastConstraint(),
                      Info.getValue2Index(E.IsSigned));
        dbgs() << "\n";
      });

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
          Changed |= tryToSimplifyOverflowMath(II, Info, ToRemove);
          continue;
        }
        auto *Cmp = dyn_cast<ICmpInst>(&I);
        if (!Cmp)
          continue;

        Changed |= checkAndReplaceCondition(Cmp, Info);
      }
      continue;
    }

    ICmpInst::Predicate Pred;
    Value *A, *B;
    if (match(CB.Condition, m_ICmp(Pred, m_Value(A), m_Value(B)))) {
      // Use the inverse predicate if required.
      if (CB.Not)
        Pred = CmpInst::getInversePredicate(Pred);

      Info.addFact(Pred, A, B, CB.NumIn, CB.NumOut, DFSInStack);
      Info.transferToOtherSystem(Pred, A, B, CB.NumIn, CB.NumOut, DFSInStack);
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
