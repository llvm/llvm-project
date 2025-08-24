//===- VPlanPatternMatch.h - Match on VPValues and recipes ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matches on the VPlan values and recipes, based on
// LLVM's IR pattern matchers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORM_VECTORIZE_VPLANPATTERNMATCH_H
#define LLVM_TRANSFORM_VECTORIZE_VPLANPATTERNMATCH_H

#include "VPlan.h"

namespace llvm {
namespace VPlanPatternMatch {

template <typename Val, typename Pattern> bool match(Val *V, const Pattern &P) {
  return P.match(V);
}

template <typename Pattern> bool match(VPUser *U, const Pattern &P) {
  auto *R = dyn_cast<VPRecipeBase>(U);
  return R && match(R, P);
}

template <typename Class> struct class_match {
  template <typename ITy> bool match(ITy *V) const { return isa<Class>(V); }
};

/// Match an arbitrary VPValue and ignore it.
inline class_match<VPValue> m_VPValue() { return class_match<VPValue>(); }

template <typename Class> struct bind_ty {
  Class *&VR;

  bind_ty(Class *&V) : VR(V) {}

  template <typename ITy> bool match(ITy *V) const {
    if (auto *CV = dyn_cast<Class>(V)) {
      VR = CV;
      return true;
    }
    return false;
  }
};

/// Match a specified VPValue.
struct specificval_ty {
  const VPValue *Val;

  specificval_ty(const VPValue *V) : Val(V) {}

  bool match(VPValue *VPV) const { return VPV == Val; }
};

inline specificval_ty m_Specific(const VPValue *VPV) { return VPV; }

/// Stores a reference to the VPValue *, not the VPValue * itself,
/// thus can be used in commutative matchers.
struct deferredval_ty {
  VPValue *const &Val;

  deferredval_ty(VPValue *const &V) : Val(V) {}

  bool match(VPValue *const V) const { return V == Val; }
};

/// Like m_Specific(), but works if the specific value to match is determined
/// as part of the same match() expression. For example:
/// m_Mul(m_VPValue(X), m_Specific(X)) is incorrect, because m_Specific() will
/// bind X before the pattern match starts.
/// m_Mul(m_VPValue(X), m_Deferred(X)) is correct, and will check against
/// whichever value m_VPValue(X) populated.
inline deferredval_ty m_Deferred(VPValue *const &V) { return V; }

/// Match an integer constant or vector of constants if Pred::isValue returns
/// true for the APInt. \p BitWidth optionally specifies the bitwidth the
/// matched constant must have. If it is 0, the matched constant can have any
/// bitwidth.
template <typename Pred, unsigned BitWidth = 0> struct int_pred_ty {
  Pred P;

  int_pred_ty(Pred P) : P(std::move(P)) {}
  int_pred_ty() : P() {}

  bool match(VPValue *VPV) const {
    if (!VPV->isLiveIn())
      return false;
    Value *V = VPV->getLiveInIRValue();
    if (!V)
      return false;
    const auto *CI = dyn_cast<ConstantInt>(V);
    if (!CI && V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        CI = dyn_cast_or_null<ConstantInt>(
            C->getSplatValue(/*AllowPoison=*/false));
    if (!CI)
      return false;

    if (BitWidth != 0 && CI->getBitWidth() != BitWidth)
      return false;
    return P.isValue(CI->getValue());
  }
};

/// Match a specified integer value or vector of all elements of that
/// value. \p BitWidth optionally specifies the bitwidth the matched constant
/// must have. If it is 0, the matched constant can have any bitwidth.
struct is_specific_int {
  APInt Val;

  is_specific_int(APInt Val) : Val(std::move(Val)) {}

  bool isValue(const APInt &C) const { return APInt::isSameValue(Val, C); }
};

template <unsigned Bitwidth = 0>
using specific_intval = int_pred_ty<is_specific_int, Bitwidth>;

inline specific_intval<0> m_SpecificInt(uint64_t V) {
  return specific_intval<0>(is_specific_int(APInt(64, V)));
}

inline specific_intval<1> m_False() {
  return specific_intval<1>(is_specific_int(APInt(64, 0)));
}

inline specific_intval<1> m_True() {
  return specific_intval<1>(is_specific_int(APInt(64, 1)));
}

struct is_all_ones {
  bool isValue(const APInt &C) const { return C.isAllOnes(); }
};

/// Match an integer or vector with all bits set.
/// For vectors, this includes constants with undefined elements.
inline int_pred_ty<is_all_ones> m_AllOnes() {
  return int_pred_ty<is_all_ones>();
}

/// Matching combinators
template <typename LTy, typename RTy> struct match_combine_or {
  LTy L;
  RTy R;

  match_combine_or(const LTy &Left, const RTy &Right) : L(Left), R(Right) {}

  template <typename ITy> bool match(ITy *V) const {
    return L.match(V) || R.match(V);
  }
};

template <typename LTy, typename RTy> struct match_combine_and {
  LTy L;
  RTy R;

  match_combine_and(const LTy &Left, const RTy &Right) : L(Left), R(Right) {}

  template <typename ITy> bool match(ITy *V) const {
    return L.match(V) && R.match(V);
  }
};

/// Combine two pattern matchers matching L || R
template <typename LTy, typename RTy>
inline match_combine_or<LTy, RTy> m_CombineOr(const LTy &L, const RTy &R) {
  return match_combine_or<LTy, RTy>(L, R);
}

/// Combine two pattern matchers matching L && R
template <typename LTy, typename RTy>
inline match_combine_and<LTy, RTy> m_CombineAnd(const LTy &L, const RTy &R) {
  return match_combine_and<LTy, RTy>(L, R);
}

/// Match a VPValue, capturing it if we match.
inline bind_ty<VPValue> m_VPValue(VPValue *&V) { return V; }

/// Match a VPInstruction, capturing if we match.
inline bind_ty<VPInstruction> m_VPInstruction(VPInstruction *&V) { return V; }

template <typename Ops_t, unsigned Opcode, bool Commutative,
          typename... RecipeTys>
struct Recipe_match {
  Ops_t Ops;

  template <typename... OpTy> Recipe_match(OpTy... Ops) : Ops(Ops...) {
    static_assert(std::tuple_size<Ops_t>::value == sizeof...(Ops) &&
                  "number of operands in constructor doesn't match Ops_t");
    static_assert((!Commutative || std::tuple_size<Ops_t>::value == 2) &&
                  "only binary ops can be commutative");
  }

  bool match(const VPValue *V) const {
    auto *DefR = V->getDefiningRecipe();
    return DefR && match(DefR);
  }

  bool match(const VPSingleDefRecipe *R) const {
    return match(static_cast<const VPRecipeBase *>(R));
  }

  bool match(const VPRecipeBase *R) const {
    if (std::tuple_size<Ops_t>::value == 0) {
      assert(Opcode == VPInstruction::BuildVector &&
             "can only match BuildVector with empty ops");
      auto *VPI = dyn_cast<VPInstruction>(R);
      return VPI && VPI->getOpcode() == VPInstruction::BuildVector;
    }

    if ((!matchRecipeAndOpcode<RecipeTys>(R) && ...))
      return false;

    assert(R->getNumOperands() == std::tuple_size<Ops_t>::value &&
           "recipe with matched opcode does not have the expected number of "
           "operands");

    auto IdxSeq = std::make_index_sequence<std::tuple_size<Ops_t>::value>();
    if (all_of_tuple_elements(IdxSeq, [R](auto Op, unsigned Idx) {
          return Op.match(R->getOperand(Idx));
        }))
      return true;

    return Commutative &&
           all_of_tuple_elements(IdxSeq, [R](auto Op, unsigned Idx) {
             return Op.match(R->getOperand(R->getNumOperands() - Idx - 1));
           });
  }

private:
  template <typename RecipeTy>
  static bool matchRecipeAndOpcode(const VPRecipeBase *R) {
    auto *DefR = dyn_cast<RecipeTy>(R);
    // Check for recipes that do not have opcodes.
    if constexpr (std::is_same<RecipeTy, VPScalarIVStepsRecipe>::value ||
                  std::is_same<RecipeTy, VPCanonicalIVPHIRecipe>::value ||
                  std::is_same<RecipeTy, VPDerivedIVRecipe>::value ||
                  std::is_same<RecipeTy, VPWidenGEPRecipe>::value)
      return DefR;
    else
      return DefR && DefR->getOpcode() == Opcode;
  }

  /// Helper to check if predicate \p P holds on all tuple elements in Ops using
  /// the provided index sequence.
  template <typename Fn, std::size_t... Is>
  bool all_of_tuple_elements(std::index_sequence<Is...>, Fn P) const {
    return (P(std::get<Is>(Ops), Is) && ...);
  }
};

template <unsigned Opcode, typename... OpTys>
using AllRecipe_match =
    Recipe_match<std::tuple<OpTys...>, Opcode, /*Commutative*/ false,
                 VPWidenRecipe, VPReplicateRecipe, VPWidenCastRecipe,
                 VPInstruction, VPWidenSelectRecipe>;

template <unsigned Opcode, typename... OpTys>
using AllRecipe_commutative_match =
    Recipe_match<std::tuple<OpTys...>, Opcode, /*Commutative*/ true,
                 VPWidenRecipe, VPReplicateRecipe, VPInstruction>;

template <unsigned Opcode, typename... OpTys>
using VPInstruction_match = Recipe_match<std::tuple<OpTys...>, Opcode,
                                         /*Commutative*/ false, VPInstruction>;

template <unsigned Opcode, typename... OpTys>
inline VPInstruction_match<Opcode, OpTys...>
m_VPInstruction(const OpTys &...Ops) {
  return VPInstruction_match<Opcode, OpTys...>(Ops...);
}

/// BuildVector is matches only its opcode, w/o matching its operands as the
/// number of operands is not fixed.
inline VPInstruction_match<VPInstruction::BuildVector> m_BuildVector() {
  return m_VPInstruction<VPInstruction::BuildVector>();
}

template <typename Op0_t>
inline VPInstruction_match<Instruction::Freeze, Op0_t>
m_Freeze(const Op0_t &Op0) {
  return m_VPInstruction<Instruction::Freeze>(Op0);
}

template <typename Op0_t>
inline VPInstruction_match<VPInstruction::BranchOnCond, Op0_t>
m_BranchOnCond(const Op0_t &Op0) {
  return m_VPInstruction<VPInstruction::BranchOnCond>(Op0);
}

template <typename Op0_t>
inline VPInstruction_match<VPInstruction::Broadcast, Op0_t>
m_Broadcast(const Op0_t &Op0) {
  return m_VPInstruction<VPInstruction::Broadcast>(Op0);
}

template <typename Op0_t>
inline VPInstruction_match<VPInstruction::ExtractLastElement, Op0_t>
m_ExtractLastElement(const Op0_t &Op0) {
  return m_VPInstruction<VPInstruction::ExtractLastElement>(Op0);
}
template <typename Op0_t, typename Op1_t>
inline VPInstruction_match<VPInstruction::ActiveLaneMask, Op0_t, Op1_t>
m_ActiveLaneMask(const Op0_t &Op0, const Op1_t &Op1) {
  return m_VPInstruction<VPInstruction::ActiveLaneMask>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline VPInstruction_match<VPInstruction::BranchOnCount, Op0_t, Op1_t>
m_BranchOnCount(const Op0_t &Op0, const Op1_t &Op1) {
  return m_VPInstruction<VPInstruction::BranchOnCount>(Op0, Op1);
}

template <unsigned Opcode, typename Op0_t>
inline AllRecipe_match<Opcode, Op0_t> m_Unary(const Op0_t &Op0) {
  return AllRecipe_match<Opcode, Op0_t>(Op0);
}

template <typename Op0_t>
inline AllRecipe_match<Instruction::Trunc, Op0_t> m_Trunc(const Op0_t &Op0) {
  return m_Unary<Instruction::Trunc, Op0_t>(Op0);
}

template <typename Op0_t>
inline AllRecipe_match<Instruction::ZExt, Op0_t> m_ZExt(const Op0_t &Op0) {
  return m_Unary<Instruction::ZExt, Op0_t>(Op0);
}

template <typename Op0_t>
inline AllRecipe_match<Instruction::SExt, Op0_t> m_SExt(const Op0_t &Op0) {
  return m_Unary<Instruction::SExt, Op0_t>(Op0);
}

template <typename Op0_t>
inline match_combine_or<AllRecipe_match<Instruction::ZExt, Op0_t>,
                        AllRecipe_match<Instruction::SExt, Op0_t>>
m_ZExtOrSExt(const Op0_t &Op0) {
  return m_CombineOr(m_ZExt(Op0), m_SExt(Op0));
}

template <unsigned Opcode, typename Op0_t, typename Op1_t>
inline AllRecipe_match<Opcode, Op0_t, Op1_t> m_Binary(const Op0_t &Op0,
                                                      const Op1_t &Op1) {
  return AllRecipe_match<Opcode, Op0_t, Op1_t>(Op0, Op1);
}

template <unsigned Opcode, typename Op0_t, typename Op1_t>
inline AllRecipe_commutative_match<Opcode, Op0_t, Op1_t>
m_c_Binary(const Op0_t &Op0, const Op1_t &Op1) {
  return AllRecipe_commutative_match<Opcode, Op0_t, Op1_t>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline AllRecipe_commutative_match<Instruction::Add, Op0_t, Op1_t>
m_c_Add(const Op0_t &Op0, const Op1_t &Op1) {
  return m_c_Binary<Instruction::Add, Op0_t, Op1_t>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline AllRecipe_match<Instruction::Sub, Op0_t, Op1_t> m_Sub(const Op0_t &Op0,
                                                             const Op1_t &Op1) {
  return m_Binary<Instruction::Sub, Op0_t, Op1_t>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline AllRecipe_match<Instruction::Mul, Op0_t, Op1_t> m_Mul(const Op0_t &Op0,
                                                             const Op1_t &Op1) {
  return m_Binary<Instruction::Mul, Op0_t, Op1_t>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline AllRecipe_commutative_match<Instruction::Mul, Op0_t, Op1_t>
m_c_Mul(const Op0_t &Op0, const Op1_t &Op1) {
  return m_c_Binary<Instruction::Mul, Op0_t, Op1_t>(Op0, Op1);
}

/// Match a binary OR operation. Note that while conceptually the operands can
/// be matched commutatively, \p Commutative defaults to false in line with the
/// IR-based pattern matching infrastructure. Use m_c_BinaryOr for a commutative
/// version of the matcher.
template <typename Op0_t, typename Op1_t>
inline AllRecipe_match<Instruction::Or, Op0_t, Op1_t>
m_BinaryOr(const Op0_t &Op0, const Op1_t &Op1) {
  return m_Binary<Instruction::Or, Op0_t, Op1_t>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline AllRecipe_commutative_match<Instruction::Or, Op0_t, Op1_t>
m_c_BinaryOr(const Op0_t &Op0, const Op1_t &Op1) {
  return m_c_Binary<Instruction::Or, Op0_t, Op1_t>(Op0, Op1);
}

/// ICmp_match is a variant of BinaryRecipe_match that also binds the comparison
/// predicate.
template <typename Op0_t, typename Op1_t> struct ICmp_match {
  CmpPredicate *Predicate = nullptr;
  Op0_t Op0;
  Op1_t Op1;

  ICmp_match(CmpPredicate &Pred, const Op0_t &Op0, const Op1_t &Op1)
      : Predicate(&Pred), Op0(Op0), Op1(Op1) {}
  ICmp_match(const Op0_t &Op0, const Op1_t &Op1) : Op0(Op0), Op1(Op1) {}

  bool match(const VPValue *V) const {
    auto *DefR = V->getDefiningRecipe();
    return DefR && match(DefR);
  }

  bool match(const VPRecipeBase *V) const {
    if (m_Binary<Instruction::ICmp>(Op0, Op1).match(V)) {
      if (Predicate)
        *Predicate = cast<VPRecipeWithIRFlags>(V)->getPredicate();
      return true;
    }
    return false;
  }
};

/// SpecificICmp_match is a variant of ICmp_match that matches the comparison
/// predicate, instead of binding it.
template <typename Op0_t, typename Op1_t> struct SpecificICmp_match {
  const CmpPredicate Predicate;
  Op0_t Op0;
  Op1_t Op1;

  SpecificICmp_match(CmpPredicate Pred, const Op0_t &LHS, const Op1_t &RHS)
      : Predicate(Pred), Op0(LHS), Op1(RHS) {}

  bool match(const VPValue *V) const {
    CmpPredicate CurrentPred;
    return ICmp_match<Op0_t, Op1_t>(CurrentPred, Op0, Op1).match(V) &&
           CmpPredicate::getMatching(CurrentPred, Predicate);
  }
};

template <typename Op0_t, typename Op1_t>
inline ICmp_match<Op0_t, Op1_t> m_ICmp(const Op0_t &Op0, const Op1_t &Op1) {
  return ICmp_match<Op0_t, Op1_t>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline ICmp_match<Op0_t, Op1_t> m_ICmp(CmpPredicate &Pred, const Op0_t &Op0,
                                       const Op1_t &Op1) {
  return ICmp_match<Op0_t, Op1_t>(Pred, Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline SpecificICmp_match<Op0_t, Op1_t>
m_SpecificICmp(CmpPredicate MatchPred, const Op0_t &Op0, const Op1_t &Op1) {
  return SpecificICmp_match<Op0_t, Op1_t>(MatchPred, Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
using GEPLikeRecipe_match =
    Recipe_match<std::tuple<Op0_t, Op1_t>, Instruction::GetElementPtr,
                 /*Commutative*/ false, VPWidenRecipe, VPReplicateRecipe,
                 VPWidenGEPRecipe, VPInstruction>;

template <typename Op0_t, typename Op1_t>
inline GEPLikeRecipe_match<Op0_t, Op1_t> m_GetElementPtr(const Op0_t &Op0,
                                                         const Op1_t &Op1) {
  return GEPLikeRecipe_match<Op0_t, Op1_t>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t, typename Op2_t>
inline AllRecipe_match<Instruction::Select, Op0_t, Op1_t, Op2_t>
m_Select(const Op0_t &Op0, const Op1_t &Op1, const Op2_t &Op2) {
  return AllRecipe_match<Instruction::Select, Op0_t, Op1_t, Op2_t>(
      {Op0, Op1, Op2});
}

template <typename Op0_t>
inline match_combine_or<VPInstruction_match<VPInstruction::Not, Op0_t>,
                        AllRecipe_commutative_match<
                            Instruction::Xor, int_pred_ty<is_all_ones>, Op0_t>>
m_Not(const Op0_t &Op0) {
  return m_CombineOr(m_VPInstruction<VPInstruction::Not>(Op0),
                     m_c_Binary<Instruction::Xor>(m_AllOnes(), Op0));
}

template <typename Op0_t, typename Op1_t>
inline match_combine_or<
    VPInstruction_match<VPInstruction::LogicalAnd, Op0_t, Op1_t>,
    AllRecipe_match<Instruction::Select, Op0_t, Op1_t, specific_intval<1>>>
m_LogicalAnd(const Op0_t &Op0, const Op1_t &Op1) {
  return m_CombineOr(
      m_VPInstruction<VPInstruction::LogicalAnd, Op0_t, Op1_t>(Op0, Op1),
      m_Select(Op0, Op1, m_False()));
}

template <typename Op0_t, typename Op1_t>
inline AllRecipe_match<Instruction::Select, Op0_t, specific_intval<1>, Op1_t>
m_LogicalOr(const Op0_t &Op0, const Op1_t &Op1) {
  return m_Select(Op0, m_True(), Op1);
}

template <typename Op0_t, typename Op1_t, typename Op2_t>
using VPScalarIVSteps_match = Recipe_match<std::tuple<Op0_t, Op1_t, Op2_t>, 0,
                                           false, VPScalarIVStepsRecipe>;

template <typename Op0_t, typename Op1_t, typename Op2_t>
inline VPScalarIVSteps_match<Op0_t, Op1_t, Op2_t>
m_ScalarIVSteps(const Op0_t &Op0, const Op1_t &Op1, const Op2_t &Op2) {
  return VPScalarIVSteps_match<Op0_t, Op1_t, Op2_t>({Op0, Op1, Op2});
}

template <typename Op0_t, typename Op1_t, typename Op2_t>
using VPDerivedIV_match =
    Recipe_match<std::tuple<Op0_t, Op1_t, Op2_t>, 0, false, VPDerivedIVRecipe>;

template <typename Op0_t, typename Op1_t, typename Op2_t>
inline VPDerivedIV_match<Op0_t, Op1_t, Op2_t>
m_DerivedIV(const Op0_t &Op0, const Op1_t &Op1, const Op2_t &Op2) {
  return VPDerivedIV_match<Op0_t, Op1_t, Op2_t>({Op0, Op1, Op2});
}

/// Match a call argument at a given argument index.
template <typename Opnd_t> struct Argument_match {
  /// Call argument index to match.
  unsigned OpI;
  Opnd_t Val;

  Argument_match(unsigned OpIdx, const Opnd_t &V) : OpI(OpIdx), Val(V) {}

  template <typename OpTy> bool match(OpTy *V) const {
    if (const auto *R = dyn_cast<VPWidenIntrinsicRecipe>(V))
      return Val.match(R->getOperand(OpI));
    if (const auto *R = dyn_cast<VPWidenCallRecipe>(V))
      return Val.match(R->getOperand(OpI));
    if (const auto *R = dyn_cast<VPReplicateRecipe>(V))
      if (isa<CallInst>(R->getUnderlyingInstr()))
        return Val.match(R->getOperand(OpI + 1));
    return false;
  }
};

/// Match a call argument.
template <unsigned OpI, typename Opnd_t>
inline Argument_match<Opnd_t> m_Argument(const Opnd_t &Op) {
  return Argument_match<Opnd_t>(OpI, Op);
}

/// Intrinsic matchers.
struct IntrinsicID_match {
  unsigned ID;

  IntrinsicID_match(Intrinsic::ID IntrID) : ID(IntrID) {}

  template <typename OpTy> bool match(OpTy *V) const {
    if (const auto *R = dyn_cast<VPWidenIntrinsicRecipe>(V))
      return R->getVectorIntrinsicID() == ID;
    if (const auto *R = dyn_cast<VPWidenCallRecipe>(V))
      return R->getCalledScalarFunction()->getIntrinsicID() == ID;
    if (const auto *R = dyn_cast<VPReplicateRecipe>(V))
      if (const auto *CI = dyn_cast<CallInst>(R->getUnderlyingInstr()))
        if (const auto *F = CI->getCalledFunction())
          return F->getIntrinsicID() == ID;
    return false;
  }
};

/// Intrinsic matches are combinations of ID matchers, and argument
/// matchers. Higher arity matcher are defined recursively in terms of and-ing
/// them with lower arity matchers. Here's some convenient typedefs for up to
/// several arguments, and more can be added as needed
template <typename T0 = void, typename T1 = void, typename T2 = void,
          typename T3 = void>
struct m_Intrinsic_Ty;
template <typename T0> struct m_Intrinsic_Ty<T0> {
  using Ty = match_combine_and<IntrinsicID_match, Argument_match<T0>>;
};
template <typename T0, typename T1> struct m_Intrinsic_Ty<T0, T1> {
  using Ty =
      match_combine_and<typename m_Intrinsic_Ty<T0>::Ty, Argument_match<T1>>;
};
template <typename T0, typename T1, typename T2>
struct m_Intrinsic_Ty<T0, T1, T2> {
  using Ty = match_combine_and<typename m_Intrinsic_Ty<T0, T1>::Ty,
                               Argument_match<T2>>;
};
template <typename T0, typename T1, typename T2, typename T3>
struct m_Intrinsic_Ty {
  using Ty = match_combine_and<typename m_Intrinsic_Ty<T0, T1, T2>::Ty,
                               Argument_match<T3>>;
};

/// Match intrinsic calls like this:
/// m_Intrinsic<Intrinsic::fabs>(m_VPValue(X), ...)
template <Intrinsic::ID IntrID> inline IntrinsicID_match m_Intrinsic() {
  return IntrinsicID_match(IntrID);
}

template <Intrinsic::ID IntrID, typename T0>
inline typename m_Intrinsic_Ty<T0>::Ty m_Intrinsic(const T0 &Op0) {
  return m_CombineAnd(m_Intrinsic<IntrID>(), m_Argument<0>(Op0));
}

template <Intrinsic::ID IntrID, typename T0, typename T1>
inline typename m_Intrinsic_Ty<T0, T1>::Ty m_Intrinsic(const T0 &Op0,
                                                       const T1 &Op1) {
  return m_CombineAnd(m_Intrinsic<IntrID>(Op0), m_Argument<1>(Op1));
}

template <Intrinsic::ID IntrID, typename T0, typename T1, typename T2>
inline typename m_Intrinsic_Ty<T0, T1, T2>::Ty
m_Intrinsic(const T0 &Op0, const T1 &Op1, const T2 &Op2) {
  return m_CombineAnd(m_Intrinsic<IntrID>(Op0, Op1), m_Argument<2>(Op2));
}

template <Intrinsic::ID IntrID, typename T0, typename T1, typename T2,
          typename T3>
inline typename m_Intrinsic_Ty<T0, T1, T2, T3>::Ty
m_Intrinsic(const T0 &Op0, const T1 &Op1, const T2 &Op2, const T3 &Op3) {
  return m_CombineAnd(m_Intrinsic<IntrID>(Op0, Op1, Op2), m_Argument<3>(Op3));
}

} // namespace VPlanPatternMatch
} // namespace llvm

#endif
