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
// Currently it provides generic matchers for unary and binary VPInstructions,
// and specialized matchers like m_Not, m_ActiveLaneMask, m_BranchOnCond,
// m_BranchOnCount to match specific VPInstructions.
// TODO: Add missing matchers for additional opcodes and recipes as needed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORM_VECTORIZE_VPLANPATTERNMATCH_H
#define LLVM_TRANSFORM_VECTORIZE_VPLANPATTERNMATCH_H

#include "VPlan.h"

namespace llvm {
namespace VPlanPatternMatch {

template <typename Val, typename Pattern> bool match(Val *V, const Pattern &P) {
  return const_cast<Pattern &>(P).match(V);
}

template <typename Class> struct class_match {
  template <typename ITy> bool match(ITy *V) { return isa<Class>(V); }
};

/// Match an arbitrary VPValue and ignore it.
inline class_match<VPValue> m_VPValue() { return class_match<VPValue>(); }

template <typename Class> struct bind_ty {
  Class *&VR;

  bind_ty(Class *&V) : VR(V) {}

  template <typename ITy> bool match(ITy *V) {
    if (auto *CV = dyn_cast<Class>(V)) {
      VR = CV;
      return true;
    }
    return false;
  }
};

/// Match a specified integer value or vector of all elements of that
/// value.
struct specific_intval {
  APInt Val;

  specific_intval(APInt V) : Val(std::move(V)) {}

  bool match(VPValue *VPV) {
    if (!VPV->isLiveIn())
      return false;
    Value *V = VPV->getLiveInIRValue();
    const auto *CI = dyn_cast<ConstantInt>(V);
    if (!CI && V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        CI = dyn_cast_or_null<ConstantInt>(
            C->getSplatValue(/*UndefsAllowed=*/false));

    return CI && APInt::isSameValue(CI->getValue(), Val);
  }
};

inline specific_intval m_SpecificInt(uint64_t V) {
  return specific_intval(APInt(64, V));
}

/// Matching combinators
template <typename LTy, typename RTy> struct match_combine_or {
  LTy L;
  RTy R;

  match_combine_or(const LTy &Left, const RTy &Right) : L(Left), R(Right) {}

  template <typename ITy> bool match(ITy *V) {
    if (L.match(V))
      return true;
    if (R.match(V))
      return true;
    return false;
  }
};

template <typename LTy, typename RTy>
inline match_combine_or<LTy, RTy> m_CombineOr(const LTy &L, const RTy &R) {
  return match_combine_or<LTy, RTy>(L, R);
}

/// Match a VPValue, capturing it if we match.
inline bind_ty<VPValue> m_VPValue(VPValue *&V) { return V; }

namespace detail {

/// A helper to match an opcode against multiple recipe types.
template <unsigned Opcode, typename...> struct MatchRecipeAndOpcode {};

template <unsigned Opcode, typename RecipeTy>
struct MatchRecipeAndOpcode<Opcode, RecipeTy> {
  static bool match(const VPRecipeBase *R) {
    auto *DefR = dyn_cast<RecipeTy>(R);
    return DefR && DefR->getOpcode() == Opcode;
  }
};

template <unsigned Opcode, typename RecipeTy, typename... RecipeTys>
struct MatchRecipeAndOpcode<Opcode, RecipeTy, RecipeTys...> {
  static bool match(const VPRecipeBase *R) {
    return MatchRecipeAndOpcode<Opcode, RecipeTy>::match(R) ||
           MatchRecipeAndOpcode<Opcode, RecipeTys...>::match(R);
  }
};
} // namespace detail

template <typename Op0_t, unsigned Opcode, typename... RecipeTys>
struct UnaryRecipe_match {
  Op0_t Op0;

  UnaryRecipe_match(Op0_t Op0) : Op0(Op0) {}

  bool match(const VPValue *V) {
    auto *DefR = V->getDefiningRecipe();
    return DefR && match(DefR);
  }

  bool match(const VPRecipeBase *R) {
    if (!detail::MatchRecipeAndOpcode<Opcode, RecipeTys...>::match(R))
      return false;
    assert(R->getNumOperands() == 1 &&
           "recipe with matched opcode does not have 1 operands");
    return Op0.match(R->getOperand(0));
  }
};

template <typename Op0_t, unsigned Opcode>
using UnaryVPInstruction_match =
    UnaryRecipe_match<Op0_t, Opcode, VPInstruction>;

template <typename Op0_t, unsigned Opcode>
using AllUnaryRecipe_match =
    UnaryRecipe_match<Op0_t, Opcode, VPWidenRecipe, VPReplicateRecipe,
                      VPWidenCastRecipe, VPInstruction>;

template <typename Op0_t, typename Op1_t, unsigned Opcode,
          typename... RecipeTys>
struct BinaryRecipe_match {
  Op0_t Op0;
  Op1_t Op1;

  BinaryRecipe_match(Op0_t Op0, Op1_t Op1) : Op0(Op0), Op1(Op1) {}

  bool match(const VPValue *V) {
    auto *DefR = V->getDefiningRecipe();
    return DefR && match(DefR);
  }

  bool match(const VPSingleDefRecipe *R) {
    return match(static_cast<const VPRecipeBase *>(R));
  }

  bool match(const VPRecipeBase *R) {
    if (!detail::MatchRecipeAndOpcode<Opcode, RecipeTys...>::match(R))
      return false;
    assert(R->getNumOperands() == 2 &&
           "recipe with matched opcode does not have 2 operands");
    return Op0.match(R->getOperand(0)) && Op1.match(R->getOperand(1));
  }
};

template <typename Op0_t, typename Op1_t, unsigned Opcode>
using BinaryVPInstruction_match =
    BinaryRecipe_match<Op0_t, Op1_t, Opcode, VPInstruction>;

template <typename Op0_t, typename Op1_t, unsigned Opcode>
using AllBinaryRecipe_match =
    BinaryRecipe_match<Op0_t, Op1_t, Opcode, VPWidenRecipe, VPReplicateRecipe,
                       VPWidenCastRecipe, VPInstruction>;

template <unsigned Opcode, typename Op0_t>
inline UnaryVPInstruction_match<Op0_t, Opcode>
m_VPInstruction(const Op0_t &Op0) {
  return UnaryVPInstruction_match<Op0_t, Opcode>(Op0);
}

template <unsigned Opcode, typename Op0_t, typename Op1_t>
inline BinaryVPInstruction_match<Op0_t, Op1_t, Opcode>
m_VPInstruction(const Op0_t &Op0, const Op1_t &Op1) {
  return BinaryVPInstruction_match<Op0_t, Op1_t, Opcode>(Op0, Op1);
}

template <typename Op0_t>
inline UnaryVPInstruction_match<Op0_t, VPInstruction::Not>
m_Not(const Op0_t &Op0) {
  return m_VPInstruction<VPInstruction::Not>(Op0);
}

template <typename Op0_t>
inline UnaryVPInstruction_match<Op0_t, VPInstruction::BranchOnCond>
m_BranchOnCond(const Op0_t &Op0) {
  return m_VPInstruction<VPInstruction::BranchOnCond>(Op0);
}

template <typename Op0_t, typename Op1_t>
inline BinaryVPInstruction_match<Op0_t, Op1_t, VPInstruction::ActiveLaneMask>
m_ActiveLaneMask(const Op0_t &Op0, const Op1_t &Op1) {
  return m_VPInstruction<VPInstruction::ActiveLaneMask>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline BinaryVPInstruction_match<Op0_t, Op1_t, VPInstruction::BranchOnCount>
m_BranchOnCount(const Op0_t &Op0, const Op1_t &Op1) {
  return m_VPInstruction<VPInstruction::BranchOnCount>(Op0, Op1);
}

template <unsigned Opcode, typename Op0_t>
inline AllUnaryRecipe_match<Op0_t, Opcode> m_Unary(const Op0_t &Op0) {
  return AllUnaryRecipe_match<Op0_t, Opcode>(Op0);
}

template <typename Op0_t>
inline AllUnaryRecipe_match<Op0_t, Instruction::Trunc>
m_Trunc(const Op0_t &Op0) {
  return m_Unary<Instruction::Trunc, Op0_t>(Op0);
}

template <typename Op0_t>
inline AllUnaryRecipe_match<Op0_t, Instruction::ZExt> m_ZExt(const Op0_t &Op0) {
  return m_Unary<Instruction::ZExt, Op0_t>(Op0);
}

template <typename Op0_t>
inline AllUnaryRecipe_match<Op0_t, Instruction::SExt> m_SExt(const Op0_t &Op0) {
  return m_Unary<Instruction::SExt, Op0_t>(Op0);
}

template <typename Op0_t>
inline match_combine_or<AllUnaryRecipe_match<Op0_t, Instruction::ZExt>,
                        AllUnaryRecipe_match<Op0_t, Instruction::SExt>>
m_ZExtOrSExt(const Op0_t &Op0) {
  return m_CombineOr(m_ZExt(Op0), m_SExt(Op0));
}

template <unsigned Opcode, typename Op0_t, typename Op1_t>
inline AllBinaryRecipe_match<Op0_t, Op1_t, Opcode> m_Binary(const Op0_t &Op0,
                                                            const Op1_t &Op1) {
  return AllBinaryRecipe_match<Op0_t, Op1_t, Opcode>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline AllBinaryRecipe_match<Op0_t, Op1_t, Instruction::Mul>
m_Mul(const Op0_t &Op0, const Op1_t &Op1) {
  return m_Binary<Instruction::Mul, Op0_t, Op1_t>(Op0, Op1);
}

} // namespace VPlanPatternMatch
} // namespace llvm

#endif
