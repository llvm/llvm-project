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

/// Match a VPValue, capturing it if we match.
inline bind_ty<VPValue> m_VPValue(VPValue *&V) { return V; }

template <typename Op0_t, unsigned Opcode> struct UnaryVPInstruction_match {
  Op0_t Op0;

  UnaryVPInstruction_match(Op0_t Op0) : Op0(Op0) {}

  bool match(const VPValue *V) {
    auto *DefR = V->getDefiningRecipe();
    return DefR && match(DefR);
  }

  bool match(const VPRecipeBase *R) {
    auto *DefR = dyn_cast<VPInstruction>(R);
    if (!DefR || DefR->getOpcode() != Opcode)
      return false;
    assert(DefR->getNumOperands() == 1 &&
           "recipe with matched opcode does not have 1 operands");
    return Op0.match(DefR->getOperand(0));
  }
};

template <typename Op0_t, typename Op1_t, unsigned Opcode>
struct BinaryVPInstruction_match {
  Op0_t Op0;
  Op1_t Op1;

  BinaryVPInstruction_match(Op0_t Op0, Op1_t Op1) : Op0(Op0), Op1(Op1) {}

  bool match(const VPValue *V) {
    auto *DefR = V->getDefiningRecipe();
    return DefR && match(DefR);
  }

  bool match(const VPRecipeBase *R) {
    auto *DefR = dyn_cast<VPInstruction>(R);
    if (!DefR || DefR->getOpcode() != Opcode)
      return false;
    assert(DefR->getNumOperands() == 2 &&
           "recipe with matched opcode does not have 2 operands");
    return Op0.match(DefR->getOperand(0)) && Op1.match(DefR->getOperand(1));
  }
};

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
} // namespace VPlanPatternMatch
} // namespace llvm

#endif
