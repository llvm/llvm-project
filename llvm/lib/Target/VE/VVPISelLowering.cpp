//===-- VVPISelLowering.cpp - VE DAG Lowering Implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering and legalization of vector instructions to
// VVP_*layer SDNodes.
//
//===----------------------------------------------------------------------===//

#include "VECustomDAG.h"
#include "VEISelLowering.h"

using namespace llvm;

#define DEBUG_TYPE "ve-lower"

SDValue VETargetLowering::legalizeInternalVectorOp(SDValue Op,
                                                   SelectionDAG &DAG) const {
  VECustomDAG CDAG(DAG, Op);
  // TODO: Implement odd/even splitting.
  return legalizePackedAVL(Op, CDAG);
}

SDValue VETargetLowering::legalizePackedAVL(SDValue Op,
                                            VECustomDAG &CDAG) const {
  LLVM_DEBUG(dbgs() << "::legalizePackedAVL\n";);
  // Only required for VEC and VVP ops.
  if (!isVVPOrVEC(Op->getOpcode()))
    return Op;

  // Operation already has a legal AVL.
  auto AVL = getNodeAVL(Op);
  if (isLegalAVL(AVL))
    return Op;

  // Half and round up EVL for 32bit element types.
  SDValue LegalAVL = AVL;
  if (isPackedVectorType(Op.getValueType())) {
    assert(maySafelyIgnoreMask(Op) &&
           "TODO Shift predication from EVL into Mask");

    if (auto *ConstAVL = dyn_cast<ConstantSDNode>(AVL)) {
      LegalAVL = CDAG.getConstant((ConstAVL->getZExtValue() + 1) / 2, MVT::i32);
    } else {
      auto ConstOne = CDAG.getConstant(1, MVT::i32);
      auto PlusOne = CDAG.getNode(ISD::ADD, MVT::i32, {AVL, ConstOne});
      LegalAVL = CDAG.getNode(ISD::SRL, MVT::i32, {PlusOne, ConstOne});
    }
  }

  SDValue AnnotatedLegalAVL = CDAG.annotateLegalAVL(LegalAVL);

  // Copy the operand list.
  int NumOp = Op->getNumOperands();
  auto AVLPos = getAVLPos(Op->getOpcode());
  std::vector<SDValue> FixedOperands;
  for (int i = 0; i < NumOp; ++i) {
    if (AVLPos && (i == *AVLPos)) {
      FixedOperands.push_back(AnnotatedLegalAVL);
      continue;
    }
    FixedOperands.push_back(Op->getOperand(i));
  }

  // Clone the operation with fixed operands.
  auto Flags = Op->getFlags();
  SDValue NewN =
      CDAG.getNode(Op->getOpcode(), Op->getVTList(), FixedOperands, Flags);
  return NewN;
}
