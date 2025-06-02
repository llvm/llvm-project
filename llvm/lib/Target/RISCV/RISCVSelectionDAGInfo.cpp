//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "RISCVGenSDNodeInfo.inc"

using namespace llvm;

RISCVSelectionDAGInfo::RISCVSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(RISCVGenSDNodeInfo) {}

RISCVSelectionDAGInfo::~RISCVSelectionDAGInfo() = default;

void RISCVSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                             const SDNode *N) const {
#ifndef NDEBUG
  switch (N->getOpcode()) {
  default:
    return SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
  case RISCVISD::VQDOT_VL:
  case RISCVISD::VQDOTU_VL:
  case RISCVISD::VQDOTSU_VL: {
    assert(N->getNumValues() == 1 && "Expected one result!");
    assert(N->getNumOperands() == 5 && "Expected five operands!");
    EVT VT = N->getValueType(0);
    assert(VT.isScalableVector() && VT.getVectorElementType() == MVT::i32 &&
           "Expected result to be an i32 scalable vector");
    assert(N->getOperand(0).getValueType() == VT &&
           N->getOperand(1).getValueType() == VT &&
           N->getOperand(2).getValueType() == VT &&
           "Expected result and first 3 operands to have the same type!");
    EVT MaskVT = N->getOperand(3).getValueType();
    assert(MaskVT.isScalableVector() &&
           MaskVT.getVectorElementType() == MVT::i1 &&
           MaskVT.getVectorElementCount() == VT.getVectorElementCount() &&
           "Expected mask VT to be an i1 scalable vector with same number of "
           "elements as the result");
    assert((N->getOperand(4).getValueType() == MVT::i32 ||
            N->getOperand(4).getValueType() == MVT::i64) &&
           "Expect VL operand to be i32 or i64");
    break;
  }
  }
#endif
}
