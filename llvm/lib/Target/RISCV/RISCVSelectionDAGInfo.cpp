//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVSelectionDAGInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"

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
  case RISCVISD::TUPLE_EXTRACT:
    assert(N->getNumOperands() == 2 && "Expected three operands!");
    assert(N->getOperand(1).getOpcode() == ISD::TargetConstant &&
           N->getOperand(1).getValueType() == MVT::i32 &&
           "Expected index to be an i32 target constant!");
    break;
  case RISCVISD::TUPLE_INSERT:
    assert(N->getNumOperands() == 3 && "Expected three operands!");
    assert(N->getOperand(2).getOpcode() == ISD::TargetConstant &&
           N->getOperand(2).getValueType() == MVT::i32 &&
           "Expected index to be an i32 target constant!");
    break;
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

SDValue RISCVSelectionDAGInfo::EmitTargetCodeForMemset(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo) const {
  const RISCVSubtarget &Subtarget =
      DAG.getMachineFunction().getSubtarget<RISCVSubtarget>();
  // We currently do this only for Xqcilsm
  if (!Subtarget.hasVendorXqcilsm())
    return SDValue();

  // Do this only if we know the size at compile time.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (!ConstantSize)
    return SDValue();

  uint64_t NumberOfBytesToWrite = ConstantSize->getZExtValue();

  // Do this only if it is word aligned and we write multiple of 4 bytes.
  if (!((Alignment.value() & 3) == 0 && (NumberOfBytesToWrite & 3) == 0))
    return SDValue();

  SmallVector<SDValue, 8> OutChains;
  SDValue SizeWords, OffsetSetwmi;
  SDValue SrcValueReplicated = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, Src);
  int NumberOfWords = NumberOfBytesToWrite / 4;

  // Helper for constructing the QC_SETWMI instruction
  auto getSetwmiNode = [&](SDValue SizeWords, SDValue OffsetSetwmi) -> SDValue {
    SDValue Ops[] = {Chain, SrcValueReplicated, Dst, SizeWords, OffsetSetwmi};
    return DAG.getNode(RISCVISD::QC_SETWMI, dl, MVT::Other, Ops);
  };

  bool IsZeroVal =
      isa<ConstantSDNode>(Src) && cast<ConstantSDNode>(Src)->isZero();

  // If i8 type and constant non-zero value.
  if ((Src.getValueType() == MVT::i8) && !IsZeroVal)
    // Replicate byte to word by multiplication with 0x01010101.
    SrcValueReplicated = DAG.getNode(ISD::MUL, dl, MVT::i32, SrcValueReplicated,
                                     DAG.getConstant(16843009, dl, MVT::i32));

  // We limit a QC_SETWMI to 16 words or less to improve interruptibility.
  // So for 1-16 words we use a single QC_SETWMI:
  //
  // QC_SETWMI reg1, N, 0(reg2)
  //
  // For 17-32 words we use two QC_SETWMI's with the first as 16 words and the
  // second for the remainder:
  //
  // QC_SETWMI reg1, 16, 0(reg2)
  // QC_SETWMI reg1, N, 64(reg2)
  //
  // For 33-48 words, we would like to use (16, 16, n), but that means the last
  // QC_SETWMI needs an offset of 128 which the instruction doesnt support.
  // So in this case we use a length of 15 for the second instruction and we do
  // the rest with the third instruction.
  // This means the maximum inlined number of words is 47 (for now):
  //
  // QC_SETWMI R2, R0, 16, 0
  // QC_SETWMI R2, R0, 15, 64
  // QC_SETWMI R2, R0, N, 124
  //
  // For 48 words or more, call the target independent memset
  if (NumberOfWords <= 16) {
    // 1 - 16 words
    SizeWords = DAG.getTargetConstant(NumberOfWords, dl, MVT::i32);
    SDValue OffsetSetwmi = DAG.getTargetConstant(0, dl, MVT::i32);
    return getSetwmiNode(SizeWords, OffsetSetwmi);
  } else if (NumberOfWords <= 47) {
    if (NumberOfWords <= 32) {
      // 17 - 32 words
      SizeWords = DAG.getTargetConstant(NumberOfWords - 16, dl, MVT::i32);
      OffsetSetwmi = DAG.getTargetConstant(64, dl, MVT::i32);
      OutChains.push_back(getSetwmiNode(SizeWords, OffsetSetwmi));

      SizeWords = DAG.getTargetConstant(16, dl, MVT::i32);
      OffsetSetwmi = DAG.getTargetConstant(0, dl, MVT::i32);
      OutChains.push_back(getSetwmiNode(SizeWords, OffsetSetwmi));
    } else {
      // 33 - 47 words
      SizeWords = DAG.getTargetConstant(NumberOfWords - 31, dl, MVT::i32);
      OffsetSetwmi = DAG.getTargetConstant(124, dl, MVT::i32);
      OutChains.push_back(getSetwmiNode(SizeWords, OffsetSetwmi));

      SizeWords = DAG.getTargetConstant(15, dl, MVT::i32);
      OffsetSetwmi = DAG.getTargetConstant(64, dl, MVT::i32);
      OutChains.push_back(getSetwmiNode(SizeWords, OffsetSetwmi));

      SizeWords = DAG.getTargetConstant(16, dl, MVT::i32);
      OffsetSetwmi = DAG.getTargetConstant(0, dl, MVT::i32);
      OutChains.push_back(getSetwmiNode(SizeWords, OffsetSetwmi));
    }
    return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains);
  }

  // >= 48 words. Call target independent memset.
  return SDValue();
}
