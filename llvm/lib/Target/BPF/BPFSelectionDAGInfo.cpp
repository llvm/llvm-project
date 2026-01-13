//===-- BPFSelectionDAGInfo.cpp - BPF SelectionDAG Info -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BPFSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "BPFSelectionDAGInfo.h"
#include "BPFTargetMachine.h"
#include "llvm/CodeGen/SelectionDAG.h"

#define GET_SDNODE_DESC
#include "BPFGenSDNodeInfo.inc"

using namespace llvm;

#define DEBUG_TYPE "bpf-selectiondag-info"

BPFSelectionDAGInfo::BPFSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(BPFGenSDNodeInfo) {}

SDValue BPFSelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  // Requires the copy size to be a constant.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (!ConstantSize)
    return SDValue();

  unsigned CopyLen = ConstantSize->getZExtValue();
  unsigned StoresNumEstimate = alignTo(CopyLen, Alignment) >> Log2(Alignment);
  // Impose the same copy length limit as MaxStoresPerMemcpy.
  if (StoresNumEstimate > getCommonMaxStoresPerMemFunc())
    return SDValue();

  return DAG.getNode(BPFISD::MEMCPY, dl, MVT::Other, Chain, Dst, Src,
                     DAG.getConstant(CopyLen, dl, MVT::i64),
                     DAG.getConstant(Alignment.value(), dl, MVT::i64));
}
