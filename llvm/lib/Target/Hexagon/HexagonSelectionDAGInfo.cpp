//===-- HexagonSelectionDAGInfo.cpp - Hexagon SelectionDAG Info -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HexagonSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "hexagon-selectiondag-info"
#include "HexagonTargetMachine.h"
using namespace llvm;

bool llvm::flag_aligned_memcpy;

HexagonSelectionDAGInfo::HexagonSelectionDAGInfo(const HexagonTargetMachine
                                                 &TM)
  : TargetSelectionDAGInfo(TM) {
}

HexagonSelectionDAGInfo::~HexagonSelectionDAGInfo() {
}

SDValue
HexagonSelectionDAGInfo::
EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl, SDValue Chain,
                        SDValue Dst, SDValue Src, SDValue Size, unsigned Align,
                        bool isVolatile, bool AlwaysInline,
                        MachinePointerInfo DstPtrInfo,
                        MachinePointerInfo SrcPtrInfo) const {
  flag_aligned_memcpy = false;
  if ((Align & 0x3) == 0) {
    ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
    if (ConstantSize) {
      uint64_t SizeVal = ConstantSize->getZExtValue();
      if ((SizeVal > 32) && ((SizeVal % 8) == 0))
        flag_aligned_memcpy = true;
    }
  }

  return SDValue();
}
