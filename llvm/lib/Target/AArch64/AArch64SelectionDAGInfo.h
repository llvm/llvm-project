//===-- AArch64SelectionDAGInfo.h - AArch64 SelectionDAG Info ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AArch64 subclass for SelectionDAGTargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64SELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64SELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

namespace llvm {

class AArch64SelectionDAGInfo : public SelectionDAGTargetInfo {
public:
  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, unsigned Align, bool isVolatile,
                                  MachinePointerInfo DstPtrInfo) const override;
  SDValue EmitTargetCodeForSetTag(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Op1, SDValue Op2,
                                  MachinePointerInfo DstPtrInfo,
                                  bool ZeroData) const override;
  bool generateFMAsInMachineCombiner(CodeGenOpt::Level OptLevel) const override;
};
}

#endif
