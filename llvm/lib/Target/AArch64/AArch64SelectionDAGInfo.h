//===-- AArch64SelectionDAGInfo.h - AArch64 SelectionDAG Info ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AArch64 subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64SELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64SELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class AArch64SelectionDAGInfo : public TargetSelectionDAGInfo {
public:

  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, SDLoc dl, SDValue Chain,
                                  SDValue Dst, SDValue Src, SDValue Size,
                                  unsigned Align, bool isVolatile,
                                  MachinePointerInfo DstPtrInfo) const override;
};
}

#endif
