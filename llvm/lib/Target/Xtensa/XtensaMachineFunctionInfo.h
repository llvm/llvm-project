//==- XtensaMachineFunctionInfo.h - Xtensa machine function info --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares Xtensa-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_XTENSAMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_XTENSA_XTENSAMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class XtensaMachineFunctionInfo : public MachineFunctionInfo {
  /// FrameIndex of the spill slot for the scratch register in BranchRelaxation.
  int BranchRelaxationScratchFrameIndex = -1;
  unsigned VarArgsFirstGPR;
  int VarArgsOnStackFrameIndex;
  int VarArgsInRegsFrameIndex;

public:
  explicit XtensaMachineFunctionInfo(const Function &F,
                                     const TargetSubtargetInfo *STI)
      : VarArgsFirstGPR(0), VarArgsOnStackFrameIndex(0),
        VarArgsInRegsFrameIndex(0) {}

  int getBranchRelaxationScratchFrameIndex() const {
    return BranchRelaxationScratchFrameIndex;
  }
  void setBranchRelaxationScratchFrameIndex(int Index) {
    BranchRelaxationScratchFrameIndex = Index;
  }

  unsigned getVarArgsFirstGPR() const { return VarArgsFirstGPR; }
  void setVarArgsFirstGPR(unsigned GPR) { VarArgsFirstGPR = GPR; }

  int getVarArgsOnStackFrameIndex() const { return VarArgsOnStackFrameIndex; }
  void setVarArgsOnStackFrameIndex(int FI) { VarArgsOnStackFrameIndex = FI; }

  // Get and set the frame index of the first stack vararg.
  int getVarArgsInRegsFrameIndex() const { return VarArgsInRegsFrameIndex; }
  void setVarArgsInRegsFrameIndex(int FI) { VarArgsInRegsFrameIndex = FI; }
};

} // namespace llvm

#endif /* LLVM_LIB_TARGET_XTENSA_XTENSAMACHINEFUNCTIONINFO_H */
