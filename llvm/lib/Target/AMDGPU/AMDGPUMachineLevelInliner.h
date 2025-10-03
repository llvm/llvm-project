//===-- AMDGPUMachineLevelInliner.h - AMDGPU Machine Level Inliner -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGPUMachineLevelInliner pass, which performs
// machine-level inlining for AMDGPU targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINELEVELINLINER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINELEVELINLINER_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

namespace llvm {

class AMDGPUMachineLevelInliner : public MachineFunctionPass {
public:
  static char ID; // Pass identification

  AMDGPUMachineLevelInliner();

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  StringRef getPassName() const override {
    return "AMDGPU Machine Level Inliner";
  }

private:
  bool shouldInlineCallsTo(const Function &Callee) {
    return Callee.getCallingConv() == CallingConv::AMDGPU_Gfx_WholeWave;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINELEVELINLINER_H
