//=- MachineRegisterClassInfo.h - Machine Register Class Info -----*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis calculates register class info via RegisterClassInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEREGISTERCLASSINFO_H
#define LLVM_CODEGEN_MACHINEREGISTERCLASSINFO_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/Pass.h"

namespace llvm {

class MachineRegisterClassInfoAnalysis
    : public AnalysisInfoMixin<MachineRegisterClassInfoAnalysis> {
  friend AnalysisInfoMixin<MachineRegisterClassInfoAnalysis>;

  static AnalysisKey Key;

public:
  using Result = RegisterClassInfo;

  Result run(MachineFunction &, MachineFunctionAnalysisManager &);
};

class MachineRegisterClassInfoWrapperPass : public MachineFunctionPass {
  virtual void anchor();

  RegisterClassInfo RCI;

public:
  static char ID;

  MachineRegisterClassInfoWrapperPass();

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  RegisterClassInfo &getRCI() { return RCI; }
  const RegisterClassInfo &getRCI() const { return RCI; }
};
} // namespace llvm

#endif
