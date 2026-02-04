//===- llvm/CodeGen/BranchFoldingPass.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CODEGEN_BRANCHFOLDINGPASS_H
#define LLVM_CODEGEN_BRANCHFOLDINGPASS_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class BranchFolderPass : public PassInfoMixin<BranchFolderPass> {
  bool EnableTailMerge;

public:
  BranchFolderPass(bool EnableTailMerge) : EnableTailMerge(EnableTailMerge) {}
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setNoPHIs();
  }
};

} // namespace llvm

#endif // LLVM_CODEGEN_BRANCHFOLDINGPASS_H
