//===- llvm/CodeGen/PostRASchedulerList.h ------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_POSTRASCHEDULERLIST_H
#define LLVM_CODEGEN_POSTRASCHEDULERLIST_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class PostRASchedulerPass : public PassInfoMixin<PostRASchedulerPass> {
  const TargetMachine *TM;

public:
  PostRASchedulerPass(const TargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }
};

} // namespace llvm

#endif // LLVM_CODEGEN_POSTRASCHEDULERLIST_H
