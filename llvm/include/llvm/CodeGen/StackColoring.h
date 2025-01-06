//===- llvm/CodeGen/StackColoring.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_STACKCOLORINGPASS_H
#define LLVM_CODEGEN_STACKCOLORINGPASS_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class StackColoringPass : public PassInfoMixin<StackColoringPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_STACKCOLORINGPASS_H
