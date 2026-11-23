//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_IFCONVERSION_H
#define LLVM_CODEGEN_IFCONVERSION_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class IfConverterPass : public OptionalPassInfoMixin<IfConverterPass> {
  std::function<bool(const MachineFunction &)> PredicateFtor;

public:
  IfConverterPass(std::function<bool(const MachineFunction &)> Ftor = nullptr)
      : PredicateFtor(Ftor) {}
  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_IFCONVERSION_H
