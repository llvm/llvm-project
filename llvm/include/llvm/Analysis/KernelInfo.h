//=- KernelInfo.h - Kernel Analysis -------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the KernelInfoPrinter class used to emit remarks about
// function properties from a GPU kernel.
//
// See llvm/docs/KernelInfo.rst.
// ===---------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_KERNELINFO_H
#define LLVM_ANALYSIS_KERNELINFO_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class TargetMachine;

class KernelInfoPrinter : public PassInfoMixin<KernelInfoPrinter> {
  TargetMachine *TM;

public:
  explicit KernelInfoPrinter(TargetMachine *TM) : TM(TM) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};
} // namespace llvm
#endif // LLVM_ANALYSIS_KERNELINFO_H
