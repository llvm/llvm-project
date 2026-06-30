//===-- CopyProf.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the instrumentation passes for CopyProf that insert
// callbacks into special member functions, and add store instrumentation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_COPYPROF_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_COPYPROF_H

#include "llvm/IR/PassManager.h"
namespace llvm {

// Early-stage pass that instruments special member functions to call into the
// CopyProf runtime.
class CopyProfPass : public PassInfoMixin<CopyProfPass> {
public:
  CopyProfPass() = default;
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

  static bool isRequired() { return true; }
};

// Module-level pass that inserts the CopyProf runtime initialization
// constructor and hooks it into @llvm.global_ctors.
class ModuleCopyProfPass : public PassInfoMixin<ModuleCopyProfPass> {
public:
  ModuleCopyProfPass() = default;
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }
};

// Late-stage pass that instruments store instructions to detect whether an
// object copy has been modified before it is destructed.
class CopyProfStoresPass : public PassInfoMixin<CopyProfStoresPass> {
public:
  CopyProfStoresPass() = default;
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_COPYPROF_H
