//===--------- HipStdPar.h - Standard Parallelism passes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// AcceleratorCodeSelection - Identify all functions reachable from a kernel,
/// removing those that are unreachable.
///
/// AllocationInterposition - Forward calls to allocation / deallocation
//  functions to runtime provided equivalents that allocate memory that is
//  accessible for an accelerator
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_HIPSTDPAR_HIPSTDPAR_H
#define LLVM_TRANSFORMS_HIPSTDPAR_HIPSTDPAR_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Module;

class HipStdParAcceleratorCodeSelectionPass
  : public PassInfoMixin<HipStdParAcceleratorCodeSelectionPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

class HipStdParAllocationInterpositionPass
  : public PassInfoMixin<HipStdParAllocationInterpositionPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_HIPSTDPAR_HIPSTDPAR_H
