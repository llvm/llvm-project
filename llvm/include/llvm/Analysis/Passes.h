//===-- llvm/Analysis/Passes.h - Constructors for analyses ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the analysis libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PASSES_H
#define LLVM_ANALYSIS_PASSES_H

#include "llvm/Support/Compiler.h"

namespace llvm {
  class FunctionPass;
  class ImmutablePass;
  class ModulePass;

  //===--------------------------------------------------------------------===//
  //
  /// createLazyValueInfoPass - This creates an instance of the LazyValueInfo
  /// pass.
  LLVM_FUNC_ABI FunctionPass *createLazyValueInfoPass();

  //===--------------------------------------------------------------------===//
  //
  // createDependenceAnalysisWrapperPass - This creates an instance of the
  // DependenceAnalysisWrapper pass.
  //
  LLVM_FUNC_ABI FunctionPass *createDependenceAnalysisWrapperPass();

  //===--------------------------------------------------------------------===//
  //
  // createCostModelAnalysisPass - This creates an instance of the
  // CostModelAnalysis pass.
  //
  LLVM_FUNC_ABI FunctionPass *createCostModelAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // createDelinearizationPass - This pass implements attempts to restore
  // multidimensional array indices from linearized expressions.
  //
  LLVM_FUNC_ABI FunctionPass *createDelinearizationPass();

  //===--------------------------------------------------------------------===//
  //
  // Minor pass prototypes, allowing us to expose them through bugpoint and
  // analyze.
  LLVM_FUNC_ABI FunctionPass *createInstCountPass();

  //===--------------------------------------------------------------------===//
  //
  // createRegionInfoPass - This pass finds all single entry single exit regions
  // in a function and builds the region hierarchy.
  //
  LLVM_FUNC_ABI FunctionPass *createRegionInfoPass();
}

#endif
