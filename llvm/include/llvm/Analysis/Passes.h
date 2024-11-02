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

namespace llvm {
  class FunctionPass;
  class ImmutablePass;
  class ModulePass;

  //===--------------------------------------------------------------------===//
  //
  /// createLazyValueInfoPass - This creates an instance of the LazyValueInfo
  /// pass.
  FunctionPass *createLazyValueInfoPass();

  //===--------------------------------------------------------------------===//
  //
  // createDependenceAnalysisWrapperPass - This creates an instance of the
  // DependenceAnalysisWrapper pass.
  //
  FunctionPass *createDependenceAnalysisWrapperPass();

  //===--------------------------------------------------------------------===//
  //
  // createCostModelAnalysisPass - This creates an instance of the
  // CostModelAnalysis pass.
  //
  FunctionPass *createCostModelAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // createDelinearizationPass - This pass implements attempts to restore
  // multidimensional array indices from linearized expressions.
  //
  FunctionPass *createDelinearizationPass();

  //===--------------------------------------------------------------------===//
  //
  // createLegacyDivergenceAnalysisPass - This pass determines which branches in a GPU
  // program are divergent.
  //
  FunctionPass *createLegacyDivergenceAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // Minor pass prototypes, allowing us to expose them through bugpoint and
  // analyze.
  FunctionPass *createInstCountPass();

  //===--------------------------------------------------------------------===//
  //
  // createRegionInfoPass - This pass finds all single entry single exit regions
  // in a function and builds the region hierarchy.
  //
  FunctionPass *createRegionInfoPass();

  // Print module-level debug info metadata in human-readable form.
  ModulePass *createModuleDebugInfoPrinterPass();

  //===--------------------------------------------------------------------===//
  //
  // createMemDepPrinter - This pass exhaustively collects all memdep
  // information and prints it with -analyze.
  //
  FunctionPass *createMemDepPrinter();

  //===--------------------------------------------------------------------===//
  //
  // createMemDerefPrinter - This pass collects memory dereferenceability
  // information and prints it with -analyze.
  //
  FunctionPass *createMemDerefPrinter();

  //===--------------------------------------------------------------------===//
  //
  // createMustExecutePrinter - This pass collects information about which
  // instructions within a loop are guaranteed to execute if the loop header is
  // entered and prints it with -analyze.
  //
  FunctionPass *createMustExecutePrinter();

  //===--------------------------------------------------------------------===//
  //
  // createMustBeExecutedContextPrinter - This pass prints information about which
  // instructions are guaranteed to execute together (run with -analyze).
  //
  ModulePass *createMustBeExecutedContextPrinter();

}

#endif
