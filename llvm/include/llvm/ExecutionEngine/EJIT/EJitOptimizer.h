//===-- EJitOptimizer.h - JIT Optimization Pipeline -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITOPTIMIZER_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITOPTIMIZER_H

#include "llvm/ExecutionEngine/EJIT/EJitCommon.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptions.h"
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/Module.h"
#include "llvm/ExecutionEngine/EJIT/EJitPassBuilder.h"

namespace llvm {
namespace ejit {

/// JIT optimization pipeline. Runs on the extracted bitcode module during
/// JIT compilation to specialize the code for the current time-window values.
/// Holds persistent AnalysisManagers to avoid re-registering analyses on
/// every compilation.
class EJitOptimizer {
public:
  EJitOptimizer(PeriodArrayRegistry &reg);

  /// Run the full JIT specialization pipeline:
  ///   1. Parameter substitution (ejit_period_arr_ind → constants)
  ///   2. InstCombine (fold GEP chains from substituted params)
  ///   3. StructFieldPass (may_const loads → constants)
  ///   4. Core optimization pipeline (L1/L2/L3)
  ///   5. Cleanup passes (only for L2/L3 which may expose new loads)
  void runPipeline(Module &M, const SpecializationContext &ctx);

  /// Clear all cached analysis results. Must be called between compilations
  /// to avoid dangling pointers to IR units from previous modules.
  void clearAnalyses();

private:
  /// Replace ejit_period_arr_ind parameters with their runtime constants.
  void preReplacePeriodIndices(Module &M, const SpecializationContext &ctx);

  /// Run InstCombine on all functions (single pass).
  void runInstCombine(Module &M);

  /// Run EJitStructFieldPass on all functions.
  void runStructFieldPass(Module &M);

  /// Run core optimization: L1 = SCCP+ADCE+SimplifyCFG,
  /// L2 = + AlwaysInliner + cleanup, L3 = + LoopUnroll + cleanup.
  void runOptimizationPipeline(Module &M, OptimizationLevel level);

  PeriodArrayRegistry &registry_;

  // Persistent analysis managers — registered once, reused across compilations.
  // Invalidated per-function by the pass infrastructure as needed.
  LoopAnalysisManager LAM_;
  FunctionAnalysisManager FAM_;
  CGSCCAnalysisManager CGAM_;
  ModuleAnalysisManager MAM_;

  // Cached L1 pass pipeline — SCCP+ADCE+SimplifyCFG combination created once.
  FunctionPassManager L1FPM_;
};

} // namespace ejit
} // namespace llvm

#endif
