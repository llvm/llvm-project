//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_ANALYSIS_ANALYSISOPTIONS_H
#define LLVM_LIB_ANALYSIS_ANALYSISOPTIONS_H

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include <optional>

namespace llvm {

enum class OutputCostKind {
  RecipThroughput,
  Latency,
  CodeSize,
  SizeAndLatency,
  All,
};

enum class IntrinsicCostStrategy {
  InstructionCost,
  IntrinsicCost,
  TypeBasedIntrinsicCost,
};

// Types of dependence test routines.
enum class DependenceTestType {
  Default, // All tests except BanerjeeMIV
  All,
  StrongSIV,
  WeakCrossingSIV,
  ExactSIV,
  WeakZeroSIV,
  ExactRDIV,
  GCDMIV,
  BanerjeeMIV,
};

enum class ForceSummaryHotness { None, AllNonCritical, All };

enum class InlinePriorityMode : int { Size, Cost, CostBenefit, ML };

enum class StencilMergePolicy { Off, Auto, Force };

enum class SkipMLPolicyCriteria { Never, IfCallerIsNotCold };

} // namespace llvm

#define OPTIONS_STRUCT_DECL
#include "AnalysisOptions.inc"

namespace llvm {

class Loop;
class Module;

inline const AnalysisOptions &getAnalysisOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<AnalysisOptions>();
}

inline const AnalysisOptions &getAnalysisOptions(const Function &F) {
  return getAnalysisOptions(F.getContext());
}

const AnalysisOptions &getAnalysisOptions(const Module &M);
const AnalysisOptions &getAnalysisOptions(const Loop &L);

} // namespace llvm

#endif // LLVM_LIB_ANALYSIS_ANALYSISOPTIONS_H
