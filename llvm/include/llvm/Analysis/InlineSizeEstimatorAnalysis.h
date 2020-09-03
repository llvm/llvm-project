//===- InlineSizeEstimatorAnalysis.h - ML size estimator --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_ANALYSIS_INLINESIZEESTIMATORANALYSIS_H
#define LLVM_ANALYSIS_INLINESIZEESTIMATORANALYSIS_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Function;

class TFModelEvaluator;
class InlineSizeEstimatorAnalysis
    : public AnalysisInfoMixin<InlineSizeEstimatorAnalysis> {
public:
  InlineSizeEstimatorAnalysis();
  InlineSizeEstimatorAnalysis(InlineSizeEstimatorAnalysis &&);
  ~InlineSizeEstimatorAnalysis();

  static AnalysisKey Key;
  using Result = Optional<size_t>;
  Result run(const Function &F, FunctionAnalysisManager &FAM);
  static bool isEvaluatorRequested();

private:
  std::unique_ptr<TFModelEvaluator> Evaluator;
};
} // namespace llvm
#endif // LLVM_ANALYSIS_INLINESIZEESTIMATORANALYSIS_H