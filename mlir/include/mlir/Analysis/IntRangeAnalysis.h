//===- IntRangeAnalysis.h - Infer Ranges Interfaces --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dataflow analysis class for integer range inference
// so that it can be used in transformations over the `arith` dialect such as
// branch elimination or signed->unsigned rewriting
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_INTRANGEANALYSIS_H
#define MLIR_ANALYSIS_INTRANGEANALYSIS_H

#include "mlir/Interfaces/InferIntRangeInterface.h"

namespace mlir {
namespace detail {
class IntRangeAnalysisImpl;
} // end namespace detail

class IntRangeAnalysis {
public:
  /// Analyze all operations rooted under (but not including)
  /// `topLevelOperation`.
  IntRangeAnalysis(Operation *topLevelOperation);
  IntRangeAnalysis(IntRangeAnalysis &&other);
  ~IntRangeAnalysis();

  /// Get inferred range for value `v` if one exists.
  Optional<ConstantIntRanges> getResult(Value v);

private:
  std::unique_ptr<detail::IntRangeAnalysisImpl> impl;
};
} // end namespace mlir

#endif
