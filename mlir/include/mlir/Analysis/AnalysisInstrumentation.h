//===- AnalysisInstrumentation.h - Analysis instrumentation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_ANALYSISINSTRUMENTATION_H
#define MLIR_ANALYSIS_ANALYSISINSTRUMENTATION_H

#include "mlir/Support/TypeID.h"

namespace mlir {
class Operation;

/// Receives callbacks when an analysis is computed for an operation.
class AnalysisInstrumentation {
public:
  virtual ~AnalysisInstrumentation();
  virtual void runBeforeAnalysis(StringRef name, TypeID id, Operation *op) = 0;
  virtual void runAfterAnalysis(StringRef name, TypeID id, Operation *op) = 0;
};
} // namespace mlir

#endif // MLIR_ANALYSIS_ANALYSISINSTRUMENTATION_H
