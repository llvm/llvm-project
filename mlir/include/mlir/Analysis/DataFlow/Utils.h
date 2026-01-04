//===-Utils.h - DataFlow utility functions ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utility functions for dataflow analyses.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_UTILS_H
#define MLIR_ANALYSIS_DATAFLOW_UTILS_H

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir {
namespace dataflow {

/// Populates a DataFlowSolver with analyses that are required to ensure
/// user-defined analyses are run properly.
///
/// This helper is intended to be an interim fix until a more robust solution
/// can be implemented in the DataFlow framework directly. Cf.
/// https://discourse.llvm.org/t/mlir-dead-code-analysis/67568
inline void loadBaselineAnalyses(DataFlowSolver &solver) {
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
}

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_INTEGERANGEANALYSIS_H
