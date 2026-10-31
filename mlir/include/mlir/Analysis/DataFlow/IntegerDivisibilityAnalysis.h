//===- IntegerDivisibilityAnalysis.h - Integer divisibility -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dataflow analysis class for integer divisibility
// inference. Operations participate in the analysis by implementing
// `InferIntDivisibilityOpInterface`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_INTEGERDIVISIBILITYANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_INTEGERDIVISIBILITYANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/InferIntDivisibilityOpInterface.h"

namespace mlir::dataflow {

/// This lattice element represents the integer divisibility of an SSA value.
class IntegerDivisibilityLattice : public Lattice<IntegerDivisibility> {
public:
  using Lattice::Lattice;
};

/// Integer divisibility analysis determines, for each integer-typed SSA
/// value, a divisor that the value is guaranteed to be a multiple of. It
/// uses operations that implement `InferIntDivisibilityOpInterface` and
/// also sets the divisibility of induction variables of loops with known
/// lower bounds and steps.
///
/// This analysis depends on DeadCodeAnalysis, and will be a silent no-op
/// if DeadCodeAnalysis is not loaded in the same solver context.
class IntegerDivisibilityAnalysis
    : public SparseForwardDataFlowAnalysis<IntegerDivisibilityLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// At an entry point, set the lattice to the most pessimistic state,
  /// indicating that no further reasoning can be done.
  void setToEntryState(IntegerDivisibilityLattice *lattice) override;

  /// Visit an operation, invoking the transfer function.
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const IntegerDivisibilityLattice *> operands,
                 ArrayRef<IntegerDivisibilityLattice *> results) override;

  /// Visit block arguments or operation results of an operation with region
  /// control-flow for which values are not defined by region control-flow. This
  /// function tries to infer the divisibility of loop induction variables based
  /// on known loop bounds and steps.
  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange successorInputs,
      ArrayRef<IntegerDivisibilityLattice *> argLattices) override;
};

} // namespace mlir::dataflow

#endif // MLIR_ANALYSIS_DATAFLOW_INTEGERDIVISIBILITYANALYSIS_H
