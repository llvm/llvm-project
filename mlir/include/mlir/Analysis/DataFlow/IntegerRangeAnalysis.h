//===-IntegerRangeAnalysis.h - Integer range analysis -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dataflow analysis class for integer range inference
// so that it can be used in transformations over the `arith` dialect such as
// branch elimination or signed->unsigned rewriting.
//
// One can also implement InferIntRangeInterface on ops in custom dialects,
// and then use this analysis to propagate ranges with custom semantics.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_INTEGERANGEANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_INTEGERANGEANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

namespace mlir {
class RewriterBase;
namespace dataflow {

/// This lattice element represents the integer value range of an SSA value.
/// When this lattice is updated, it automatically updates the constant value
/// of the SSA value (if the range can be narrowed to one).
class IntegerValueRangeLattice : public Lattice<IntegerValueRange> {
public:
  using Lattice::Lattice;

  /// If the range can be narrowed to an integer constant, update the constant
  /// value of the SSA value.
  void onUpdate(DataFlowSolver *solver) const override;
};

/// Integer range analysis determines the integer value range of SSA values
/// using operations that define `InferIntRangeInterface` and also sets the
/// range of iteration indices of loops with known bounds.
///
/// This analysis depends on DeadCodeAnalysis, and will be a silent no-op
/// if DeadCodeAnalysis is not loaded in the same solver context.
class IntegerRangeAnalysis
    : public SparseForwardDataFlowAnalysis<IntegerValueRangeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// At an entry point, we cannot reason about interger value ranges.
  void setToEntryState(IntegerValueRangeLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(IntegerValueRange::getMaxRange(
                                    lattice->getAnchor())));
  }

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferIntRangeInterface`.
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const IntegerValueRangeLattice *> operands,
                 ArrayRef<IntegerValueRangeLattice *> results) override;

  /// Visit block arguments or operation results of an operation with region
  /// control-flow for which values are not defined by region control-flow. This
  /// function calls `InferIntRangeInterface` to provide values for block
  /// arguments or tries to reduce the range on loop induction variables with
  /// known bounds.
  void
  visitNonControlFlowArguments(Operation *op, const RegionSuccessor &successor,
                               ArrayRef<IntegerValueRangeLattice *> argLattices,
                               unsigned firstIndex) override;
};

/// Succeeds if an op can be converted to its unsigned equivalent without
/// changing its semantics. This is the case when none of its openands or
/// results can be below 0 when analyzed from a signed perspective.
LogicalResult staticallyNonNegative(DataFlowSolver &solver, Operation *op);

/// Succeeds when a value is statically non-negative in that it has a lower
/// bound on its value (if it is treated as signed) and that bound is
/// non-negative.
/// Note, the results of this query may not be accurate for `index` if you plan
/// to use a non-64-bit index.
LogicalResult staticallyNonNegative(DataFlowSolver &solver, Value v);

LogicalResult maybeReplaceWithConstant(DataFlowSolver &solver,
                                       RewriterBase &rewriter, Value value);

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_INTEGERANGEANALYSIS_H
