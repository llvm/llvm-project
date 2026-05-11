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

/// Number of strict refinements `IntegerValueRangeLattice` is allowed to
/// absorb at framework merge sites before `WidenableLattice::join` forces
/// the range to its max as a sound over-approximation. Sized to be much
/// larger than realistic merge counts on naturally bounded accumulators
/// (e.g. `arith.minsi`/`arith.andi`-clamped iter args) but still a tight
/// ceiling on the +1 ratchet pathology this widening exists to cut off
/// (loop-carried ranges growing by one per worklist visit on `scf.while`
/// loops with dynamic bounds and nested region ops, which can otherwise
/// take up to ~2^31 iterations on i32 to converge).
inline constexpr unsigned kIntegerRangeWideningBudget = 128;

/// Lattice element for `IntegerRangeAnalysis`, with a per-state widening
/// budget supplied by `WidenableLattice` to guarantee convergence on
/// loop-carried values. See the comment on `WidenableLattice` for the
/// merge-path / transfer-function distinction; integer ranges' top
/// element here is `IntegerValueRange::getMaxRange(anchor)`.
class IntegerValueRangeLattice
    : public WidenableLattice<IntegerValueRange, kIntegerRangeWideningBudget> {
public:
  using WidenableLattice::WidenableLattice;

  IntegerValueRange getWidenedValue() const override {
    return IntegerValueRange::getMaxRange(cast<Value>(getAnchor()));
  }
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

  /// At an entry point, we cannot reason about integer value ranges.
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
  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange nonSuccessorInputs,
      ArrayRef<IntegerValueRangeLattice *> nonSuccessorInputLattices) override;
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
