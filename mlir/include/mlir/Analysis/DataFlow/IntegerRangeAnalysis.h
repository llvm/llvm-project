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
// branch elimination or signed->unsigned rewriting
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_INTEGERANGEANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_INTEGERANGEANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

namespace mlir {
namespace dataflow {

/// This lattice value represents the integer range of an SSA value.
class IntegerValueRange {
public:
  /// Create a maximal range ([0, uint_max(t)] / [int_min(t), int_max(t)])
  /// range that is used to mark the value as unable to be analyzed further,
  /// where `t` is the type of `value`.
  static IntegerValueRange getMaxRange(Value value);

  /// Create an integer value range lattice value.
  IntegerValueRange(Optional<ConstantIntRanges> value = std::nullopt)
      : value(std::move(value)) {}

  /// Whether the range is uninitialized. This happens when the state hasn't
  /// been set during the analysis.
  bool isUninitialized() const { return !value.has_value(); }

  /// Get the known integer value range.
  const ConstantIntRanges &getValue() const {
    assert(!isUninitialized());
    return *value;
  }

  /// Compare two ranges.
  bool operator==(const IntegerValueRange &rhs) const {
    return value == rhs.value;
  }

  /// Take the union of two ranges.
  static IntegerValueRange join(const IntegerValueRange &lhs,
                                const IntegerValueRange &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    return IntegerValueRange{lhs.getValue().rangeUnion(rhs.getValue())};
  }

  /// Print the integer value range.
  void print(raw_ostream &os) const { os << value; }

private:
  /// The known integer value range.
  Optional<ConstantIntRanges> value;
};

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
class IntegerRangeAnalysis
    : public SparseDataFlowAnalysis<IntegerValueRangeLattice> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  /// At an entry point, we cannot reason about interger value ranges.
  void setToEntryState(IntegerValueRangeLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(IntegerValueRange::getMaxRange(
                                    lattice->getPoint())));
  }

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferIntRangeInterface`.
  void visitOperation(Operation *op,
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

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_INTEGERANGEANALYSIS_H
