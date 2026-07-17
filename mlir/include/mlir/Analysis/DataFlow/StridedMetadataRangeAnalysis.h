//===- StridedMetadataRange.h - Strided metadata range analysis -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_STRIDEDMETADATARANGE_H
#define MLIR_ANALYSIS_DATAFLOW_STRIDEDMETADATARANGE_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/InferStridedMetadataInterface.h"

namespace mlir {
namespace dataflow {

/// This lattice element represents the strided metadata of an SSA value.
class StridedMetadataRangeLattice : public Lattice<StridedMetadataRange> {
public:
  using Lattice::Lattice;
};

/// Strided metadata range analysis determines the strided metadata ranges of
/// SSA values using operations that define `InferStridedMetadataInterface`.
///
/// This analysis depends on DeadCodeAnalysis, SparseConstantPropagation, and
/// IntegerRangeAnalysis, and will be a silent no-op if the analyses are not
/// loaded in the same solver context.
class StridedMetadataRangeAnalysis
    : public SparseForwardDataFlowAnalysis<StridedMetadataRangeLattice> {
public:
  StridedMetadataRangeAnalysis(DataFlowSolver &solver,
                               int32_t indexBitwidth = 64);

  /// At an entry point, we cannot reason about strided metadata ranges unless
  /// the type also encodes the data. For example, a memref with static layout.
  void setToEntryState(StridedMetadataRangeLattice *lattice) override;

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferStridedMetadataInterface`.
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const StridedMetadataRangeLattice *> operands,
                 ArrayRef<StridedMetadataRangeLattice *> results) override;

private:
  /// Index bitwidth to use when operating with the int-ranges.
  int32_t indexBitwidth = 64;
};
} // namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_STRIDEDMETADATARANGE_H
