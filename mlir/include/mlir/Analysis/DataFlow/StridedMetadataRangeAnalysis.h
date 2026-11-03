//===- StridedMetadataRange.h - Strided metadata range analysis -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_STRIDEDMETADATARANGE_H
#define MLIR_ANALYSIS_DATAFLOW_STRIDEDMETADATARANGE_H

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Interfaces/InferStridedMetadataInterface.h"

namespace mlir {
namespace dataflow {

/// This lattice element represents the strided metadata of an SSA value.
class StridedMetadataRangeLattice : public Lattice<StridedMetadataRange> {
public:
  using Lattice::Lattice;

  void setIndexBitwidth(std::optional<unsigned> width) {
    indexBitwidth = width;
  }
  StridedMetadataRange getEntryState() const;
  ChangeResult join(const StridedMetadataRange &rhs);
  ChangeResult join(const AbstractSparseLattice &rhs) override;

private:
  std::optional<unsigned> indexBitwidth;
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
  using IndexBitwidthFn = IntegerRangeAnalysis::IndexBitwidthFn;

  /// Use the callback's runtime index width at each operation. An absent width
  /// gives conservative metadata bounds.
  StridedMetadataRangeAnalysis(DataFlowSolver &solver,
                               IndexBitwidthFn getIndexBitwidth = {});

  /// At an entry point, we cannot reason about strided metadata ranges unless
  /// the type also encodes the data. For example, a memref with static layout.
  void setToEntryState(StridedMetadataRangeLattice *lattice) override;

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferStridedMetadataInterface`.
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const StridedMetadataRangeLattice *> operands,
                 ArrayRef<StridedMetadataRangeLattice *> results) override;

protected:
  StridedMetadataRangeLattice *getLatticeElement(Value value) override;

private:
  std::optional<unsigned> getWidth(Operation *op) const {
    return getIndexBitwidth ? getIndexBitwidth(op) : std::nullopt;
  }
  IndexBitwidthFn getIndexBitwidth;
};
} // namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_STRIDEDMETADATARANGE_H
