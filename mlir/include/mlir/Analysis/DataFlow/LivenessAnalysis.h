//===- LivenessAnalysis.h - Liveness analysis -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements liveness analysis using the sparse backward data-flow
// analysis framework. Theoretically, liveness analysis assigns liveness to each
// (value, program point) pair in the program and it is thus a dense analysis.
// However, since values are immutable in MLIR, a sparse analysis, which will
// assign liveness to each value in the program, suffices here.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_LIVENESSANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_LIVENESSANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include <optional>

namespace mlir {
namespace dataflow {

//===----------------------------------------------------------------------===//
// LivenessAnalysis
//===----------------------------------------------------------------------===//

/// This lattice represents, for a given value, whether or not it is "live". A
/// value is considered "live" iff it is being written to memory using a
/// `memref.store` operation or is needed to compute a value that is written to
/// memory using a `memref.store` operation.
/// TODO(srisrivastava): Enhance the definition of "live" in this analysis to
/// make it more accurate. Currently some values will be marked "not live" which
/// are theoretically live.
struct Liveness : public AbstractSparseLattice {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Liveness)
  using AbstractSparseLattice::AbstractSparseLattice;

  void print(raw_ostream &os) const override;

  ChangeResult markLive();

  ChangeResult meet(const AbstractSparseLattice &other) override;

  // At the beginning of the analysis, everything is marked "not live" and as
  // the analysis progresses, values are marked "live" if they are found to be
  // live.
  bool isLive = false;
};

/// An analysis that, by going backwards along the dataflow graph, annotates
/// each value with a boolean storing true iff it is "live".
class LivenessAnalysis : public SparseBackwardDataFlowAnalysis<Liveness> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  /// Flow the liveness backward starting from the `results` of the `op`.
  /// `operands` here are the operands of `op`.
  void backwardFlowLivenessFromResults(Operation *op,
                                       ArrayRef<Liveness *> operands,
                                       ArrayRef<const Liveness *> results);

  void visitOperation(Operation *op, ArrayRef<Liveness *> operands,
                      ArrayRef<const Liveness *> results) override;

  void visitBranchOperand(OpOperand &operand) override;

  void setToExitState(Liveness *lattice) override;
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_LIVENESSANALYSIS_H
