//===- IntegerDivisibility.h - Divisibility analysis -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_ANALYSIS_INTEGERDIVISIBILITY_H
#define MLIR_DIALECT_ARITH_ANALYSIS_INTEGERDIVISIBILITY_H

#include "mlir/Analysis/DataFlow/IntegerDivisibilityAnalysis.h"

namespace mlir::arith {
/// Infer power-of-two divisibility using the existing divisibility lattice and
/// fixed-point/control-flow analysis. Support arith integer constants,
/// add/subtract/multiply/select, and loop induction variables with bounds
/// representable in int64_t. Other operation transfer functions are
/// pessimistic.
///
/// Restrict facts to powers of two so modular arithmetic cannot invalidate
/// them. Nonzero divisor products are saturated rather than encoded as zero:
/// a zero divisor is reserved for a value known to be zero. Wide constants are
/// supported without narrowing or taking signed absolute values.
///
/// Like IntegerDivisibilityAnalysis, this requires DeadCodeAnalysis. Load
/// SparseConstantPropagation to analyze loop induction variables. Results must
/// not be reused after modifying the analyzed IR.
class PowerOfTwoDivisibilityAnalysis
    : public dataflow::IntegerDivisibilityAnalysis {
public:
  using IntegerDivisibilityAnalysis::IntegerDivisibilityAnalysis;

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerDivisibilityLattice *> operands,
      ArrayRef<dataflow::IntegerDivisibilityLattice *> results) override;

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange successorInputs,
      ArrayRef<dataflow::IntegerDivisibilityLattice *> argLattices) override;
};
} // namespace mlir::arith

#endif // MLIR_DIALECT_ARITH_ANALYSIS_INTEGERDIVISIBILITY_H
