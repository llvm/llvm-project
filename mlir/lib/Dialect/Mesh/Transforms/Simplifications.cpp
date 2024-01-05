//===- Patterns.cpp - Mesh Patterns -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Transforms/Simplifications.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace mesh {

void populateSimplificationPatterns(RewritePatternSet &patterns) {
  populateAllReduceEndomorphismSimplificationPatterns<arith::AddFOp>(
      patterns, Partial::Sum);
  populateAllReduceEndomorphismSimplificationPatterns<arith::AddIOp>(
      patterns, Partial::Sum);

  populateAllReduceEndomorphismSimplificationPatterns<arith::MinimumFOp>(
      patterns, Partial::Min);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MinSIOp>(
      patterns, Partial::Min);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MinUIOp>(
      patterns, Partial::Min);

  populateAllReduceEndomorphismSimplificationPatterns<arith::MaximumFOp>(
      patterns, Partial::Max);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MaxSIOp>(
      patterns, Partial::Max);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MaxUIOp>(
      patterns, Partial::Max);

  // TODO: add simplifications for all-gather and other collectives.
}

} // namespace mesh
} // namespace mlir
