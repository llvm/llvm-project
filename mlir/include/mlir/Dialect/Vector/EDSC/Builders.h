//===- Builders.h - MLIR Declarative Vector Builders ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides intuitive composable interfaces for building structured MLIR
// snippets in a declarative fashion.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_VECTOR_EDSC_BUILDERS_H_
#define MLIR_DIALECT_VECTOR_EDSC_BUILDERS_H_

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace edsc {
namespace ops {

/// Build a generic vector contraction, that is a `vector.contract` op with
/// specified `iteratorTypes`. The client is responsible for specifying proper
/// indexings when creating the StructuredIndexed.
/// The computation represents a notional (A * B + C) where indexings specify
/// which dimensions are reduced and reordered.
/// Return the result of the `vector.contract` op
///
/// Prerequisites:
/// A, B and C capture values of proper vector types, and indexing expressions
/// that match semantics of the `vector.contract` op.
Value vector_contraction(StructuredIndexed A, StructuredIndexed B,
                         StructuredIndexed C,
                         ArrayRef<IteratorType> iteratorTypes);

/// Build a generic vector contraction that computes a matmul on vectors.
/// Return the result of C(i, j) + sum_k {A(i, k) * B(k, j)} on vectors.
///
/// Prerequisites:
/// A, B and C capture values of proper vector types. For instance
/// `A: vector<4x8xf32>`, `B: vector<8x16f32>` and `C: vector<4x16xf32>`.
Value vector_contraction_matmul(Value A, Value B, Value C);

} // namespace ops
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_EDSC_BUILDERS_H_
