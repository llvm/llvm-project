//===- AffineToStandard.h - Convert Affine to Standard dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_AFFINETOSTANDARD_AFFINETOSTANDARD_H
#define MLIR_CONVERSION_AFFINETOSTANDARD_AFFINETOSTANDARD_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class AffineExpr;
class AffineForOp;
class AffineMap;
class AffineParallelOp;
class Location;
struct LogicalResult;
class OpBuilder;
class Pass;
class RewritePattern;
class Value;
class ValueRange;

class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

/// Emit code that computes the given affine expression using standard
/// arithmetic operations applied to the provided dimension and symbol values.
Value expandAffineExpr(OpBuilder &builder, Location loc, AffineExpr expr,
                       ValueRange dimValues, ValueRange symbolValues);

/// Create a sequence of operations that implement the `affineMap` applied to
/// the given `operands` (as it it were an AffineApplyOp).
Optional<SmallVector<Value, 8>> expandAffineMap(OpBuilder &builder,
                                                Location loc,
                                                AffineMap affineMap,
                                                ValueRange operands);

/// Collect a set of patterns to convert from the Affine dialect to the Standard
/// dialect, in particular convert structured affine control flow into CFG
/// branch-based control flow.
void populateAffineToStdConversionPatterns(OwningRewritePatternList &patterns);

/// Collect a set of patterns to convert vector-related Affine ops to the Vector
/// dialect.
void populateAffineToVectorConversionPatterns(
    OwningRewritePatternList &patterns);

/// Emit code that computes the lower bound of the given affine loop using
/// standard arithmetic operations.
Value lowerAffineLowerBound(AffineForOp op, OpBuilder &builder);

/// Emit code that computes the upper bound of the given affine loop using
/// standard arithmetic operations.
Value lowerAffineUpperBound(AffineForOp op, OpBuilder &builder);

/// Lowers affine control flow operations (ForStmt, IfStmt and AffineApplyOp)
/// to equivalent lower-level constructs (flow of basic blocks and arithmetic
/// primitives).
std::unique_ptr<Pass> createLowerAffinePass();

} // namespace mlir

#endif // MLIR_CONVERSION_AFFINETOSTANDARD_AFFINETOSTANDARD_H
