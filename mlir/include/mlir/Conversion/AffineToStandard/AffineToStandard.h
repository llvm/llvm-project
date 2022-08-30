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
class AffineForOp;
class Location;
struct LogicalResult;
class OpBuilder;
class Pass;
class RewritePattern;
class Value;
class ValueRange;

class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTAFFINETOSTANDARDPASS
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the Affine dialect to the Standard
/// dialect, in particular convert structured affine control flow into CFG
/// branch-based control flow.
void populateAffineToStdConversionPatterns(RewritePatternSet &patterns);

/// Collect a set of patterns to convert vector-related Affine ops to the Vector
/// dialect.
void populateAffineToVectorConversionPatterns(RewritePatternSet &patterns);

/// Emit code that computes the lower bound of the given affine loop using
/// standard arithmetic operations.
Value lowerAffineLowerBound(AffineForOp op, OpBuilder &builder);

/// Emit code that computes the upper bound of the given affine loop using
/// standard arithmetic operations.
Value lowerAffineUpperBound(AffineForOp op, OpBuilder &builder);

} // namespace mlir

#endif // MLIR_CONVERSION_AFFINETOSTANDARD_AFFINETOSTANDARD_H
