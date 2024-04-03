//===- Transforms.h - EmitC transformations as patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_EMITC_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace emitc {

//===----------------------------------------------------------------------===//
// Expression transforms
//===----------------------------------------------------------------------===//

ExpressionOp createExpression(Operation *op, OpBuilder &builder);

//===----------------------------------------------------------------------===//
// Populate functions
//===----------------------------------------------------------------------===//

/// Populates `patterns` with expression-related patterns.
void populateExpressionPatterns(RewritePatternSet &patterns);

} // namespace emitc
} // namespace mlir

#endif // MLIR_DIALECT_EMITC_TRANSFORMS_TRANSFORMS_H
