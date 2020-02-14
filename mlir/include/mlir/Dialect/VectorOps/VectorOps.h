//===- VectorOps.h - MLIR Super Vectorizer Operations -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Vector dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOROPS_VECTOROPS_H
#define MLIR_DIALECT_VECTOROPS_VECTOROPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
class MLIRContext;
class OwningRewritePatternList;
namespace vector {

/// Dialect for Ops on higher-dimensional vector types.
class VectorOpsDialect : public Dialect {
public:
  VectorOpsDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "vector"; }

  /// Materialize a single constant operation from a given attribute value with
  /// the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

/// Collect a set of vector-to-vector canonicalization patterns.
void populateVectorToVectorCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context);

/// Collect a set of vector-to-vector transformation patterns.
void populateVectorToVectorTransformationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context);

/// Collect a set of vector slices transformation patterns:
///    ExtractSlicesOpLowering, InsertSlicesOpLowering
/// Useful for clients that want to express all vector "slices"
/// ops in terms of more elementary vector "slice" ops. If all
/// "produced" tuple values are "consumed" (the most common
/// use for "slices" ops), this lowering removes all tuple related
/// operations as well (through DCE and folding). If tuple values
/// "leak" coming in, however, some tuple related ops will remain.
void populateVectorSlicesLoweringPatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *context);

/// Collect a set of vector contraction transformation patterns
/// that express all vector.contract ops in terms of more elementary
/// extraction and reduction ops.
void populateVectorContractLoweringPatterns(OwningRewritePatternList &patterns,
                                            MLIRContext *context);

/// Returns the integer type required for subscripts in the vector dialect.
IntegerType getVectorSubscriptType(Builder &builder);

/// Returns an integer array attribute containing the given values using
/// the integer type required for subscripts in the vector dialect.
ArrayAttr getVectorSubscriptAttr(Builder &b, ArrayRef<int64_t> values);

#define GET_OP_CLASSES
#include "mlir/Dialect/VectorOps/VectorOps.h.inc"

} // end namespace vector
} // end namespace mlir

#endif // MLIR_DIALECT_VECTOROPS_VECTOROPS_H
