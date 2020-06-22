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

#ifndef MLIR_DIALECT_VECTOR_VECTOROPS_H
#define MLIR_DIALECT_VECTOR_VECTOROPS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class MLIRContext;
class OwningRewritePatternList;
namespace vector {

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

/// Enum to control the lowering of `vector.contract` operations.
enum class VectorContractLowering {
  /// Progressively lower to finer grained `vector.contract` and `vector.fma`.
  FMA = 0,
  /// Lower to `vector.matrix_multiply`, maps 1-1 to LLVM matrix intrinsics.
  Matmul = 1,
  /// Lower to `vector.outerproduct`.
  OuterProduct = 2,
};
/// Enum to control the lowering of `vector.transpose` operations.
enum class VectorTransposeLowering {
  // Lower transpose into element-wise extract and inserts.
  EltWise = 0,
  /// Lower 2-D transpose to `vector.flat_transpose`, maps 1-1 to LLVM matrix
  /// intrinsics.
  Flat = 1,
};
/// Structure to control the behavior of vector transform patterns.
struct VectorTransformsOptions {
  VectorContractLowering vectorContractLowering = VectorContractLowering::FMA;
  VectorTransposeLowering vectorTransposeLowering =
      VectorTransposeLowering::EltWise;
  VectorTransformsOptions &
  setVectorTransformsOptions(VectorContractLowering opt) {
    vectorContractLowering = opt;
    return *this;
  }
};

/// Collect a set of transformation patterns that are related to contracting
/// or expanding vector operations:
///   ContractionOpLowering,
///   ShapeCastOp2DDownCastRewritePattern,
///   ShapeCastOp2DUpCastRewritePattern
///   BroadcastOpLowering,
///   TransposeOpLowering
///   OuterproductOpLowering
/// These transformation express higher level vector ops in terms of more
/// elementary extraction, insertion, reduction, product, and broadcast ops.
void populateVectorContractLoweringPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context,
    VectorTransformsOptions vectorTransformOptions = VectorTransformsOptions());

/// Returns the integer type required for subscripts in the vector dialect.
IntegerType getVectorSubscriptType(Builder &builder);

/// Returns an integer array attribute containing the given values using
/// the integer type required for subscripts in the vector dialect.
ArrayAttr getVectorSubscriptAttr(Builder &b, ArrayRef<int64_t> values);

namespace impl {
/// Build the default minor identity map suitable for a vector transfer. This
/// also handles the case memref<... x vector<...>> -> vector<...> in which the
/// rank of the identity map must take the vector element type into account.
AffineMap getTransferMinorIdentityMap(MemRefType memRefType,
                                      VectorType vectorType);
} // namespace impl

#define GET_OP_CLASSES
#include "mlir/Dialect/Vector/VectorOps.h.inc"

#include "mlir/Dialect/Vector/VectorOpsDialect.h.inc"

} // end namespace vector
} // end namespace mlir

#endif // MLIR_DIALECT_VECTOR_VECTOROPS_H
