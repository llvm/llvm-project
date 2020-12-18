//===- VectorOps.h - MLIR Vector Dialect Operations -------------*- C++ -*-===//
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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

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
  /// Progressively lower to finer grained `vector.contract` and dot-products.
  Dot = 0,
  /// Lower to `vector.matrix_multiply`, maps 1-1 to LLVM matrix intrinsics.
  Matmul = 1,
  /// Lower to `vector.outerproduct`.
  OuterProduct = 2,
};
/// Enum to control the lowering of `vector.transpose` operations.
enum class VectorTransposeLowering {
  /// Lower transpose into element-wise extract and inserts.
  EltWise = 0,
  /// Lower 2-D transpose to `vector.flat_transpose`, maps 1-1 to LLVM matrix
  /// intrinsics.
  Flat = 1,
};
/// Enum to control the splitting of `vector.transfer` operations into masked
/// and unmasked variants.
enum class VectorTransferSplit {
  /// Do not split vector transfer operations.
  None = 0,
  /// Split using masked + unmasked vector.transfer operations.
  VectorTransfer = 1,
  /// Split using a unmasked vector.transfer + linalg.fill + linalg.copy
  /// operations.
  LinalgCopy = 2,
  /// Do not split vector transfer operation but instead mark it as "unmasked".
  ForceUnmasked = 3
};
/// Structure to control the behavior of vector transform patterns.
struct VectorTransformsOptions {
  /// Option to control the lowering of vector.contract.
  VectorContractLowering vectorContractLowering = VectorContractLowering::Dot;
  VectorTransformsOptions &
  setVectorTransformsOptions(VectorContractLowering opt) {
    vectorContractLowering = opt;
    return *this;
  }
  /// Option to control the lowering of vector.transpose.
  VectorTransposeLowering vectorTransposeLowering =
      VectorTransposeLowering::EltWise;
  VectorTransformsOptions &
  setVectorTransposeLowering(VectorTransposeLowering opt) {
    vectorTransposeLowering = opt;
    return *this;
  }
  /// Option to control the splitting of vector transfers.
  VectorTransferSplit vectorTransferSplit = VectorTransferSplit::None;
  VectorTransformsOptions &setVectorTransferSplit(VectorTransferSplit opt) {
    vectorTransferSplit = opt;
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
AffineMap getTransferMinorIdentityMap(ShapedType shapedType,
                                      VectorType vectorType);
} // namespace impl
} // end namespace vector
} // end namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Vector/VectorOps.h.inc"
#include "mlir/Dialect/Vector/VectorOpsDialect.h.inc"

#endif // MLIR_DIALECT_VECTOR_VECTOROPS_H
