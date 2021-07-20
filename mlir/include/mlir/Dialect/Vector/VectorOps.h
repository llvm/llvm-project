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

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/StringExtras.h"

// Pull in all enum type definitions and utility function declarations.
#include "mlir/Dialect/Vector/VectorOpsEnums.h.inc"

namespace mlir {
class MLIRContext;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

namespace vector {
class VectorDialect;

namespace detail {
struct BitmaskEnumStorage;
} // namespace detail

/// Collect a set of vector-to-vector canonicalization patterns.
void populateVectorToVectorCanonicalizationPatterns(
    RewritePatternSet &patterns);

/// Collect a set of leading one dimension removal patterns.
///
/// These patterns insert vector.shape_cast to remove leading one dimensions
/// to expose more canonical forms of read/write/insert/extract operations.
/// With them, there are more chances that we can cancel out extract-insert
/// pairs or forward write-read pairs.
void populateCastAwayVectorLeadingOneDimPatterns(RewritePatternSet &patterns);

/// Collect a set of patterns that bubble up/down bitcast ops.
///
/// These patterns move vector.bitcast ops to be before insert ops or after
/// extract ops where suitable. With them, bitcast will happen on smaller
/// vectors and there are more chances to share extract/insert ops.
void populateBubbleVectorBitCastOpPatterns(RewritePatternSet &patterns);

/// Collect a set of transfer read/write lowering patterns.
///
/// These patterns lower transfer ops to simpler ops like `vector.load`,
/// `vector.store` and `vector.broadcast`. Only transfers with a transfer rank
/// of a most `maxTransferRank` are lowered. This is useful when combined with
/// VectorToSCF, which reduces the rank of vector transfer ops.
void populateVectorTransferLoweringPatterns(
    RewritePatternSet &patterns,
    llvm::Optional<unsigned> maxTransferRank = llvm::None);

/// Collect a set of transfer read/write lowering patterns that simplify the
/// permutation map (e.g., converting it to a minor identity map) by inserting
/// broadcasts and transposes.
void populateVectorTransferPermutationMapLoweringPatterns(
    RewritePatternSet &patterns);

/// These patterns materialize masks for various vector ops such as transfers.
void populateVectorMaskMaterializationPatterns(RewritePatternSet &patterns,
                                               bool enableIndexOptimizations);

// Collect a set of patterns to convert vector.multi_reduction op into
// a sequence of vector.reduction ops.
void populateVectorMultiReductionLoweringPatterns(RewritePatternSet &patterns);

/// Collect a set of patterns to propagate insert_map/extract_map in the ssa
/// chain.
void populatePropagateVectorDistributionPatterns(RewritePatternSet &patterns);

/// An attribute that specifies the combining function for `vector.contract`,
/// and `vector.reduction`.
class CombiningKindAttr
    : public Attribute::AttrBase<CombiningKindAttr, Attribute,
                                 detail::BitmaskEnumStorage> {
public:
  using Base::Base;

  static CombiningKindAttr get(CombiningKind kind, MLIRContext *context);

  CombiningKind getKind() const;

  void print(DialectAsmPrinter &p) const;
  static Attribute parse(DialectAsmParser &parser);
};

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
/// Enum to control the splitting of `vector.transfer` operations into
/// in-bounds and out-of-bounds variants.
enum class VectorTransferSplit {
  /// Do not split vector transfer operations.
  None = 0,
  /// Split using in-bounds + out-of-bounds vector.transfer operations.
  VectorTransfer = 1,
  /// Split using an in-bounds vector.transfer + linalg.fill + linalg.copy
  /// operations.
  LinalgCopy = 2,
  /// Do not split vector transfer operation but instead mark it as "in-bounds".
  ForceInBounds = 3
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
///   OuterproductOpLowering
/// These transformation express higher level vector ops in terms of more
/// elementary extraction, insertion, reduction, product, and broadcast ops.
void populateVectorContractLoweringPatterns(
    RewritePatternSet &patterns,
    VectorTransformsOptions vectorTransformOptions = VectorTransformsOptions());

/// Insert TransposeLowering patterns into extraction/insertion.
void populateVectorTransposeLoweringPatterns(
    RewritePatternSet &patterns,
    VectorTransformsOptions vectorTransformOptions = VectorTransformsOptions());

/// Returns the integer type required for subscripts in the vector dialect.
IntegerType getVectorSubscriptType(Builder &builder);

/// Returns an integer array attribute containing the given values using
/// the integer type required for subscripts in the vector dialect.
ArrayAttr getVectorSubscriptAttr(Builder &b, ArrayRef<int64_t> values);

/// Returns the value obtained by reducing the vector into a scalar using the
/// operation kind associated with a binary AtomicRMWKind op.
Value getVectorReductionOp(AtomicRMWKind op, OpBuilder &builder, Location loc,
                           Value vector);

/// Return true if the last dimension of the MemRefType has unit stride. Also
/// return true for memrefs with no strides.
bool isLastMemrefDimUnitStride(MemRefType type);

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
