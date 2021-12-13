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

/// Return whether `srcType` can be broadcast to `dstVectorType` under the
/// semantics of the `vector.broadcast` op.
enum class BroadcastableToResult {
  Success = 0,
  SourceRankHigher = 1,
  DimensionMismatch = 2,
  SourceTypeNotAVector = 3
};
BroadcastableToResult
isBroadcastableTo(Type srcType, VectorType dstVectorType,
                  std::pair<int, int> *mismatchingDims = nullptr);

/// Collect a set of vector-to-vector canonicalization patterns.
void populateVectorToVectorCanonicalizationPatterns(
    RewritePatternSet &patterns);

/// Collect a set of vector.shape_cast folding patterns.
void populateShapeCastFoldingPatterns(RewritePatternSet &patterns);

/// Collect a set of leading one dimension removal patterns.
///
/// These patterns insert vector.shape_cast to remove leading one dimensions
/// to expose more canonical forms of read/write/insert/extract operations.
/// With them, there are more chances that we can cancel out extract-insert
/// pairs or forward write-read pairs.
void populateCastAwayVectorLeadingOneDimPatterns(RewritePatternSet &patterns);

/// Collect a set of one dimension removal patterns.
///
/// These patterns insert rank-reducing memref.subview ops to remove one
/// dimensions. With them, there are more chances that we can avoid
/// potentially exensive vector.shape_cast operations.
void populateVectorTransferDropUnitDimsPatterns(RewritePatternSet &patterns);

/// Collect a set of patterns to flatten n-D vector transfers on contiguous
/// memref.
///
/// These patterns insert memref.collapse_shape + vector.shape_cast patterns
/// to transform multiple small n-D transfers into a larger 1-D transfer where
/// the memref contiguity properties allow it.
void populateFlattenVectorTransferPatterns(RewritePatternSet &patterns);

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

/// These patterns materialize masks for various vector ops such as transfers.
void populateVectorMaskMaterializationPatterns(RewritePatternSet &patterns,
                                               bool indexOptimizations);

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

  void print(AsmPrinter &p) const;
  static Attribute parse(AsmParser &parser, Type type);
};

/// Collects patterns to progressively lower vector.broadcast ops on high-D
/// vectors to low-D vector ops.
void populateVectorBroadcastLoweringPatterns(RewritePatternSet &patterns);

/// Collects patterns to progressively lower vector mask ops into elementary
/// selection and insertion ops.
void populateVectorMaskOpLoweringPatterns(RewritePatternSet &patterns);

/// Collects patterns to progressively lower vector.shape_cast ops on high-D
/// vectors into 1-D/2-D vector ops by generating data movement extract/insert
/// ops.
void populateVectorShapeCastLoweringPatterns(RewritePatternSet &patterns);

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
} // namespace vector
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Vector/VectorOps.h.inc"
#include "mlir/Dialect/Vector/VectorOpsDialect.h.inc"

#endif // MLIR_DIALECT_VECTOR_VECTOROPS_H
