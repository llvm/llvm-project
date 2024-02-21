//===- VectorTransforms.cpp - Conversion within the Vector dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites as 1->N patterns.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "vector-to-vector"

using namespace mlir;
using namespace mlir::vector;

template <typename IntType>
static SmallVector<IntType> extractVector(ArrayAttr arrayAttr) {
  return llvm::to_vector<4>(llvm::map_range(
      arrayAttr.getAsRange<IntegerAttr>(),
      [](IntegerAttr attr) { return static_cast<IntType>(attr.getInt()); }));
}

// Helper to find an index in an affine map.
static std::optional<int64_t> getResultIndex(AffineMap map, int64_t index) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
    int64_t idx = map.getDimPosition(i);
    if (idx == index)
      return i;
  }
  return std::nullopt;
}

namespace {

/// ShapeCastOpFolder folds cancelling ShapeCastOps away.
//
// Example:
//
//  The following MLIR with cancelling ShapeCastOps:
//
//   %0 = source : vector<5x4x2xf32>
//   %1 = shape_cast %0 : vector<5x4x2xf32> to vector<20x2xf32>
//   %2 = shape_cast %1 : vector<20x2xf32> to vector<5x4x2xf32>
//   %3 = user %2 : vector<5x4x2xf32>
//
//  Should canonicalize to the following:
//
//   %0 = source : vector<5x4x2xf32>
//   %1 = user %0 : vector<5x4x2xf32>
//
struct ShapeCastOpFolder : public OpRewritePattern<vector::ShapeCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                PatternRewriter &rewriter) const override {
    // Check if 'shapeCastOp' has vector source/result type.
    auto sourceVectorType =
        dyn_cast_or_null<VectorType>(shapeCastOp.getSource().getType());
    auto resultVectorType =
        dyn_cast_or_null<VectorType>(shapeCastOp.getResult().getType());
    if (!sourceVectorType || !resultVectorType)
      return failure();

    // Check if shape cast op source operand is also a shape cast op.
    auto sourceShapeCastOp = dyn_cast_or_null<vector::ShapeCastOp>(
        shapeCastOp.getSource().getDefiningOp());
    if (!sourceShapeCastOp)
      return failure();
    auto operandSourceVectorType =
        cast<VectorType>(sourceShapeCastOp.getSource().getType());
    auto operandResultVectorType = sourceShapeCastOp.getType();

    // Check if shape cast operations invert each other.
    if (operandSourceVectorType != resultVectorType ||
        operandResultVectorType != sourceVectorType)
      return failure();

    rewriter.replaceOp(shapeCastOp, sourceShapeCastOp.getSource());
    return success();
  }
};

/// Convert MulIOp/MulFOp + MultiDimReductionOp<add> into ContractionOp.
/// Ex:
/// ```
///   %0 = arith.mulf %arg0, %arg1 : vector<8x32x16xf32>
///   %1 = vector.multi_reduction add, %0 [1]
///     : vector<8x32x16xf32> to vector<8x16xf32>
/// ```
/// Gets converted to:
/// ```
///   %1 = vector.contract {indexing_maps = [
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1)>],
///    iterator_types = ["parallel", "parallel", "reduction"],
///    kind = add} %0, %arg1, %cst_f0
///    : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
///  ```
struct MultiReduceToContract
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (reduceOp.getKind() != vector::CombiningKind::ADD)
      return failure();
    Operation *mulOp = reduceOp.getSource().getDefiningOp();
    if (!mulOp || !isa<arith::MulIOp, arith::MulFOp>(mulOp))
      return failure();
    SmallVector<bool> reductionMask = reduceOp.getReductionMask();
    auto srcMap = rewriter.getMultiDimIdentityMap(reductionMask.size());
    SmallVector<AffineExpr> exprs;
    SmallVector<vector::IteratorType> iteratorTypes;
    for (const auto &isReduceDim : llvm::enumerate(reductionMask)) {
      if (!isReduceDim.value()) {
        iteratorTypes.push_back(vector::IteratorType::parallel);
        exprs.push_back(rewriter.getAffineDimExpr(isReduceDim.index()));
      } else {
        iteratorTypes.push_back(vector::IteratorType::reduction);
      }
    }
    auto dstMap =
        AffineMap::get(/*dimCount=*/reductionMask.size(),
                       /*symbolCount=*/0, exprs, reduceOp.getContext());
    rewriter.replaceOpWithNewOp<mlir::vector::ContractionOp>(
        reduceOp, mulOp->getOperand(0), mulOp->getOperand(1), reduceOp.getAcc(),
        rewriter.getAffineMapArrayAttr({srcMap, srcMap, dstMap}),
        rewriter.getArrayAttr(llvm::to_vector(llvm::map_range(
            iteratorTypes, [&](IteratorType t) -> mlir::Attribute {
              return IteratorTypeAttr::get(rewriter.getContext(), t);
            }))));
    return success();
  }
};

/// Merge LHS/RHS (A/B) TransposeOp into ContractionOp user.
/// Ex:
/// ```
///   %0 = vector.transpose %arg0, [2, 0, 1]
///     : vector<32x16x8xf32> to vector<8x32x16xf32>
///   %1 = vector.contract {indexing_maps = [
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1)>],
///    iterator_types = ["parallel", "parallel", "reduction"],
///    kind = add} %0, %arg1, %cst_f0
///    : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
/// ```
/// Gets converted to:
/// ```
///   %1 = vector.contract {indexing_maps = [
///         affine_map<(d0, d1, d2) -> (d1, d2, d0)>,
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1)>],
///    iterator_types = ["parallel", "parallel", "reduction"],
///    kind = add} %arg0, %arg1, %cst_f0
///    : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
///  ```
struct CombineContractABTranspose final
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap> maps =
        llvm::to_vector<4>(contractOp.getIndexingMapsArray());
    Value lhs = contractOp.getLhs();
    Value rhs = contractOp.getRhs();
    size_t index = 0;
    bool changed = false;
    for (Value *operand : {&lhs, &rhs}) {
      AffineMap &map = maps[index++];
      auto transposeOp = operand->getDefiningOp<vector::TransposeOp>();
      if (!transposeOp)
        continue;
      AffineMap permutationMap = AffineMap::getPermutationMap(
          transposeOp.getPermutation(), contractOp.getContext());
      map = inversePermutation(permutationMap).compose(map);
      *operand = transposeOp.getVector();
      changed = true;
    }
    if (!changed)
      return failure();
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, lhs, rhs, contractOp.getAcc(),
        rewriter.getAffineMapArrayAttr(maps), contractOp.getIteratorTypes());
    return success();
  }
};

/// Merges accumulator and result transposes into contract.
///
/// For example:
/// ```mlir
/// %accT = vector.transpose %acc, [0, 2, 1]
///   : vector<2x8x4xf32> to vector<2x4x8xf32>
/// %contract = vector.contract {
///   indexing_maps = [
///     affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
///     affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
///     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
///   ],
///   iterator_types = ["parallel", "parallel", "parallel", "reduction"],
///   kind = #vector.kind<add>
/// } %lhs, %rhs, %accT
///   : vector<2x4x4xf32>, vector<4x8xf32> into vector<2x4x8xf32>
/// %0 = vector.transpose %contract, [0, 2, 1]
///   : vector<2x4x8xf32> to vector<2x8x4>
/// ```
/// Becomes:
/// ```mlir
/// %0 = vector.contract {
///   indexing_maps = [
///     affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
///     affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
///     affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
///   ],
///   iterator_types = ["parallel", "parallel", "parallel", "reduction"],
///   kind = #vector.kind<add>
/// } %lhs, %rhs, %acc
///   : vector<2x4x4xf32>, vector<4x8xf32> into vector<2x8x4xf32>
/// ```
struct CombineContractResultTranspose final
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp resTOp,
                                PatternRewriter &rewriter) const override {
    auto contractOp = resTOp.getVector().getDefiningOp<vector::ContractionOp>();
    if (!contractOp || !contractOp->hasOneUse())
      return failure();

    auto accTOp = contractOp.getAcc().getDefiningOp<vector::TransposeOp>();
    if (!accTOp)
      return failure();

    MLIRContext *context = contractOp.getContext();
    auto maps = llvm::to_vector<3>(contractOp.getIndexingMapsArray());
    AffineMap contractMap = maps.back();

    // Accumulator transpose performs f(A) -> B. Contract performs g(C) -> B.
    // To index into A in contract, we need revert(f)(g(C)) -> A.
    auto accTMap =
        AffineMap::getPermutationMap(accTOp.getPermutation(), context);

    // Contract performs g(C) -> D. Result transpose performs h(D) -> E.
    // To index into E in contract, we need h(g(C)) -> E.
    auto resTMap =
        AffineMap::getPermutationMap(resTOp.getPermutation(), context);
    auto combinedResMap = resTMap.compose(contractMap);

    // The accumulator and result share the same indexing map. So they should be
    // the same to be able to merge. This means combinedResMap is the same as
    // inversePermutation(accTMap).compose(contractMap), which means
    if (inversePermutation(accTMap) != resTMap)
      return failure();
    maps.back() = combinedResMap;

    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        resTOp, contractOp.getLhs(), contractOp.getRhs(), accTOp.getVector(),
        rewriter.getAffineMapArrayAttr(maps), contractOp.getIteratorTypes());
    return success();
  }
};

/// Merge BroadcastOp into ContractionOp user.
/// Ex:
/// ```
///   %0 = vector.broadcast %arg0 : vector<32x16xf32> to vector<8x32x16xf32>
///   %1 = vector.contract {indexing_maps = [
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1)>],
///    iterator_types = ["parallel", "parallel", "reduction"],
///    kind = add} %0, %arg1, %cst_f0
///    : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
/// ```
/// Gets converted to:
/// ```
///   %1 = vector.contract {indexing_maps = [
///         affine_map<(d0, d1, d2) -> (d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1)>],
///    iterator_types = ["parallel", "parallel", "reduction"],
///    kind = add} %arg0, %arg1, %cst_f0
///    : vector<32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
///  ```
struct CombineContractBroadcast
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap> maps =
        llvm::to_vector<4>(contractOp.getIndexingMapsArray());
    Value lhs = contractOp.getLhs();
    Value rhs = contractOp.getRhs();
    size_t index = 0;
    bool changed = false;
    for (Value *operand : {&lhs, &rhs}) {
      AffineMap &map = maps[index++];
      auto broadcast = operand->getDefiningOp<vector::BroadcastOp>();
      if (!broadcast)
        continue;
      // contractionOp can only take vector as operands.
      auto srcType = dyn_cast<VectorType>(broadcast.getSourceType());
      if (!srcType ||
          srcType.getRank() == broadcast.getResultVectorType().getRank())
        continue;
      int64_t rankDiff =
          broadcast.getResultVectorType().getRank() - srcType.getRank();
      bool innerDimBroadcast = false;
      SmallVector<AffineExpr> originalDims;
      for (const auto &dim : llvm::enumerate(srcType.getShape())) {
        if (dim.value() != broadcast.getResultVectorType().getDimSize(
                               rankDiff + dim.index())) {
          innerDimBroadcast = true;
          break;
        }
        originalDims.push_back(
            rewriter.getAffineDimExpr(dim.index() + rankDiff));
      }
      // Contract doesn't support inner dimension broadcast. Once this is
      // relaxed we can remove this case.
      if (innerDimBroadcast)
        continue;

      // It would be incorrect to fold a broadcast onto a reduction dimension
      // of non-unit size.
      bool nonUnitDimReductionBroadcast = false;
      for (int64_t i = 0; i < rankDiff; ++i) {
        if (broadcast.getResultVectorType().getDimSize(i) != 1 &&
            isReductionIterator(contractOp.getIteratorTypes()
                                    .getValue()[map.getDimPosition(i)])) {
          nonUnitDimReductionBroadcast = true;
          break;
        }
      }
      if (nonUnitDimReductionBroadcast)
        continue;

      AffineMap broadcastMap =
          AffineMap::get(broadcast.getResultVectorType().getRank(), 0,
                         originalDims, contractOp.getContext());
      map = broadcastMap.compose(map);
      *operand = broadcast.getSource();
      changed = true;
    }

    if (!changed)
      return failure();

    // Determine which dims are usused, now that the maps have been composed
    // with the broadcast maps.
    llvm::SmallBitVector unusedDimsBitVector = getUnusedDimsBitVector(maps);
    // Compress unused dims.
    for (auto &m : maps)
      m = compressDims(m, unusedDimsBitVector);
    // Compute the combined iterators.
    SmallVector<Attribute> iterators;
    for (unsigned i = 0; i < unusedDimsBitVector.size(); ++i) {
      if (!unusedDimsBitVector.test(i))
        iterators.push_back(contractOp.getIteratorTypes().getValue()[i]);
    }
    // Check that compressing unused dims isn't removing all reduction dimension
    // pairs. For example, if the vector.contract had only one reduction
    // iterator and that was a unit-dimension created by a broadcast,
    // then we should bail here, otherwise we would create a contract without
    // a reduction dimension pair.
    bool hasReductionIteratorApplyingOnBothSides = false;
    for (unsigned i = 0; i < iterators.size(); ++i) {
      if (!isReductionIterator(iterators[i]))
        continue;
      if (getResultIndex(maps[0], i) && getResultIndex(maps[1], i)) {
        hasReductionIteratorApplyingOnBothSides = true;
        break;
      }
    }
    if (!hasReductionIteratorApplyingOnBothSides)
      return failure();

    // If the compressed maps have a dimension that is not used by either LHS or
    // RHS then the ContractionOp verifier would fail.
    if (getUnusedDimsBitVector({maps[0], maps[1]}).any())
      return failure();
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, lhs, rhs, contractOp.getAcc(),
        rewriter.getAffineMapArrayAttr(maps), rewriter.getArrayAttr(iterators));
    return success();
  }
};

/// Reorders cast(broadcast) to broadcast(cast). This makes broadcast ops and
/// contraction ops closer, which kicks in CombineContractBroadcast pattern when
/// casting ops are around these operations.
/// Ex:
/// ```
///   %0 = vector.broadcast %arg0 : vector<32x16xi8> to vector<8x32x16xi8>
///   %1 = arith.extsi %0 : vector<8x32x16xi8> to vector<8x32x16xi32>
/// ```
/// Gets converted to:
/// ```
///   %0 = arith.extsi %0 : vector<32x16xi8> to vector<32x16xi32>
///   %1 = vector.broadcast %arg0 : vector<32x16xi32> to vector<8x32x16xi32>
/// ```
struct ReorderCastOpsOnBroadcast
    : public OpInterfaceRewritePattern<CastOpInterface> {
  using OpInterfaceRewritePattern<CastOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CastOpInterface op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1)
      return failure();
    auto bcastOp = op->getOperand(0).getDefiningOp<vector::BroadcastOp>();
    if (!bcastOp)
      return failure();

    Type castResTy = getElementTypeOrSelf(op->getResult(0));
    if (auto vecTy = dyn_cast<VectorType>(bcastOp.getSourceType()))
      castResTy = vecTy.clone(castResTy);
    auto *castOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        bcastOp.getSource(), castResTy, op->getAttrs());
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        op, op->getResult(0).getType(), castOp->getResult(0));
    return success();
  }
};

/// Reorders elementwise(transpose) to transpose(elementwise). This makes
/// transpose ops and contraction ops closer, which kicks in
/// CombineContractABTranspose pattern when elementwise ops are between these
/// operations. Ex:
/// ```
/// %at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
/// %bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
/// %r = arith.addf %at, %bt : vector<2x4xf32>
/// ```
/// Gets converted to:
/// ```
/// %0 = arith.addf %a, %b : vector<4x2xf32>
/// %r = vector.transpose %0, [1, 0] : vector<2x4xf32>
/// ```
struct ReorderElementwiseOpsOnTranspose final
    : public OpTraitRewritePattern<OpTrait::Elementwise> {
  using OpTraitRewritePattern::OpTraitRewritePattern;
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumRegions() != 0)
      return failure();

    // Make sure all operands are transpose/constant ops and collect their
    // transposition maps.
    SmallVector<ArrayRef<int64_t>> transposeMaps;
    transposeMaps.reserve(op->getNumOperands());
    // Record the initial type before transposition. We'll use its shape later.
    // Any type will do here as we will check all transpose maps are the same.
    VectorType srcType;
    for (Value operand : op->getOperands()) {
      auto transposeOp = operand.getDefiningOp<vector::TransposeOp>();
      if (transposeOp) {
        transposeMaps.push_back(transposeOp.getPermutation());
        srcType = transposeOp.getSourceVectorType();
      } else if (!matchPattern(operand, m_Constant())) {
        return failure();
      }
    }
    if (transposeMaps.empty())
      return failure();
    // This is an elementwise op, so all transposed operands should have the
    // same type. We need to additionally check that all transposes uses the
    // same map.
    if (!llvm::all_equal(transposeMaps))
      return rewriter.notifyMatchFailure(op, "different transpose map");

    SmallVector<Value> srcValues;
    srcValues.reserve(op->getNumOperands());

    // If there are constant operands, we need to insert inverse transposes for
    // them. Calculate the inverse order first.
    auto order = transposeMaps.front();
    SmallVector<int64_t> invOrder(order.size());
    for (int i = 0, e = order.size(); i < e; ++i)
      invOrder[order[i]] = i;

    for (Value operand : op->getOperands()) {
      auto transposeOp = operand.getDefiningOp<vector::TransposeOp>();
      if (transposeOp) {
        srcValues.push_back(transposeOp.getVector());
      } else {
        // This is a constant. Create a reverse transpose op for it.
        auto vectorType =
            srcType.clone(cast<VectorType>(operand.getType()).getElementType());
        srcValues.push_back(rewriter.create<vector::TransposeOp>(
            operand.getLoc(), vectorType, operand, invOrder));
      }
    }

    auto vectorType = srcType.clone(
        cast<VectorType>(op->getResultTypes()[0]).getElementType());
    Operation *elementwiseOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), srcValues,
                        vectorType, op->getAttrs());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(
        op, op->getResultTypes()[0], elementwiseOp->getResult(0),
        transposeMaps.front());
    return success();
  }
};

// Returns the values in `arrayAttr` as an integer vector.
static SmallVector<int64_t> getIntValueVector(ArrayAttr arrayAttr) {
  return llvm::to_vector<4>(
      llvm::map_range(arrayAttr.getAsRange<IntegerAttr>(),
                      [](IntegerAttr attr) { return attr.getInt(); }));
}

// Shuffles vector.bitcast op after vector.extract op.
//
// This transforms IR like:
//   %0 = vector.bitcast %src : vector<4xf32> to vector<8xf16>
//   %1 = vector.extract %0[3] : f16 from vector<8xf16>
// Into:
//   %0 = vector.extract %src[1] : f32 from vector<4xf32>
//   %1 = vector.bitcast %0: vector<1xf32> to vector<2xf16>
//   %2 = vector.extract %1[1] : f16 from vector<2xf16>
struct BubbleDownVectorBitCastForExtract
    : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // Only support extracting scalars for now.
    if (extractOp.getSourceVectorType().getRank() != 1)
      return failure();

    auto castOp = extractOp.getVector().getDefiningOp<vector::BitCastOp>();
    if (!castOp)
      return failure();

    VectorType castSrcType = castOp.getSourceVectorType();
    VectorType castDstType = castOp.getResultVectorType();
    assert(castSrcType.getRank() == castDstType.getRank());

    // Fail to match if we only have one element in the cast op source.
    // This is to avoid infinite loop given that this pattern can generate
    // such cases.
    if (castSrcType.getNumElements() == 1)
      return failure();

    // Only support casting to a larger number of elements or now.
    // E.g., vector<4xf32> -> vector<8xf16>.
    if (castSrcType.getNumElements() > castDstType.getNumElements())
      return failure();

    unsigned expandRatio =
        castDstType.getNumElements() / castSrcType.getNumElements();

    auto getFirstIntValue = [](ArrayRef<OpFoldResult> values) -> uint64_t {
      assert(values[0].is<Attribute>() && "Unexpected non-constant index");
      return cast<IntegerAttr>(values[0].get<Attribute>()).getInt();
    };

    uint64_t index = getFirstIntValue(extractOp.getMixedPosition());

    // Get the single scalar (as a vector) in the source value that packs the
    // desired scalar. E.g. extract vector<1xf32> from vector<4xf32>
    Location loc = extractOp.getLoc();
    Value packedValue = rewriter.create<vector::ExtractOp>(
        loc, castOp.getSource(), index / expandRatio);
    Type packedVecType = VectorType::get(/*shape=*/{1}, packedValue.getType());
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, packedVecType, rewriter.getZeroAttr(packedVecType));
    packedValue = rewriter.create<vector::InsertOp>(loc, packedValue, zero,
                                                    /*position=*/0);

    // Cast it to a vector with the desired scalar's type.
    // E.g. f32 -> vector<2xf16>
    VectorType packedType =
        VectorType::get({expandRatio}, castDstType.getElementType());
    Value castedValue =
        rewriter.create<vector::BitCastOp>(loc, packedType, packedValue);

    // Finally extract the desired scalar.
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(extractOp, castedValue,
                                                   index % expandRatio);
    return success();
  }
};

// Shuffles vector.bitcast op after vector.extract_strided_slice op.
//
// This transforms IR like:
//    %cast = vector.bitcast %arg0: vector<4xf32> to vector<8xf16>
//     %0 = vector.extract_strided_slice %cast {
//            offsets = [4], sizes = [4], strides = [1]
//          } : vector<8xf16> to vector<4xf16>
// Into:
//   %0 = vector.extract_strided_slice %src {
//          offsets = [2], sizes = [2], strides = [1]
//        } : vector<4xf32> to vector<2xf32>
//   %1 = vector.bitcast %0 : vector<2xf32> to vector<4xf16>
struct BubbleDownBitCastForStridedSliceExtract
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto castOp = extractOp.getVector().getDefiningOp<vector::BitCastOp>();
    if (!castOp)
      return failure();

    VectorType castSrcType = castOp.getSourceVectorType();
    VectorType castDstType = castOp.getResultVectorType();
    assert(castSrcType.getRank() == castDstType.getRank());

    int64_t castSrcLastDim = castSrcType.getShape().back();
    int64_t castDstLastDim = castDstType.getShape().back();
    // Require casting to more elements for now; other cases to be implemented.
    if (castSrcLastDim > castDstLastDim)
      return failure();

    // Only accept all one strides for now.
    if (llvm::any_of(extractOp.getStrides().getAsValueRange<IntegerAttr>(),
                     [](const APInt &val) { return !val.isOne(); }))
      return failure();

    unsigned rank = extractOp.getSourceVectorType().getRank();
    assert(castDstLastDim % castSrcLastDim == 0);
    int64_t expandRatio = castDstLastDim / castSrcLastDim;

    // If we have a less number of offsets than the rank, then implicitly we
    // are selecting the full range for the last bitcasted dimension; other
    // dimensions aren't affected. Otherwise, we need to scale down the last
    // dimension's offset given we are extracting from less elements now.
    ArrayAttr newOffsets = extractOp.getOffsets();
    if (newOffsets.size() == rank) {
      SmallVector<int64_t> offsets = getIntValueVector(newOffsets);
      if (offsets.back() % expandRatio != 0)
        return failure();
      offsets.back() = offsets.back() / expandRatio;
      newOffsets = rewriter.getI64ArrayAttr(offsets);
    }

    // Similarly for sizes.
    ArrayAttr newSizes = extractOp.getSizes();
    if (newSizes.size() == rank) {
      SmallVector<int64_t> sizes = getIntValueVector(newSizes);
      if (sizes.back() % expandRatio != 0)
        return failure();
      sizes.back() = sizes.back() / expandRatio;
      newSizes = rewriter.getI64ArrayAttr(sizes);
    }

    SmallVector<int64_t> dims =
        llvm::to_vector<4>(cast<VectorType>(extractOp.getType()).getShape());
    dims.back() = dims.back() / expandRatio;
    VectorType newExtractType =
        VectorType::get(dims, castSrcType.getElementType());

    auto newExtractOp = rewriter.create<vector::ExtractStridedSliceOp>(
        extractOp.getLoc(), newExtractType, castOp.getSource(), newOffsets,
        newSizes, extractOp.getStrides());

    rewriter.replaceOpWithNewOp<vector::BitCastOp>(
        extractOp, extractOp.getType(), newExtractOp);

    return success();
  }
};

// Shuffles vector.bitcast op before vector.insert_strided_slice op.
//
// This transforms IR like:
//   %0 = vector.insert_strided_slice %src, %dst {
//          offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
//   %1 = vector.bitcast %0: vector<8xf16> to vector<4xf32>
// Into:
//   %0 = vector.bitcast %src : vector<4xf16> to vector<2xf32>
//   %1 = vector.bitcast %dst : vector<8xf16> to vector<4xf32>
//   %2 = vector.insert_strided_slice %src, %dst {
//          offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>
struct BubbleUpBitCastForStridedSliceInsert
    : public OpRewritePattern<vector::BitCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    VectorType castSrcType = bitcastOp.getSourceVectorType();
    VectorType castDstType = bitcastOp.getResultVectorType();
    assert(castSrcType.getRank() == castDstType.getRank());
    // Skip 0-D vector which will not from InsertStridedSliceOp.
    if (castSrcType.getRank() == 0)
      return failure();

    int64_t castSrcLastDim = castSrcType.getShape().back();
    int64_t castDstLastDim = castDstType.getShape().back();
    // Require casting to less elements for now; other cases to be implemented.
    if (castSrcLastDim < castDstLastDim)
      return failure();

    assert(castSrcLastDim % castDstLastDim == 0);
    int64_t shrinkRatio = castSrcLastDim / castDstLastDim;

    auto insertOp =
        bitcastOp.getSource().getDefiningOp<vector::InsertStridedSliceOp>();
    if (!insertOp)
      return failure();

    // Only accept all one strides for now.
    if (llvm::any_of(insertOp.getStrides().getAsValueRange<IntegerAttr>(),
                     [](const APInt &val) { return !val.isOne(); }))
      return failure();

    unsigned rank = insertOp.getSourceVectorType().getRank();
    // Require insert op to have the same rank for the source and destination
    // vector; other cases to be implemented.
    if (rank != insertOp.getDestVectorType().getRank())
      return failure();

    // Requires that shape of insert op src is castable to dstType.
    unsigned sourceWidth = castSrcType.getElementType().getIntOrFloatBitWidth();
    unsigned destinationWidth =
        castDstType.getElementType().getIntOrFloatBitWidth();
    unsigned numElements = destinationWidth / sourceWidth;
    if (insertOp.getSourceVectorType().getNumElements() % numElements != 0)
      return failure();

    ArrayAttr newOffsets = insertOp.getOffsets();
    assert(newOffsets.size() == rank);
    SmallVector<int64_t> offsets = getIntValueVector(newOffsets);
    if (offsets.back() % shrinkRatio != 0)
      return failure();
    offsets.back() = offsets.back() / shrinkRatio;
    newOffsets = rewriter.getI64ArrayAttr(offsets);

    SmallVector<int64_t> srcDims =
        llvm::to_vector<4>(insertOp.getSourceVectorType().getShape());
    srcDims.back() = srcDims.back() / shrinkRatio;
    VectorType newCastSrcType =
        VectorType::get(srcDims, castDstType.getElementType());

    auto newCastSrcOp = rewriter.create<vector::BitCastOp>(
        bitcastOp.getLoc(), newCastSrcType, insertOp.getSource());

    SmallVector<int64_t> dstDims =
        llvm::to_vector<4>(insertOp.getDestVectorType().getShape());
    dstDims.back() = dstDims.back() / shrinkRatio;
    VectorType newCastDstType =
        VectorType::get(dstDims, castDstType.getElementType());

    auto newCastDstOp = rewriter.create<vector::BitCastOp>(
        bitcastOp.getLoc(), newCastDstType, insertOp.getDest());

    rewriter.replaceOpWithNewOp<vector::InsertStridedSliceOp>(
        bitcastOp, bitcastOp.getType(), newCastSrcOp, newCastDstOp, newOffsets,
        insertOp.getStrides());

    return success();
  }
};

// Breaks down vector.bitcast op
//
// This transforms IR like:
//   %1 = vector.bitcast %0: vector<8xf16> to vector<4xf32>
// Into:
//   %cst = vector.splat %c0_f32 : vector<4xf32>
//   %1 = vector.extract_strided_slice %0 {
//          offsets = [0], sizes = [4], strides = [1]
//        } : vector<8xf16> to vector<4xf16>
//   %2 = vector.bitcast %1 : vector<4xf16> to vector<2xf32>
//   %4 = vector.insert_strided_slice %2, %cst {
//          offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>
//   %5 = vector.extract_strided_slice %0 {
//          offsets = [4], sizes = [4], strides = [1]
//        } : vector<8xf16> to vector<4xf16>
//   %6 = vector.bitcast %5 : vector<4xf16> to vector<2xf32>
//   %7 = vector.insert_strided_slice %6, %cst {
//          offsets = [2], strides = [1]} : vector<2xf32> into vector<4xf32>
struct BreakDownVectorBitCast : public OpRewritePattern<vector::BitCastOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  BreakDownVectorBitCast(MLIRContext *context,
                         std::function<bool(vector::BitCastOp)> controlFn,
                         PatternBenefit benefit)
      : OpRewritePattern(context, benefit), controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(vector::BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {

    if (controlFn && !controlFn(bitcastOp))
      return failure();

    VectorType castSrcType = bitcastOp.getSourceVectorType();
    VectorType castDstType = bitcastOp.getResultVectorType();
    assert(castSrcType.getRank() == castDstType.getRank());

    // Only support rank 1 case for now.
    if (castSrcType.getRank() != 1)
      return failure();

    int64_t castSrcLastDim = castSrcType.getShape().back();
    int64_t castDstLastDim = castDstType.getShape().back();
    // Require casting to less elements for now; other cases to be implemented.
    if (castSrcLastDim < castDstLastDim)
      return failure();

    assert(castSrcLastDim % castDstLastDim == 0);
    int64_t shrinkRatio = castSrcLastDim / castDstLastDim;
    // Nothing to do if it is already bitcasting to a single element.
    if (castSrcLastDim == shrinkRatio)
      return failure();

    Location loc = bitcastOp.getLoc();
    Type elemType = castDstType.getElementType();
    assert(elemType.isSignlessIntOrIndexOrFloat());

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getZeroAttr(elemType));
    Value res = rewriter.create<SplatOp>(loc, castDstType, zero);

    SmallVector<int64_t> sliceShape{castDstLastDim};
    SmallVector<int64_t> strides{1};
    VectorType newCastDstType =
        VectorType::get(SmallVector<int64_t>{castDstLastDim / shrinkRatio},
                        castDstType.getElementType());

    for (int i = 0, e = shrinkRatio; i < e; ++i) {
      Value extracted = rewriter.create<ExtractStridedSliceOp>(
          loc, bitcastOp.getSource(), ArrayRef<int64_t>{i * castDstLastDim},
          sliceShape, strides);
      Value bitcast =
          rewriter.create<BitCastOp>(loc, newCastDstType, extracted);
      res = rewriter.create<InsertStridedSliceOp>(
          loc, bitcast, res,
          ArrayRef<int64_t>{i * castDstLastDim / shrinkRatio}, strides);
    }
    rewriter.replaceOp(bitcastOp, res);
    return success();
  }

private:
  std::function<bool(BitCastOp)> controlFn;
};

/// Reorders elementwise(broadcast/splat) to broadcast(elementwise). Ex:
/// ```
/// %a = vector.broadcast %arg1 : index to vector<1x4xindex>
/// %b = vector.broadcast %arg2 : index to vector<1x4xindex>
/// %r = arith.addi %a, %b : vector<1x4xindex>
/// ```
/// Gets converted to:
/// ```
/// %r = arith.addi %arg0, %arg1 : index
/// %b = vector.broadcast %r : index to vector<1x4xindex>
/// ```
///
/// Both `vector.broadcast` and `vector.splat` are supported as broadcasting
/// ops.
struct ReorderElementwiseOpsOnBroadcast final
    : public OpTraitRewritePattern<OpTrait::Elementwise> {
  using OpTraitRewritePattern::OpTraitRewritePattern;
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();
    if (!llvm::isa<ShapedType>(op->getResults()[0].getType()))
      return failure();
    if (!OpTrait::hasElementwiseMappableTraits(op))
      return failure();
    if (op->getNumOperands() == 0 ||
        op->getResults()[0].getType() != op->getOperand(0).getType()) {
      return failure();
    }
    // Avoid operations that only accept vector types, since broadcast
    // source might be scalar types.
    if (isa<vector::FMAOp>(op)) {
      return failure();
    }

    // Get the type of the lhs operand
    auto *lhsBcastOrSplat = op->getOperand(0).getDefiningOp();
    if (!lhsBcastOrSplat ||
        !isa<vector::BroadcastOp, vector::SplatOp>(*lhsBcastOrSplat))
      return failure();
    auto lhsBcastOrSplatType = lhsBcastOrSplat->getOperand(0).getType();

    // Make sure that all operands are broadcast from identical types:
    //  * scalar (`vector.broadcast` + `vector.splat`), or
    //  * vector (`vector.broadcast`).
    // Otherwise the re-ordering wouldn't be safe.
    if (!llvm::all_of(op->getOperands(), [&lhsBcastOrSplatType](Value val) {
          auto bcast = val.getDefiningOp<vector::BroadcastOp>();
          if (bcast)
            return (bcast.getOperand().getType() == lhsBcastOrSplatType);
          auto splat = val.getDefiningOp<vector::SplatOp>();
          if (splat)
            return (splat.getOperand().getType() == lhsBcastOrSplatType);
          return false;
        })) {
      return failure();
    }

    // Collect the source values before broadcasting
    SmallVector<Value> srcValues;
    srcValues.reserve(op->getNumOperands());
    for (Value operand : op->getOperands()) {
      srcValues.push_back(operand.getDefiningOp()->getOperand(0));
    }

    // Create the "elementwise" Op
    Operation *elementwiseOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), srcValues,
                        lhsBcastOrSplatType, op->getAttrs());

    // Replace the original Op with the elementwise Op
    auto vectorType = op->getResultTypes()[0];
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        op, vectorType, elementwiseOp->getResults());

    return success();
  }
};

// Helper that returns a vector comparison that constructs a mask:
//     mask = [0,1,..,n-1] + [o,o,..,o] < [b,b,..,b]
//
// If `dim == 0` then the result will be a 0-D vector.
//
// NOTE: The LLVM::GetActiveLaneMaskOp intrinsic would provide an alternative,
//       much more compact, IR for this operation, but LLVM eventually
//       generates more elaborate instructions for this intrinsic since it
//       is very conservative on the boundary conditions.
static Value buildVectorComparison(PatternRewriter &rewriter, Operation *op,
                                   bool force32BitVectorIndices, int64_t dim,
                                   Value b, Value *off = nullptr) {
  auto loc = op->getLoc();
  // If we can assume all indices fit in 32-bit, we perform the vector
  // comparison in 32-bit to get a higher degree of SIMD parallelism.
  // Otherwise we perform the vector comparison using 64-bit indices.
  Type idxType =
      force32BitVectorIndices ? rewriter.getI32Type() : rewriter.getI64Type();
  DenseIntElementsAttr indicesAttr;
  if (dim == 0 && force32BitVectorIndices) {
    indicesAttr = DenseIntElementsAttr::get(
        VectorType::get(ArrayRef<int64_t>{}, idxType), ArrayRef<int32_t>{0});
  } else if (dim == 0) {
    indicesAttr = DenseIntElementsAttr::get(
        VectorType::get(ArrayRef<int64_t>{}, idxType), ArrayRef<int64_t>{0});
  } else if (force32BitVectorIndices) {
    indicesAttr = rewriter.getI32VectorAttr(
        llvm::to_vector<4>(llvm::seq<int32_t>(0, dim)));
  } else {
    indicesAttr = rewriter.getI64VectorAttr(
        llvm::to_vector<4>(llvm::seq<int64_t>(0, dim)));
  }
  Value indices = rewriter.create<arith::ConstantOp>(loc, indicesAttr);
  // Add in an offset if requested.
  if (off) {
    Value o = getValueOrCreateCastToIndexLike(rewriter, loc, idxType, *off);
    Value ov = rewriter.create<vector::SplatOp>(loc, indices.getType(), o);
    indices = rewriter.create<arith::AddIOp>(loc, ov, indices);
  }
  // Construct the vector comparison.
  Value bound = getValueOrCreateCastToIndexLike(rewriter, loc, idxType, b);
  Value bounds =
      rewriter.create<vector::SplatOp>(loc, indices.getType(), bound);
  return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, indices,
                                        bounds);
}

template <typename ConcreteOp>
struct MaterializeTransferMask : public OpRewritePattern<ConcreteOp> {
public:
  explicit MaterializeTransferMask(MLIRContext *context, bool enableIndexOpt,
                                   PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<ConcreteOp>(context, benefit),
        force32BitVectorIndices(enableIndexOpt) {}

  LogicalResult matchAndRewrite(ConcreteOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (!xferOp.hasOutOfBoundsDim())
      return failure();

    if (xferOp.getVectorType().getRank() > 1 || xferOp.getIndices().empty())
      return failure();

    Location loc = xferOp->getLoc();
    VectorType vtp = xferOp.getVectorType();

    // Create the in-bounds mask with all elements between [0 .. dim - offset)
    // set and [dim - offset .. vector_length) unset.
    //
    // TODO: when the leaf transfer rank is k > 1, we need the last `k`
    //       dimensions here.
    unsigned lastIndex = llvm::size(xferOp.getIndices()) - 1;
    Value off = xferOp.getIndices()[lastIndex];
    Value dim =
        vector::createOrFoldDimOp(rewriter, loc, xferOp.getSource(), lastIndex);
    Value b = rewriter.create<arith::SubIOp>(loc, dim.getType(), dim, off);
    Value mask = rewriter.create<vector::CreateMaskOp>(
        loc,
        VectorType::get(vtp.getShape(), rewriter.getI1Type(),
                        vtp.getScalableDims()),
        b);
    if (xferOp.getMask()) {
      // Intersect the in-bounds with the mask specified as an op parameter.
      mask = rewriter.create<arith::AndIOp>(loc, mask, xferOp.getMask());
    }

    rewriter.modifyOpInPlace(xferOp, [&]() {
      xferOp.getMaskMutable().assign(mask);
      xferOp.setInBoundsAttr(rewriter.getBoolArrayAttr({true}));
    });

    return success();
  }

private:
  const bool force32BitVectorIndices;
};

/// Conversion pattern for a `vector.create_mask` (0-D and 1-D only).
class VectorCreateMaskOpConversion
    : public OpRewritePattern<vector::CreateMaskOp> {
public:
  explicit VectorCreateMaskOpConversion(MLIRContext *context,
                                        bool enableIndexOpt,
                                        PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<vector::CreateMaskOp>(context, benefit),
        force32BitVectorIndices(enableIndexOpt) {}

  LogicalResult matchAndRewrite(vector::CreateMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = op.getType();
    if (cast<VectorType>(dstType).isScalable())
      return failure();
    int64_t rank = dstType.getRank();
    if (rank > 1)
      return failure();
    rewriter.replaceOp(
        op, buildVectorComparison(rewriter, op, force32BitVectorIndices,
                                  rank == 0 ? 0 : dstType.getDimSize(0),
                                  op.getOperand(0)));
    return success();
  }

private:
  const bool force32BitVectorIndices;
};

/// Returns true if all the `i1` elements of `constantOp` are set to `value`.
static bool allI1ConstantValuesSetTo(arith::ConstantOp constantOp, bool value) {
  auto denseAttr = dyn_cast<DenseIntElementsAttr>(constantOp.getValue());
  // TODO: Support non-dense constant.
  if (!denseAttr)
    return false;

  assert(denseAttr.getElementType().isInteger(1) && "Unexpected type");
  return denseAttr.isSplat() && denseAttr.getSplatValue<bool>() == value;
}

/// Folds a select operation between an all-true and all-false vector. For now,
/// only single element vectors (i.e., vector<1xi1>) are supported. That is:
///
///   %true = arith.constant dense<true> : vector<1xi1>
///   %false = arith.constant dense<false> : vector<1xi1>
///   %result = arith.select %cond, %true, %false : i1, vector<1xi1>
///   =>
///   %result = vector.broadcast %cond : i1 to vector<1xi1>
///
/// InstCombine seems to handle vectors with multiple elements but not the
/// single element ones.
struct FoldI1Select : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = dyn_cast<VectorType>(selectOp.getType());
    if (!vecType || !vecType.getElementType().isInteger(1))
      return failure();

    // Only scalar conditions can be folded.
    Value cond = selectOp.getCondition();
    if (isa<VectorType>(cond.getType()))
      return failure();

    // TODO: Support n-D and scalable vectors.
    if (vecType.getRank() != 1 || vecType.isScalable())
      return failure();

    // TODO: Support vectors with multiple elements.
    if (vecType.getShape()[0] != 1)
      return failure();

    auto trueConst = selectOp.getTrueValue().getDefiningOp<arith::ConstantOp>();
    if (!trueConst || !allI1ConstantValuesSetTo(trueConst, true))
      return failure();

    auto falseConst =
        selectOp.getFalseValue().getDefiningOp<arith::ConstantOp>();
    if (!falseConst || !allI1ConstantValuesSetTo(falseConst, false))
      return failure();

    // Replace select with its condition broadcasted to single element vector.
    auto elemType = rewriter.getIntegerType(vecType.getNumElements());
    auto bcastType = VectorType::get(/*shape=*/{1}, elemType);
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(selectOp, bcastType, cond);
    return success();
  }
};

/// Returns the number of dims can be folded away from transfer ops. It returns
/// a failure if it can not determine the number of dims to be folded.
/// Example 1: it returns "2" if `srcType` is memref<512x16x1x1xf32> and
/// `vectorType` is vector<16x16x1x1xf32>. Because there two inner most dims
/// can be dropped by memref.subview ops.
/// Example 2: it returns "1" if `srcType` is the same memref type with
/// [8192, 16, 8, 1] strides.
static FailureOr<size_t>
getTransferFoldableInnerUnitDims(MemRefType srcType, VectorType vectorType) {
  SmallVector<int64_t> srcStrides;
  int64_t srcOffset;
  if (failed(getStridesAndOffset(srcType, srcStrides, srcOffset)))
    return failure();

  // According to vector.transfer_read/write semantics, the vector can be a
  // slice. Thus, we have to offset the check index with `rankDiff` in
  // `srcStrides` and source dim sizes.
  size_t result = 0;
  int rankDiff = srcType.getRank() - vectorType.getRank();
  for (int64_t i = 0, e = vectorType.getRank(); i < e; ++i) {
    // Check that the inner dim size is 1 for both memref type and vector slice.
    // It can be folded only if they are 1 and the stride is 1.
    int dim = vectorType.getRank() - i - 1;
    if (srcStrides[dim + rankDiff] != 1 ||
        srcType.getDimSize(dim + rankDiff) != 1 ||
        vectorType.getDimSize(dim) != 1)
      break;
    result++;
  }
  return result;
}

/// Returns a MemRef type that drops inner `dimsToDrop` dimensions from
/// `srcType`. E.g., if `srcType` is memref<512x16x1x1xf32> and `dimsToDrop` is
/// two, it returns memref<512x16x16> type.
static MemRefType getMemRefTypeWithDroppingInnerDims(OpBuilder &builder,
                                                     MemRefType srcType,
                                                     size_t dimsToDrop) {
  MemRefLayoutAttrInterface layout = srcType.getLayout();
  if (isa<AffineMapAttr>(layout) && layout.isIdentity()) {
    return MemRefType::get(srcType.getShape().drop_back(dimsToDrop),
                           srcType.getElementType(), nullptr,
                           srcType.getMemorySpace());
  }
  MemRefLayoutAttrInterface updatedLayout;
  if (auto strided = dyn_cast<StridedLayoutAttr>(layout)) {
    auto strides = llvm::to_vector(strided.getStrides().drop_back(dimsToDrop));
    updatedLayout = StridedLayoutAttr::get(strided.getContext(),
                                           strided.getOffset(), strides);
    return MemRefType::get(srcType.getShape().drop_back(dimsToDrop),
                           srcType.getElementType(), updatedLayout,
                           srcType.getMemorySpace());
  }

  // Non-strided layout case.
  AffineMap map = srcType.getLayout().getAffineMap();
  int numSymbols = map.getNumSymbols();
  for (size_t i = 0; i < dimsToDrop; ++i) {
    int dim = srcType.getRank() - i - 1;
    map = map.replace(builder.getAffineDimExpr(dim),
                      builder.getAffineConstantExpr(0), map.getNumDims() - 1,
                      numSymbols);
  }
  return MemRefType::get(srcType.getShape().drop_back(dimsToDrop),
                         srcType.getElementType(), updatedLayout,
                         srcType.getMemorySpace());
}

/// Drop inner most contiguous unit dimensions from transfer_read operand.
class DropInnerMostUnitDimsTransferRead
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (readOp.getTransferRank() == 0)
      return failure();

    // TODO: support mask.
    if (readOp.getMask())
      return failure();

    auto srcType = dyn_cast<MemRefType>(readOp.getSource().getType());
    if (!srcType)
      return failure();

    if (!readOp.getPermutationMap().isMinorIdentity())
      return failure();

    auto targetType = readOp.getVectorType();
    if (targetType.getRank() <= 1)
      return failure();

    FailureOr<size_t> maybeDimsToDrop =
        getTransferFoldableInnerUnitDims(srcType, targetType);
    if (failed(maybeDimsToDrop))
      return failure();

    size_t dimsToDrop = maybeDimsToDrop.value();
    if (dimsToDrop == 0)
      return failure();

    auto resultTargetVecType =
        VectorType::get(targetType.getShape().drop_back(dimsToDrop),
                        targetType.getElementType());

    auto loc = readOp.getLoc();
    SmallVector<OpFoldResult> sizes =
        memref::getMixedSizes(rewriter, loc, readOp.getSource());
    SmallVector<OpFoldResult> offsets(srcType.getRank(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(srcType.getRank(),
                                      rewriter.getIndexAttr(1));
    MemRefType resultMemrefType =
        getMemRefTypeWithDroppingInnerDims(rewriter, srcType, dimsToDrop);
    ArrayAttr inBoundsAttr =
        readOp.getInBounds()
            ? rewriter.getArrayAttr(
                  readOp.getInBoundsAttr().getValue().drop_back(dimsToDrop))
            : ArrayAttr();
    Value rankedReducedView = rewriter.create<memref::SubViewOp>(
        loc, resultMemrefType, readOp.getSource(), offsets, sizes, strides);
    auto permMap = getTransferMinorIdentityMap(
        cast<ShapedType>(rankedReducedView.getType()), resultTargetVecType);
    Value result = rewriter.create<vector::TransferReadOp>(
        loc, resultTargetVecType, rankedReducedView,
        readOp.getIndices().drop_back(dimsToDrop), AffineMapAttr::get(permMap),
        readOp.getPadding(),
        // TODO: support mask.
        /*mask=*/Value(), inBoundsAttr);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(readOp, targetType,
                                                     result);
    return success();
  }
};

/// Drop inner most contiguous unit dimensions from transfer_write operand.
/// E.g.,
///    vector.transfer_write %arg1, %arg0[%c0, %arg2, %c0, %c0, %c0]
///      {in_bounds = [true, true, true, true, true]}
///      : vector<1x16x16x1x1xf32>, memref<1x512x16x1x1xf32>
///
/// will be replaced with
///
///    %subview = memref.subview %arg0
///      [0, 0, 0, 0, 0] [1, 512, 16, 1, 1] [1, 1, 1, 1, 1]
///      : memref<1x512x16x1x1xf32> to memref<1x512x16xf32>
///    %0 = vector.shape_cast %arg1 : vector<1x16x16x1x1xf32>
///      to vector<1x16x16xf32>
///    vector.transfer_write %0, %subview[%c0, %arg2, %c0]
///      {in_bounds = [true, true, true]}
///      : vector<1x16x16xf32>, memref<1x512x16xf32>
class DropInnerMostUnitDimsTransferWrite
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (writeOp.getTransferRank() == 0)
      return failure();

    // TODO: support mask.
    if (writeOp.getMask())
      return failure();

    auto srcType = dyn_cast<MemRefType>(writeOp.getSource().getType());
    if (!srcType)
      return failure();

    if (!writeOp.getPermutationMap().isMinorIdentity())
      return failure();

    auto targetType = writeOp.getVectorType();
    if (targetType.getRank() <= 1)
      return failure();

    FailureOr<size_t> maybeDimsToDrop =
        getTransferFoldableInnerUnitDims(srcType, targetType);
    if (failed(maybeDimsToDrop))
      return failure();

    size_t dimsToDrop = maybeDimsToDrop.value();
    if (dimsToDrop == 0)
      return failure();

    auto resultTargetVecType =
        VectorType::get(targetType.getShape().drop_back(dimsToDrop),
                        targetType.getElementType());

    Location loc = writeOp.getLoc();
    SmallVector<OpFoldResult> sizes =
        memref::getMixedSizes(rewriter, loc, writeOp.getSource());
    SmallVector<OpFoldResult> offsets(srcType.getRank(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(srcType.getRank(),
                                      rewriter.getIndexAttr(1));
    MemRefType resultMemrefType =
        getMemRefTypeWithDroppingInnerDims(rewriter, srcType, dimsToDrop);
    ArrayAttr inBoundsAttr =
        writeOp.getInBounds()
            ? rewriter.getArrayAttr(
                  writeOp.getInBoundsAttr().getValue().drop_back(dimsToDrop))
            : ArrayAttr();

    Value rankedReducedView = rewriter.create<memref::SubViewOp>(
        loc, resultMemrefType, writeOp.getSource(), offsets, sizes, strides);
    auto permMap = getTransferMinorIdentityMap(
        cast<ShapedType>(rankedReducedView.getType()), resultTargetVecType);

    auto shapeCast = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, resultTargetVecType, writeOp.getVector());
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        writeOp, shapeCast, rankedReducedView,
        writeOp.getIndices().drop_back(dimsToDrop), AffineMapAttr::get(permMap),
        // TODO: support mask.
        /*mask=*/Value(), inBoundsAttr);
    return success();
  }
};

/// Canonicalization of a `vector.contraction %a, %b, %c` with row-major matmul
/// semantics to a contraction suitable for MMT (matrix matrix multiplication
/// with the RHS transposed) lowering.
struct CanonicalizeContractMatmulToMMT final
    : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  using FilterConstraintType =
      std::function<LogicalResult(vector::ContractionOp op)>;

  CanonicalizeContractMatmulToMMT(MLIRContext *context, PatternBenefit benefit,
                                  FilterConstraintType constraint)
      : OpRewritePattern<vector::ContractionOp>(context, benefit),
        filter(std::move(constraint)) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter(op)))
      return failure();

    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value res = op.getAcc();

    // Set up the parallel/reduction structure in right form.
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [&](MapList m) {
      return AffineMap::inferFromExprList(m, op.getContext());
    };
    AffineExpr m;
    AffineExpr n;
    AffineExpr k;
    bindDims(rewriter.getContext(), m, n, k);
    static constexpr std::array<int64_t, 2> perm = {1, 0};
    auto iteratorTypes = op.getIteratorTypes().getValue();
    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    if (iteratorTypes.size() != 3 ||
        !vector::isParallelIterator(iteratorTypes[0]) ||
        !vector::isParallelIterator(iteratorTypes[1]) ||
        !vector::isReductionIterator(iteratorTypes[2]))
      return rewriter.notifyMatchFailure(op, "contraction is not a gemm");

    // The canonical form is "TNT" = A row-major, B col-major, C row-major.
    const auto canonicalForm = infer({{m, k}, {n, k}, {m, n}});
    if (maps == canonicalForm)
      return rewriter.notifyMatchFailure(op, "already in the canonical form");

    // Create a vector transpose making sure to emit zero/sign-extend at the
    // end.
    auto createTranspose = [&rewriter, loc](Value mat) -> Value {
      if (auto sext = mat.getDefiningOp<arith::ExtSIOp>()) {
        Value trans =
            rewriter.create<vector::TransposeOp>(loc, sext.getIn(), perm);
        VectorType newType =
            cast<VectorType>(trans.getType())
                .clone(cast<VectorType>(mat.getType()).getElementType());
        return rewriter.create<arith::ExtSIOp>(loc, newType, trans);
      }
      if (auto zext = mat.getDefiningOp<arith::ExtUIOp>()) {
        Value trans =
            rewriter.create<vector::TransposeOp>(loc, zext.getIn(), perm);
        VectorType newType =
            VectorType::get(cast<VectorType>(trans.getType()).getShape(),
                            cast<VectorType>(mat.getType()).getElementType());
        return rewriter.create<arith::ExtUIOp>(loc, newType, trans);
      }
      return rewriter.create<vector::TransposeOp>(loc, mat, perm);
    };

    if (maps == infer({{m, k}, {k, n}, {m, n}})) {
      rhs = createTranspose(rhs);
    } else if (maps == infer({{k, m}, {n, k}, {m, n}})) {
      lhs = createTranspose(lhs);
    } else if (maps == infer({{k, m}, {k, n}, {m, n}})) {
      rhs = createTranspose(rhs);
      lhs = createTranspose(lhs);
    } else if (maps == infer({{k, m}, {k, n}, {n, m}})) {
      std::swap(rhs, lhs);
      rhs = createTranspose(rhs);
      lhs = createTranspose(lhs);
    } else if (maps == infer({{k, m}, {n, k}, {n, m}})) {
      std::swap(rhs, lhs);
      rhs = createTranspose(rhs);
    } else if (maps == infer({{m, k}, {k, n}, {n, m}})) {
      std::swap(lhs, rhs);
      lhs = createTranspose(lhs);
    } else if (maps == infer({{m, k}, {n, k}, {n, m}})) {
      std::swap(lhs, rhs);
    } else {
      return rewriter.notifyMatchFailure(op, "unhandled contraction form");
    }
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, lhs, rhs, res, rewriter.getAffineMapArrayAttr(canonicalForm),
        op.getIteratorTypes());
    return success();
  };

private:
  FilterConstraintType filter;
};

/// Pattern to fold arithmetic extensions on floating point data types into
/// vector contraction operations. linalg.matmul introduces arithmetic
/// extensions on its operands. Please mlir snippets below for more details.
/// ```mlir
///   "linalg.matmul"(%lhs, %rhs, %acc) ({
///      ^bb0(%arg1: f16, %arg2: f16, %arg3: f32):
///        %lhs_f32 = "arith.extf"(%arg1) : (f16) -> f32
///        %rhs_f32 = "arith.extf"(%arg2) : (f16) -> f32
///        %mul = "arith.mulf"(%lhs_f32, %rhs_f32) : (f32, f32) -> f32
///        %acc = "arith.addf"(%arg3, %mul) : (f32, f32) -> f32
///        "linalg.yield"(%acc) : (f32) -> ()
///     })
/// ```
/// This restricts the native usage of mixed precision NVIDIA Ampere Tensor
/// Cores, i.e, `mma.sync.*.f32.f16.f16.f32` and `mma.sync.*.f32.bf16.bf16.f32`.
/// This pattern folds the arithmetic extensions into the vector contraction and
/// enables the usage of native mixed precision Tensor Core instructions.
struct FoldArithExtIntoContractionOp
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    auto lhsDefOp = contractOp.getLhs().getDefiningOp<arith::ExtFOp>();
    auto rhsDefOp = contractOp.getRhs().getDefiningOp<arith::ExtFOp>();

    if (!lhsDefOp || !rhsDefOp) {
      return rewriter.notifyMatchFailure(contractOp,
                                         "no defining op on contract operands");
    }

    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, lhsDefOp->getOperand(0), rhsDefOp->getOperand(0),
        contractOp.getAcc(), contractOp.getIndexingMapsAttr(),
        contractOp.getIteratorTypesAttr());

    return success();
  }
};

/// Pattern to fold chained reduction to a series of vector additions and a
/// final reduction. This form should require fewer subgroup operations.
///
/// ```mlir
/// %a = vector.reduction <add> %x, %acc
/// %b = vector.reduction <add> %y, %a
///  ==>
/// %a = arith.addf %x, %y
/// %b = vector.reduction <add> %a, %acc
/// ```
struct ChainedReduction final : OpRewritePattern<vector::ReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ReductionOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Handle other combining kinds.
    if (op.getKind() != vector::CombiningKind::ADD)
      return failure();

    // Accumulator is optional.
    Value acc = op.getAcc();
    if (!acc)
      return failure();

    if (!acc.getType().isIntOrFloat())
      return failure();

    auto parentReduction = acc.getDefiningOp<vector::ReductionOp>();
    if (!parentReduction)
      return failure();

    Location loc = op.getLoc();
    Value vAdd;
    if (isa<IntegerType>(acc.getType())) {
      vAdd = rewriter.createOrFold<arith::AddIOp>(
          loc, parentReduction.getVector(), op.getVector());
    } else {
      vAdd = rewriter.create<arith::AddFOp>(loc, parentReduction.getVector(),
                                            op.getVector());
    }
    rewriter.replaceOpWithNewOp<vector::ReductionOp>(op, op.getKind(), vAdd,
                                                     parentReduction.getAcc());
    return success();
  }
};

/// For vectors with either leading or trailing unit dim, replaces:
///   elementwise(a, b)
/// with:
///   sc_a = shape_cast(a)
///   sc_b = shape_cast(b)
///   res = elementwise(sc_a, sc_b)
///   return shape_cast(res)
/// The newly inserted shape_cast Ops fold (before elementwise Op) and then
/// restore (after elementwise Op) the unit dim. Vectors `a` and `b` are
/// required to be rank > 1.
///
/// Ex:
/// ```
///  %mul = arith.mulf %B_row, %A_row : vector<1x[4]xf32>
///  %cast = vector.shape_cast %mul : vector<1x[4]xf32> to vector<[4]xf32>
/// ```
///
/// gets converted to:
///
/// ```
///  %B_row_sc = vector.shape_cast %B_row : vector<1x[4]xf32> to vector<[4]xf32>
///  %A_row_sc = vector.shape_cast %A_row : vector<1x[4]xf32> to vector<[4]xf32>
///  %mul = arith.mulf %B_row_sc, %A_row_sc : vector<[4]xf32>
///  %cast_new = vector.shape_cast %mul : vector<[4]xf32> to vector<1x[4]xf32>
///  %cast = vector.shape_cast %cast_new : vector<1x[4]xf32> to vector<[4]xf32>
/// ```
///
/// Patterns for folding shape_casts should instantly eliminate `%cast_new` and
/// `%cast`.
struct DropUnitDimFromElementwiseOps final
    : public OpTraitRewritePattern<OpTrait::Elementwise> {
  using OpTraitRewritePattern::OpTraitRewritePattern;
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumRegions() != 0)
      return failure();

    auto resultVectorType = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!resultVectorType)
      return failure();

    // Check the pre-conditions. For `Elementwise` Ops all operands are
    // guaranteed to have identical shapes and it suffices to only check the
    // first one.
    auto sourceVectorType = cast<VectorType>(op->getOperands()[0].getType());
    if (sourceVectorType.getRank() < 2)
      return failure();

    bool hasTrailingDimUnitFixed =
        ((sourceVectorType.getShape().back() == 1) &&
         (!sourceVectorType.getScalableDims().back()));
    bool hasLeadingDimUnitFixed =
        ((sourceVectorType.getShape().front() == 1) &&
         (!sourceVectorType.getScalableDims().front()));
    if (!hasLeadingDimUnitFixed && !hasTrailingDimUnitFixed)
      return failure();

    // Drop leading/trailing unit dim by applying vector.shape_cast to all
    // operands
    int64_t dim = hasLeadingDimUnitFixed ? 0 : sourceVectorType.getRank() - 1;

    SmallVector<Value> newOperands;
    auto loc = op->getLoc();
    for (auto operand : op->getOperands()) {
      auto opVectorType = cast<VectorType>(operand.getType());
      VectorType newVType = VectorType::Builder(opVectorType).dropDim(dim);
      auto opSC = rewriter.create<vector::ShapeCastOp>(loc, newVType, operand);
      newOperands.push_back(opSC);
    }

    VectorType newResultVectorType =
        VectorType::Builder(resultVectorType).dropDim(dim);
    // Create an updated elementwise Op without leading/trailing unit dim
    Operation *elementwiseOp =
        rewriter.create(loc, op->getName().getIdentifier(), newOperands,
                        newResultVectorType, op->getAttrs());

    // Restore the leading/trailing unit dim by applying vector.shape_cast
    // to the result
    rewriter.replaceOpWithNewOp<ShapeCastOp>(op, resultVectorType,
                                             elementwiseOp->getResult(0));

    return success();
  }
};

/// Pattern to eliminate redundant zero-constants added to reduction operands.
/// It's enough for there to be one initial zero value, so we can eliminate the
/// extra ones that feed into `vector.reduction <add>`. These get created by the
/// `ChainedReduction` pattern.
///
/// ```mlir
/// %a = arith.addf %x, %zero
/// %b = arith.addf %a, %y
/// %c = vector.reduction <add> %b, %acc
///  ==>
/// %b = arith.addf %a, %y
/// %c = vector.reduction <add> %b, %acc
/// ```
struct ReduceRedundantZero final : OpRewritePattern<vector::ReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ReductionOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Handle other reduction kinds and their identity values.
    if (op.getKind() != vector::CombiningKind::ADD)
      return failure();

    Type elemType = op.getSourceVectorType().getElementType();
    // The integer case should be handled by `arith.addi` folders, only check
    // for floats here.
    if (!isa<FloatType>(elemType))
      return failure();

    auto vAdd = op.getVector().getDefiningOp<arith::AddFOp>();
    if (!vAdd)
      return failure();
    auto addLhs = vAdd.getLhs().getDefiningOp<arith::AddFOp>();
    if (!addLhs)
      return failure();

    if (!matchPattern(addLhs.getRhs(), m_AnyZeroFloat()))
      return failure();

    auto newAdd = rewriter.create<arith::AddFOp>(vAdd.getLoc(), addLhs.getLhs(),
                                                 vAdd.getRhs());
    rewriter.replaceOpWithNewOp<vector::ReductionOp>(op, op.getKind(), newAdd,
                                                     op.getAcc());
    return success();
  }
};

/// Example:
/// ```
/// %a = vector.reduction <add> %x : vector<2xf32> into f32
/// ```
/// is transformed into:
/// ```
/// %y = vector.extract %x[0] : f32 from vector<2xf32>
/// %z = vector.extract %x[1] : f32 from vector<2xf32>
/// %a = arith.addf %y, %z : f32
/// ```
struct BreakDownVectorReduction final : OpRewritePattern<vector::ReductionOp> {
  BreakDownVectorReduction(MLIRContext *context,
                           unsigned maxNumElementsToExtract,
                           PatternBenefit benefit)
      : OpRewritePattern(context, benefit),
        maxNumElementsToExtract(maxNumElementsToExtract) {}

  LogicalResult matchAndRewrite(vector::ReductionOp op,
                                PatternRewriter &rewriter) const override {
    VectorType type = op.getSourceVectorType();
    if (type.isScalable() || op.isMasked())
      return failure();
    assert(type.getRank() == 1 && "Expected a 1-d vector");

    int64_t numElems = type.getNumElements();
    if (numElems > maxNumElementsToExtract) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("has too many vector elements ({0}) to break down "
                            "(max allowed: {1})",
                            numElems, maxNumElementsToExtract));
    }

    Location loc = op.getLoc();
    SmallVector<Value> extracted(numElems, nullptr);
    for (auto [idx, extractedElem] : llvm::enumerate(extracted))
      extractedElem = rewriter.create<vector::ExtractOp>(
          loc, op.getVector(), static_cast<int64_t>(idx));

    Value res = extracted.front();
    for (auto extractedElem : llvm::drop_begin(extracted))
      res = vector::makeArithReduction(rewriter, loc, op.getKind(), res,
                                       extractedElem, op.getFastmathAttr());
    if (Value acc = op.getAcc())
      res = vector::makeArithReduction(rewriter, loc, op.getKind(), res, acc,
                                       op.getFastmathAttr());

    rewriter.replaceOp(op, res);
    return success();
  }

private:
  unsigned maxNumElementsToExtract = 0;
};

} // namespace

void mlir::vector::populateFoldArithExtensionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldArithExtIntoContractionOp>(patterns.getContext());
}

void mlir::vector::populateVectorMaskMaterializationPatterns(
    RewritePatternSet &patterns, bool force32BitVectorIndices,
    PatternBenefit benefit) {
  patterns.add<VectorCreateMaskOpConversion,
               MaterializeTransferMask<vector::TransferReadOp>,
               MaterializeTransferMask<vector::TransferWriteOp>>(
      patterns.getContext(), force32BitVectorIndices, benefit);
  patterns.add<FoldI1Select>(patterns.getContext(), benefit);
}

void mlir::vector::populateShapeCastFoldingPatterns(RewritePatternSet &patterns,
                                                    PatternBenefit benefit) {
  patterns.add<ShapeCastOpFolder>(patterns.getContext(), benefit);
}

void mlir::vector::populateDropUnitDimWithShapeCastPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<DropUnitDimFromElementwiseOps, ShapeCastOpFolder>(
      patterns.getContext(), benefit);
}

void mlir::vector::populateBubbleVectorBitCastOpPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<BubbleDownVectorBitCastForExtract,
               BubbleDownBitCastForStridedSliceExtract,
               BubbleUpBitCastForStridedSliceInsert>(patterns.getContext(),
                                                     benefit);
}

void mlir::vector::populateBreakDownVectorBitCastOpPatterns(
    RewritePatternSet &patterns,
    std::function<bool(vector::BitCastOp)> controlFn, PatternBenefit benefit) {
  patterns.add<BreakDownVectorBitCast>(patterns.getContext(),
                                       std::move(controlFn), benefit);
}

void mlir::vector::populateVectorContractCanonicalizeMatmulToMMT(
    RewritePatternSet &patterns,
    std::function<LogicalResult(vector::ContractionOp)> constraint,
    PatternBenefit benefit) {
  patterns.add<CanonicalizeContractMatmulToMMT>(patterns.getContext(), benefit,
                                                std::move(constraint));
}

void mlir::vector::populateVectorReductionToContractPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<MultiReduceToContract, CombineContractBroadcast,
               CombineContractABTranspose, CombineContractResultTranspose,
               ReorderCastOpsOnBroadcast, ReorderElementwiseOpsOnTranspose>(
      patterns.getContext(), benefit);
}

void mlir::vector::
    populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
        RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<DropInnerMostUnitDimsTransferRead,
               DropInnerMostUnitDimsTransferWrite>(patterns.getContext(),
                                                   benefit);
}

void mlir::vector::populateSinkVectorBroadcastPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ReorderCastOpsOnBroadcast, ReorderElementwiseOpsOnBroadcast>(
      patterns.getContext(), benefit);
}

void mlir::vector::populateChainedVectorReductionFoldingPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ChainedReduction>(patterns.getContext(), benefit);
  patterns.add<ReduceRedundantZero>(patterns.getContext(),
                                    PatternBenefit(benefit.getBenefit() + 1));
}

void mlir::vector::populateBreakDownVectorReductionPatterns(
    RewritePatternSet &patterns, unsigned maxNumElementsToExtract,
    PatternBenefit benefit) {
  patterns.add<BreakDownVectorReduction>(patterns.getContext(),
                                         maxNumElementsToExtract, benefit);
}

//===----------------------------------------------------------------------===//
// TableGen'd enum attribute definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/VectorTransformsEnums.cpp.inc"
