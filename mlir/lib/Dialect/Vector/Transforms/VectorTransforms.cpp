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
        shapeCastOp.getSource().getType().dyn_cast_or_null<VectorType>();
    auto resultVectorType =
        shapeCastOp.getResult().getType().dyn_cast_or_null<VectorType>();
    if (!sourceVectorType || !resultVectorType)
      return failure();

    // Check if shape cast op source operand is also a shape cast op.
    auto sourceShapeCastOp = dyn_cast_or_null<vector::ShapeCastOp>(
        shapeCastOp.getSource().getDefiningOp());
    if (!sourceShapeCastOp)
      return failure();
    auto operandSourceVectorType =
        sourceShapeCastOp.getSource().getType().cast<VectorType>();
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
    auto dstMap = AffineMap::get(/*dimCount=*/reductionMask.size(),
                                 /*symCount=*/0, exprs, reduceOp.getContext());
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
          extractVector<unsigned>(transposeOp.getTransp()),
          contractOp.getContext());
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
    auto accTMap = AffineMap::getPermutationMap(
        extractVector<unsigned>(accTOp.getTransp()), context);

    // Contract performs g(C) -> D. Result transpose performs h(D) -> E.
    // To index into E in contract, we need h(g(C)) -> E.
    auto resTMap = AffineMap::getPermutationMap(
        extractVector<unsigned>(resTOp.getTransp()), context);
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
      auto srcType = broadcast.getSourceType().dyn_cast<VectorType>();
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
    if (auto vecTy = bcastOp.getSourceType().dyn_cast<VectorType>())
      castResTy = VectorType::get(vecTy.getShape(), castResTy);
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
    SmallVector<ArrayAttr> transposeMaps;
    transposeMaps.reserve(op->getNumOperands());
    // Record the initial type before transposition. We'll use its shape later.
    // Any type will do here as we will check all transpose maps are the same.
    VectorType srcType;
    for (Value operand : op->getOperands()) {
      auto transposeOp = operand.getDefiningOp<vector::TransposeOp>();
      if (transposeOp) {
        transposeMaps.push_back(transposeOp.getTransp());
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
    auto order = extractVector<unsigned>(transposeMaps.front());
    SmallVector<int64_t> invOrder(order.size());
    for (int i = 0, e = order.size(); i < e; ++i)
      invOrder[order[i]] = i;

    for (Value operand : op->getOperands()) {
      auto transposeOp = operand.getDefiningOp<vector::TransposeOp>();
      if (transposeOp) {
        srcValues.push_back(transposeOp.getVector());
      } else {
        // This is a constant. Create a reverse transpose op for it.
        auto vectorType = VectorType::get(
            srcType.getShape(),
            operand.getType().cast<VectorType>().getElementType());
        srcValues.push_back(rewriter.create<vector::TransposeOp>(
            operand.getLoc(), vectorType, operand,
            rewriter.getI64ArrayAttr(invOrder)));
      }
    }

    auto vectorType = VectorType::get(
        srcType.getShape(),
        op->getResultTypes()[0].cast<VectorType>().getElementType());
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
//   %1 = vector.extract %0[3] : vector<8xf16>
// Into:
//   %0 = vector.extract %src[1] : vector<4xf32>
//   %1 = vector.bitcast %0: vector<1xf32> to vector<2xf16>
//   %2 = vector.extract %1[1] : vector<2xf16>
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

    auto getFirstIntValue = [](ArrayAttr attr) -> uint64_t {
      return (*attr.getAsValueRange<IntegerAttr>().begin()).getZExtValue();
    };

    uint64_t index = getFirstIntValue(extractOp.getPosition());

    // Get the single scalar (as a vector) in the source value that packs the
    // desired scalar. E.g. extract vector<1xf32> from vector<4xf32>
    VectorType oneScalarType =
        VectorType::get({1}, castSrcType.getElementType());
    Value packedValue = rewriter.create<vector::ExtractOp>(
        extractOp.getLoc(), oneScalarType, castOp.getSource(),
        rewriter.getI64ArrayAttr(index / expandRatio));

    // Cast it to a vector with the desired scalar's type.
    // E.g. f32 -> vector<2xf16>
    VectorType packedType =
        VectorType::get({expandRatio}, castDstType.getElementType());
    Value castedValue = rewriter.create<vector::BitCastOp>(
        extractOp.getLoc(), packedType, packedValue);

    // Finally extract the desired scalar.
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(
        extractOp, extractOp.getType(), castedValue,
        rewriter.getI64ArrayAttr(index % expandRatio));

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
        llvm::to_vector<4>(extractOp.getType().cast<VectorType>().getShape());
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
                        vtp.getNumScalableDims()),
        b);
    if (xferOp.getMask()) {
      // Intersect the in-bounds with the mask specified as an op parameter.
      mask = rewriter.create<arith::AndIOp>(loc, mask, xferOp.getMask());
    }

    rewriter.updateRootInPlace(xferOp, [&]() {
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
    if (dstType.cast<VectorType>().isScalable())
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

// Drop inner most contiguous unit dimensions from transfer_read operand.
class DropInnerMostUnitDims : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (readOp.getTransferRank() == 0)
      return failure();

    // TODO: support mask.
    if (readOp.getMask())
      return failure();

    auto srcType = readOp.getSource().getType().dyn_cast<MemRefType>();
    if (!srcType || !srcType.hasStaticShape())
      return failure();

    if (!readOp.getPermutationMap().isMinorIdentity())
      return failure();

    auto targetType = readOp.getVectorType();
    if (targetType.getRank() <= 1)
      return failure();

    SmallVector<int64_t> srcStrides;
    int64_t srcOffset;
    if (failed(getStridesAndOffset(srcType, srcStrides, srcOffset)))
      return failure();

    size_t dimsToDrop = 0;
    for (size_t i = 1; i < srcStrides.size(); ++i) {
      int dim = srcType.getRank() - i - 1;
      if (srcStrides[dim] == 1) {
        dimsToDrop++;
      } else {
        break;
      }
    }
    if (dimsToDrop == 0)
      return failure();

    auto resultTargetVecType =
        VectorType::get(targetType.getShape().drop_back(dimsToDrop),
                        targetType.getElementType());

    MemRefType resultMemrefType;
    MemRefLayoutAttrInterface layout = srcType.getLayout();
    if (layout.isa<AffineMapAttr>() && layout.isIdentity()) {
      resultMemrefType = MemRefType::get(
          srcType.getShape().drop_back(dimsToDrop), srcType.getElementType(),
          nullptr, srcType.getMemorySpace());
    } else {
      MemRefLayoutAttrInterface updatedLayout;
      if (auto strided = layout.dyn_cast<StridedLayoutAttr>()) {
        auto strides =
            llvm::to_vector(strided.getStrides().drop_back(dimsToDrop));
        updatedLayout = StridedLayoutAttr::get(strided.getContext(),
                                               strided.getOffset(), strides);
      } else {
        AffineMap map = srcType.getLayout().getAffineMap();
        int numSymbols = map.getNumSymbols();
        for (size_t i = 0; i < dimsToDrop; ++i) {
          int dim = srcType.getRank() - i - 1;
          map = map.replace(rewriter.getAffineDimExpr(dim),
                            rewriter.getAffineConstantExpr(0),
                            map.getNumDims() - 1, numSymbols);
        }
      }
      resultMemrefType = MemRefType::get(
          srcType.getShape().drop_back(dimsToDrop), srcType.getElementType(),
          updatedLayout, srcType.getMemorySpace());
    }

    auto loc = readOp.getLoc();
    SmallVector<int64_t> offsets(srcType.getRank(), 0);
    SmallVector<int64_t> strides(srcType.getRank(), 1);

    ArrayAttr inBoundsAttr =
        readOp.getInBounds()
            ? rewriter.getArrayAttr(
                  readOp.getInBoundsAttr().getValue().drop_back(dimsToDrop))
            : ArrayAttr();
    Value rankedReducedView = rewriter.create<memref::SubViewOp>(
        loc, resultMemrefType, readOp.getSource(), offsets, srcType.getShape(),
        strides);
    auto permMap = getTransferMinorIdentityMap(
        rankedReducedView.getType().cast<ShapedType>(), resultTargetVecType);
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
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
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
        return rewriter.create<arith::ExtSIOp>(loc, mat.getType(), trans);
      }
      if (auto zext = mat.getDefiningOp<arith::ExtUIOp>()) {
        Value trans =
            rewriter.create<vector::TransposeOp>(loc, zext.getIn(), perm);
        return rewriter.create<arith::ExtUIOp>(loc, mat.getType(), trans);
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

} // namespace

void mlir::vector::populateVectorMaskMaterializationPatterns(
    RewritePatternSet &patterns, bool force32BitVectorIndices,
    PatternBenefit benefit) {
  patterns.add<VectorCreateMaskOpConversion,
               MaterializeTransferMask<vector::TransferReadOp>,
               MaterializeTransferMask<vector::TransferWriteOp>>(
      patterns.getContext(), force32BitVectorIndices, benefit);
}

void mlir::vector::populateShapeCastFoldingPatterns(RewritePatternSet &patterns,
                                                    PatternBenefit benefit) {
  patterns.add<ShapeCastOpFolder>(patterns.getContext(), benefit);
}

void mlir::vector::populateBubbleVectorBitCastOpPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<BubbleDownVectorBitCastForExtract,
               BubbleDownBitCastForStridedSliceExtract,
               BubbleUpBitCastForStridedSliceInsert>(patterns.getContext(),
                                                     benefit);
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
  patterns.add<DropInnerMostUnitDims>(patterns.getContext(), benefit);
}

//===----------------------------------------------------------------------===//
// TableGen'd enum attribute definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/VectorTransformsEnums.cpp.inc"
