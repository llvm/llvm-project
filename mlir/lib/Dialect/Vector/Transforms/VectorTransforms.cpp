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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

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
/// ```
///
/// For masked vector.contract, the mask requires updating when a dimension is
/// dropped. In such cases, the dropped dimensions must correspond to the mask's
/// leading unit dimensions. Supporting more generic cases (e.g. non-unit dims)
/// is not supported.
FailureOr<Value> combineContractAndBroadcast(vector::ContractionOp contractOp,
                                             MaskingOpInterface maskingOp,
                                             PatternRewriter &rewriter) {
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
      if (dim.value() !=
          broadcast.getResultVectorType().getDimSize(rankDiff + dim.index())) {
        innerDimBroadcast = true;
        break;
      }
      originalDims.push_back(rewriter.getAffineDimExpr(dim.index() + rankDiff));
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
  for (unsigned i = 0, e = unusedDimsBitVector.size(); i < e; ++i) {
    if (!unusedDimsBitVector.test(i))
      iterators.push_back(contractOp.getIteratorTypes().getValue()[i]);
  }

  // Check whether any of the unused dims is non-unit, e.g.:
  //  * vector.broadcast %arg0 : vector<8x4xi32> to vector<2x8x4xi32>
  // This is only required when collapsing a mask. If there is no mask, skip.
  VectorType oldMaskType;
  bool isAnyUnusedDimNonUnit = false;
  if (maskingOp) {
    oldMaskType = cast<VectorType>(maskingOp.getMask().getType());
    for (unsigned i = 0, e = unusedDimsBitVector.size(); i < e; ++i) {
      if (unusedDimsBitVector.test(i) && oldMaskType.getShape()[i] != 1) {
        isAnyUnusedDimNonUnit = true;
        break;
      }
    }
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

  Operation *newOp = vector::ContractionOp::create(
      rewriter, contractOp.getLoc(), lhs, rhs, contractOp.getAcc(),
      rewriter.getAffineMapArrayAttr(maps), rewriter.getArrayAttr(iterators));

  // Handle the mask.
  if (maskingOp) {
    if (isAnyUnusedDimNonUnit)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Cannont drop non-unit mask dim.");
    assert(unusedDimsBitVector.size() ==
               static_cast<size_t>(oldMaskType.getRank()) &&
           "The mask rank is incorrect!");

    // If a dimension has been dropped, update the mask accordingly. Otherwise,
    // keep it as is.
    Value mask = maskingOp.getMask();
    if (unusedDimsBitVector.count() != 0) {
      // At this point, two assumptions are made:
      //  * The unused dimensions are the leading mask dimensions
      //  (vector.contract does not support inner dim broadcasting).
      //  * The unused dimensions are all unit.
      // These conditions are effectively verified in the blocks preceeding this
      // one.
      auto newShape =
          oldMaskType.getShape().drop_front(unusedDimsBitVector.count());
      auto newShapeScalableDims =
          oldMaskType.getScalableDims().drop_front(unusedDimsBitVector.count());
      VectorType maskOpType =
          VectorType::get(newShape, rewriter.getI1Type(), newShapeScalableDims);
      mask = vector::ShapeCastOp::create(rewriter, contractOp.getLoc(),
                                         maskOpType, maskingOp.getMask())
                 .getResult();
    }

    newOp = mlir::vector::maskOperation(rewriter, newOp, mask);
  }
  return newOp->getResult(0);
}

struct CombineContractBroadcastMask
    : public MaskableOpRewritePattern<vector::ContractionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;
  FailureOr<Value>

  matchAndRewriteMaskableOp(vector::ContractionOp contractOp,
                            MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    return combineContractAndBroadcast(contractOp, maskingOp, rewriter);
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
        srcValues.push_back(vector::TransposeOp::create(
            rewriter, operand.getLoc(), vectorType, operand, invOrder));
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

    // Get the first element of the mixed position as integer.
    auto mixedPos = extractOp.getMixedPosition();
    if (!mixedPos.empty() && !isa<Attribute>(mixedPos[0]))
      return failure();
    uint64_t index = cast<IntegerAttr>(cast<Attribute>(mixedPos[0])).getInt();

    // Get the single scalar (as a vector) in the source value that packs the
    // desired scalar. E.g. extract vector<1xf32> from vector<4xf32>
    Location loc = extractOp.getLoc();
    Value packedValue = vector::ExtractOp::create(
        rewriter, loc, castOp.getSource(), index / expandRatio);
    Type packedVecType = VectorType::get(/*shape=*/{1}, packedValue.getType());
    Value zero = arith::ConstantOp::create(rewriter, loc, packedVecType,
                                           rewriter.getZeroAttr(packedVecType));
    packedValue = vector::InsertOp::create(rewriter, loc, packedValue, zero,
                                           /*position=*/0);

    // Cast it to a vector with the desired scalar's type.
    // E.g. f32 -> vector<2xf16>
    VectorType packedType =
        VectorType::get({expandRatio}, castDstType.getElementType());
    Value castedValue =
        vector::BitCastOp::create(rewriter, loc, packedType, packedValue);

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

    auto newExtractOp = vector::ExtractStridedSliceOp::create(
        rewriter, extractOp.getLoc(), newExtractType, castOp.getSource(),
        newOffsets, newSizes, extractOp.getStrides());

    rewriter.replaceOpWithNewOp<vector::BitCastOp>(
        extractOp, extractOp.getType(), newExtractOp);

    return success();
  }
};

// Shuffles vector.bitcast op before vector.insert_strided_slice op.
//
// This transforms IR like:
//   %0 = vector.insert %val, %dst[4] : vector<32xi4> into vector<8x32xi4>
//   %1 = vector.bitcast %0 : vector<8x32xi4> to vector<8x16xi8>
// Into:
//   %0 = vector.bitcast %val : vector<32xi4> to vector<16xi8>
//   %1 = vector.bitcast %dst : vector<8x32xi4> to vector<8x16xi8>
//   %2 = vector.insert %0, %1 [4] : vector<16xi8> into vector<8x16xi8>
//
struct BubbleUpBitCastForInsert : public OpRewritePattern<vector::BitCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    VectorType castSrcType = bitcastOp.getSourceVectorType();
    VectorType castDstType = bitcastOp.getResultVectorType();

    // 0-D and scalable vectors are not supported yet.
    if (castSrcType.getRank() == 0 || castSrcType.isScalable() ||
        castDstType.isScalable())
      return failure();

    int64_t castSrcLastDim = castSrcType.getShape().back();
    int64_t castDstLastDim = castDstType.getShape().back();
    bool isNumElemsShrink = castSrcLastDim >= castDstLastDim;
    int64_t ratio;
    if (isNumElemsShrink) {
      assert(castSrcLastDim % castDstLastDim == 0);
      ratio = castSrcLastDim / castDstLastDim;
    } else {
      assert(castDstLastDim % castSrcLastDim == 0);
      ratio = castDstLastDim / castSrcLastDim;
    }

    auto insertOp = bitcastOp.getSource().getDefiningOp<vector::InsertOp>();
    if (!insertOp)
      return failure();

    // Only vector sources are supported for now.
    auto insertSrcType = dyn_cast<VectorType>(insertOp.getValueToStoreType());
    if (!insertSrcType)
      return failure();

    // Bitcast the source.
    SmallVector<int64_t> srcDims(insertSrcType.getShape());
    srcDims.back() =
        isNumElemsShrink ? srcDims.back() / ratio : srcDims.back() * ratio;
    VectorType newCastSrcType =
        VectorType::get(srcDims, castDstType.getElementType());
    auto newCastSrcOp =
        vector::BitCastOp::create(rewriter, bitcastOp.getLoc(), newCastSrcType,
                                  insertOp.getValueToStore());

    SmallVector<int64_t> dstDims(insertOp.getDestVectorType().getShape());
    dstDims.back() =
        isNumElemsShrink ? dstDims.back() / ratio : dstDims.back() * ratio;
    VectorType newCastDstType =
        VectorType::get(dstDims, castDstType.getElementType());

    // Bitcast the destination.
    auto newCastDstOp = vector::BitCastOp::create(
        rewriter, bitcastOp.getLoc(), newCastDstType, insertOp.getDest());

    // Generate new insert.
    rewriter.replaceOpWithNewOp<vector::InsertOp>(
        bitcastOp, newCastSrcOp, newCastDstOp, insertOp.getMixedPosition());
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

    auto newCastSrcOp =
        vector::BitCastOp::create(rewriter, bitcastOp.getLoc(), newCastSrcType,
                                  insertOp.getValueToStore());

    SmallVector<int64_t> dstDims =
        llvm::to_vector<4>(insertOp.getDestVectorType().getShape());
    dstDims.back() = dstDims.back() / shrinkRatio;
    VectorType newCastDstType =
        VectorType::get(dstDims, castDstType.getElementType());

    auto newCastDstOp = vector::BitCastOp::create(
        rewriter, bitcastOp.getLoc(), newCastDstType, insertOp.getDest());

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

    // This transformation builds on top of
    // vector.{extract|insert}_strided_slice, which do not support
    // extracting/inserting "scallable sub-vectors". Bail out.
    if (castSrcType.isScalable())
      return rewriter.notifyMatchFailure(bitcastOp,
                                         "Scalable vectors are not supported");

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

    Value zero = arith::ConstantOp::create(rewriter, loc, elemType,
                                           rewriter.getZeroAttr(elemType));
    Value res = BroadcastOp::create(rewriter, loc, castDstType, zero);

    SmallVector<int64_t> sliceShape = {castDstLastDim};
    SmallVector<int64_t> strides = {1};
    VectorType newCastDstType =
        VectorType::get(SmallVector<int64_t>{castDstLastDim / shrinkRatio},
                        castDstType.getElementType());

    for (int i = 0, e = shrinkRatio; i < e; ++i) {
      Value extracted = ExtractStridedSliceOp::create(
          rewriter, loc, bitcastOp.getSource(),
          ArrayRef<int64_t>{i * castDstLastDim}, sliceShape, strides);
      Value bitcast =
          BitCastOp::create(rewriter, loc, newCastDstType, extracted);
      res = InsertStridedSliceOp::create(
          rewriter, loc, bitcast, res,
          ArrayRef<int64_t>{i * castDstLastDim / shrinkRatio}, strides);
    }
    rewriter.replaceOp(bitcastOp, res);
    return success();
  }

private:
  std::function<bool(BitCastOp)> controlFn;
};

static bool haveSameShapeAndScaling(Type t, Type u) {
  auto tVec = dyn_cast<VectorType>(t);
  auto uVec = dyn_cast<VectorType>(u);
  if (!tVec) {
    return !uVec;
  }
  if (!uVec) {
    return false;
  }
  return tVec.getShape() == uVec.getShape() &&
         tVec.getScalableDims() == uVec.getScalableDims();
}

/// If `type` is shaped, clone it with `newElementType`. Otherwise,
/// return `newElementType`.
static Type cloneOrReplace(Type type, Type newElementType) {
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.clone(newElementType);
  }
  return newElementType;
}

/// If `value` is the result of a splat or broadcast operation, return the input
/// of the splat/broadcast operation.
static Value getBroadcastLikeSource(Value value) {

  Operation *op = value.getDefiningOp();
  if (!op)
    return {};

  if (auto broadcast = dyn_cast<vector::BroadcastOp>(op))
    return broadcast.getSource();

  if (auto splat = dyn_cast<vector::SplatOp>(op))
    return splat.getInput();

  return {};
}

/// Reorders elementwise(broadcast/splat) to broadcast(elementwise). Ex:
///
/// Example:
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
    auto resultType = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!resultType)
      return failure();
    if (!OpTrait::hasElementwiseMappableTraits(op))
      return rewriter.notifyMatchFailure(
          op, "Op doesn't have ElementwiseMappableTraits");
    if (op->getNumOperands() == 0)
      return failure();
    if (isa<vector::FMAOp>(op)) {
      return rewriter.notifyMatchFailure(
          op,
          "Op only accepts vector types - not supported as broadcast source "
          "might be a scalar");
    }

    Type resultElemType = resultType.getElementType();

    // Get the type of the first non-constant operand
    Value splatSource;
    for (Value operand : op->getOperands()) {
      Operation *definingOp = operand.getDefiningOp();
      if (!definingOp)
        return failure();
      if (definingOp->hasTrait<OpTrait::ConstantLike>())
        continue;
      splatSource = getBroadcastLikeSource(operand);
      break;
    }
    if (!splatSource)
      return failure();
    Type unbroadcastResultType =
        cloneOrReplace(splatSource.getType(), resultElemType);

    // Make sure that all operands are broadcast from identically-shaped types:
    //  * scalar (`vector.broadcast` + `vector.splat`), or
    //  * vector (`vector.broadcast`).
    // Otherwise the re-ordering wouldn't be safe.
    if (!llvm::all_of(op->getOperands(), [splatSource](Value val) {
          if (auto source = getBroadcastLikeSource(val))
            return haveSameShapeAndScaling(source.getType(),
                                           splatSource.getType());
          SplatElementsAttr splatConst;
          return matchPattern(val, m_Constant(&splatConst));
        })) {
      return rewriter.notifyMatchFailure(
          op,
          "not all operands are constants or broadcasts from the same type");
    }

    // Collect the source values before broadcasting
    SmallVector<Value> srcValues;
    srcValues.reserve(op->getNumOperands());
    for (Value operand : op->getOperands()) {
      SplatElementsAttr splatConst;
      if (matchPattern(operand, m_Constant(&splatConst))) {
        Attribute newConst;
        Type elementType = getElementTypeOrSelf(operand.getType());
        Type newType = cloneOrReplace(unbroadcastResultType, elementType);
        if (auto newTypeShaped = dyn_cast<ShapedType>(newType)) {
          newConst = splatConst.resizeSplat(newTypeShaped);
        } else {
          newConst = splatConst.getSplatValue<Attribute>();
        }
        Operation *newConstOp =
            operand.getDefiningOp()->getDialect()->materializeConstant(
                rewriter, newConst, newType, operand.getLoc());
        srcValues.push_back(newConstOp->getResult(0));
      } else {
        srcValues.push_back(operand.getDefiningOp()->getOperand(0));
      }
    }

    // Create the "elementwise" Op
    Operation *elementwiseOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), srcValues,
                        unbroadcastResultType, op->getAttrs());

    // Replace the original Op with the elementwise Op
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        op, resultType, elementwiseOp->getResults());

    return success();
  }
};

/// Pattern to rewrite a ExtractOp(Elementwise) -> Elementwise(ExtractOp).
/// This may result in cleaner code when extracting a single value
/// from multi-element vector and also to help canonicalize 1-element vectors to
/// scalars.
///
/// Example:
/// ```
///  %0 = arith.addf %arg0, %arg1 : vector<4xf32>
///  %1 = vector.extract %0[1] : f32 from vector<4xf32>
/// ```
/// Gets converted to:
/// ```
///  %0 = vector.extract %arg0[1] : f32 from vector<4xf32>
///  %1 = vector.extract %arg1[1] : f32 from vector<4xf32>
///  %2 = arith.addf %0, %1 : f32
/// ```
class ExtractOpFromElementwise final
    : public OpRewritePattern<vector::ExtractOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    Operation *eltwise = op.getVector().getDefiningOp();

    // TODO: vector::FMAOp is not an ElemetwiseMappable even if it claims to be,
    // as it doesn't support scalars.
    if (!eltwise || !OpTrait::hasElementwiseMappableTraits(eltwise) ||
        isa<vector::FMAOp>(eltwise))
      return rewriter.notifyMatchFailure(op, "not an elementwise op");

    if (eltwise->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected single result");

    if (!eltwise->hasOneUse())
      return rewriter.notifyMatchFailure(op, "expected single op use");

    if (!llvm::all_equal(eltwise->getOperandTypes()))
      return rewriter.notifyMatchFailure(op, "operand types are different");

    // Dynamic position can cause dominance issues, so conservatively fail for
    // now.
    if (!op.getDynamicPosition().empty())
      return rewriter.notifyMatchFailure(
          op, "dynamic position not yet implemented");

    Type dstType = op.getType();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(eltwise);

    IRMapping mapping;
    Location loc = eltwise->getLoc();
    SmallVector<OpFoldResult> pos = op.getMixedPosition();
    for (Value arg : eltwise->getOperands()) {
      Value newArg = vector::ExtractOp::create(rewriter, loc, arg, pos);
      mapping.map(arg, newArg);
    }

    Operation *newEltwise = rewriter.clone(*eltwise, mapping);
    newEltwise->getResult(0).setType(dstType);

    rewriter.replaceOp(op, newEltwise);
    rewriter.eraseOp(eltwise);
    return success();
  }
};

/// Check if the element type is suitable for vector.load/store sinking.
/// Element type must be index or byte-aligned integer or floating-point type.
static bool isSupportedMemSinkElementType(Type type) {
  if (isa<IndexType>(type))
    return true;

  return type.isIntOrFloat() && type.getIntOrFloatBitWidth() % 8 == 0;
}

/// Pattern to rewrite `vector.extract(vector.load) -> vector/memref.load.
/// Only index and byte-aligned integer and floating-point element types are
/// supported for now.
///
/// Example:
/// ```
///  vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
///  vector.extract %0[1] : f32 from vector<4xf32>
/// ```
/// Gets converted to:
/// ```
/// %c1 = arith.constant 1 : index
/// %0 = arith.addi %arg1, %c1 overflow<nsw> : index
/// %1 = memref.load %arg0[%0] : memref<?xf32>
/// ```
class ExtractOpFromLoad final : public OpRewritePattern<vector::ExtractOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto loadOp = op.getVector().getDefiningOp<vector::LoadOp>();
    if (!loadOp)
      return rewriter.notifyMatchFailure(op, "expected a load op");

    // Checking for single use so we won't duplicate load ops.
    if (!loadOp->hasOneUse())
      return rewriter.notifyMatchFailure(op, "expected single op use");

    VectorType loadVecType = loadOp.getVectorType();
    if (loadVecType.isScalable())
      return rewriter.notifyMatchFailure(op,
                                         "scalable vectors are not supported");

    MemRefType memType = loadOp.getMemRefType();

    // Non-byte-aligned types are tricky and may require special handling,
    // ignore them for now.
    if (!isSupportedMemSinkElementType(memType.getElementType()))
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    int64_t rankOffset = memType.getRank() - loadVecType.getRank();
    if (rankOffset < 0)
      return rewriter.notifyMatchFailure(op, "unsupported ranks combination");

    auto extractVecType = dyn_cast<VectorType>(op.getResult().getType());
    int64_t finalRank = 0;
    if (extractVecType)
      finalRank = extractVecType.getRank();

    SmallVector<Value> indices = loadOp.getIndices();
    SmallVector<OpFoldResult> extractPos = op.getMixedPosition();

    // There may be memory stores between the load and the extract op, so we
    // need to make sure that the new load op is inserted at the same place as
    // the original load op.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(loadOp);
    Location loc = loadOp.getLoc();
    ArithIndexingBuilder idxBuilderf(rewriter, loc);
    for (auto i : llvm::seq<int64_t>(rankOffset, indices.size() - finalRank)) {
      OpFoldResult pos = extractPos[i - rankOffset];
      if (isZeroInteger(pos))
        continue;

      Value offset = getValueOrCreateConstantIndexOp(rewriter, loc, pos);
      indices[i] = idxBuilderf.add(indices[i], offset);
    }

    Value base = loadOp.getBase();
    if (extractVecType) {
      rewriter.replaceOpWithNewOp<vector::LoadOp>(op, extractVecType, base,
                                                  indices);
    } else {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(op, base, indices);
    }
    // We checked for single use so we can safely erase the load op.
    rewriter.eraseOp(loadOp);
    return success();
  }
};

/// Pattern to rewrite vector.store(vector.splat) -> vector/memref.store.
///
/// Example:
/// ```
/// %0 = vector.splat %arg2 : vector<1xf32>
/// vector.store %0, %arg0[%arg1] : memref<?xf32>, vector<1xf32>
/// ```
/// Gets converted to:
/// ```
/// memref.store %arg2, %arg0[%arg1] : memref<?xf32>
/// ```
class StoreOpFromSplatOrBroadcast final
    : public OpRewritePattern<vector::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp op,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = op.getVectorType();
    if (vecType.isScalable())
      return rewriter.notifyMatchFailure(op,
                                         "scalable vectors are not supported");

    if (isa<VectorType>(op.getMemRefType().getElementType()))
      return rewriter.notifyMatchFailure(
          op, "memrefs of vectors are not supported");

    if (vecType.getNumElements() != 1)
      return rewriter.notifyMatchFailure(
          op, "only 1-element vectors are supported");

    Value toStore = op.getValueToStore();
    Value source = getBroadcastLikeSource(toStore);
    if (!source)
      return rewriter.notifyMatchFailure(
          op, "value to store is not from a broadcast");

    // Checking for single use so we can remove splat.
    Operation *splat = toStore.getDefiningOp();
    if (!splat->hasOneUse())
      return rewriter.notifyMatchFailure(op, "expected single op use");

    Value base = op.getBase();
    ValueRange indices = op.getIndices();

    if (isa<VectorType>(source.getType())) {
      rewriter.replaceOpWithNewOp<vector::StoreOp>(op, source, base, indices);
    } else {
      rewriter.replaceOpWithNewOp<memref::StoreOp>(op, source, base, indices);
    }
    rewriter.eraseOp(splat);
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
  Value indices = arith::ConstantOp::create(rewriter, loc, indicesAttr);
  // Add in an offset if requested.
  if (off) {
    Value o = getValueOrCreateCastToIndexLike(rewriter, loc, idxType, *off);
    Value ov = vector::BroadcastOp::create(rewriter, loc, indices.getType(), o);
    indices = arith::AddIOp::create(rewriter, loc, ov, indices);
  }
  // Construct the vector comparison.
  Value bound = getValueOrCreateCastToIndexLike(rewriter, loc, idxType, b);
  Value bounds =
      vector::BroadcastOp::create(rewriter, loc, indices.getType(), bound);
  return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                               indices, bounds);
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
        vector::createOrFoldDimOp(rewriter, loc, xferOp.getBase(), lastIndex);
    Value b = arith::SubIOp::create(rewriter, loc, dim.getType(), dim, off);
    Value mask = vector::CreateMaskOp::create(
        rewriter, loc,
        VectorType::get(vtp.getShape(), rewriter.getI1Type(),
                        vtp.getScalableDims()),
        b);
    if (xferOp.getMask()) {
      // Intersect the in-bounds with the mask specified as an op parameter.
      mask = arith::AndIOp::create(rewriter, loc, mask, xferOp.getMask());
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
///
/// Ex 1: returns "2" if `srcType` is memref<512x16x1x1xf32> and
/// `vectorType` is vector<16x16x1x1xf32>
/// (there two inner most dims can be dropped by memref.subview ops)
///
/// Ex 2: returns "1" if `srcType` is memref<512x16x1x1xf32> with
/// [8192, 16, 8, 1] strides and `vectorType` is vector<16x16x1x1xf32>
/// (only the inner most unit dim of `srcType` can be dropped)
///
/// Ex 3: return "0" if `srcType` is memref<512x16x1x1xf32> and
/// `vectorType` is vector<16x16x1x[1]xf32>
/// (the most inner dim in `vectorType` is not a unit dim (it's a "scalable
/// unit")
static FailureOr<size_t>
getTransferFoldableInnerUnitDims(MemRefType srcType, VectorType vectorType) {
  SmallVector<int64_t> srcStrides;
  int64_t srcOffset;
  if (failed(srcType.getStridesAndOffset(srcStrides, srcOffset)))
    return failure();

  auto isUnitDim = [](VectorType type, int dim) {
    return type.getDimSize(dim) == 1 && !type.getScalableDims()[dim];
  };

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
        srcType.getDimSize(dim + rankDiff) != 1 || !isUnitDim(vectorType, dim))
      break;
    result++;
  }
  return result;
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

    auto srcType = dyn_cast<MemRefType>(readOp.getBase().getType());
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

    auto inBounds = readOp.getInBoundsValues();
    auto droppedInBounds = ArrayRef<bool>(inBounds).take_back(dimsToDrop);
    if (llvm::is_contained(droppedInBounds, false))
      return failure();

    auto resultTargetVecType =
        VectorType::get(targetType.getShape().drop_back(dimsToDrop),
                        targetType.getElementType(),
                        targetType.getScalableDims().drop_back(dimsToDrop));

    auto loc = readOp.getLoc();
    SmallVector<OpFoldResult> sizes =
        memref::getMixedSizes(rewriter, loc, readOp.getBase());
    SmallVector<OpFoldResult> offsets(srcType.getRank(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(srcType.getRank(),
                                      rewriter.getIndexAttr(1));
    MemRefType resultMemrefType = memref::SubViewOp::inferRankReducedResultType(
        srcType.getShape().drop_back(dimsToDrop), srcType, offsets, sizes,
        strides);
    ArrayAttr inBoundsAttr = rewriter.getArrayAttr(
        readOp.getInBoundsAttr().getValue().drop_back(dimsToDrop));
    Value rankedReducedView =
        memref::SubViewOp::create(rewriter, loc, resultMemrefType,
                                  readOp.getBase(), offsets, sizes, strides);
    auto permMap = getTransferMinorIdentityMap(
        cast<ShapedType>(rankedReducedView.getType()), resultTargetVecType);
    Value result = vector::TransferReadOp::create(
        rewriter, loc, resultTargetVecType, rankedReducedView,
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
///
/// Note, this pattern will not collapse "scalable unit" dims (i.e. `[1]`).
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

    auto srcType = dyn_cast<MemRefType>(writeOp.getBase().getType());
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

    auto inBounds = writeOp.getInBoundsValues();
    auto droppedInBounds = ArrayRef<bool>(inBounds).take_back(dimsToDrop);
    if (llvm::is_contained(droppedInBounds, false))
      return failure();

    auto resultTargetVecType =
        VectorType::get(targetType.getShape().drop_back(dimsToDrop),
                        targetType.getElementType(),
                        targetType.getScalableDims().drop_back(dimsToDrop));

    Location loc = writeOp.getLoc();
    SmallVector<OpFoldResult> sizes =
        memref::getMixedSizes(rewriter, loc, writeOp.getBase());
    SmallVector<OpFoldResult> offsets(srcType.getRank(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(srcType.getRank(),
                                      rewriter.getIndexAttr(1));
    MemRefType resultMemrefType = memref::SubViewOp::inferRankReducedResultType(
        srcType.getShape().drop_back(dimsToDrop), srcType, offsets, sizes,
        strides);
    ArrayAttr inBoundsAttr = rewriter.getArrayAttr(
        writeOp.getInBoundsAttr().getValue().drop_back(dimsToDrop));

    Value rankedReducedView =
        memref::SubViewOp::create(rewriter, loc, resultMemrefType,
                                  writeOp.getBase(), offsets, sizes, strides);
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
            vector::TransposeOp::create(rewriter, loc, sext.getIn(), perm);
        VectorType newType =
            cast<VectorType>(trans.getType())
                .clone(cast<VectorType>(mat.getType()).getElementType());
        return arith::ExtSIOp::create(rewriter, loc, newType, trans);
      }
      if (auto zext = mat.getDefiningOp<arith::ExtUIOp>()) {
        Value trans =
            vector::TransposeOp::create(rewriter, loc, zext.getIn(), perm);
        VectorType newType =
            VectorType::get(cast<VectorType>(trans.getType()).getShape(),
                            cast<VectorType>(mat.getType()).getElementType());
        return arith::ExtUIOp::create(rewriter, loc, newType, trans);
      }
      return vector::TransposeOp::create(rewriter, loc, mat, perm);
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
template <typename ExtOp>
struct FoldArithExtIntoContractionOp
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    auto lhsDefOp = contractOp.getLhs().getDefiningOp<ExtOp>();
    auto rhsDefOp = contractOp.getRhs().getDefiningOp<ExtOp>();

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
      vAdd = arith::AddFOp::create(rewriter, loc, parentReduction.getVector(),
                                   op.getVector());
    }
    rewriter.replaceOpWithNewOp<vector::ReductionOp>(op, op.getKind(), vAdd,
                                                     parentReduction.getAcc());
    return success();
  }
};

// Helper function dropping unit non-scalable dimension from a VectorType
// keeping at least 1 dimension to avoid generating 0-D vectors. Scalable unit
// dimensions are not dropped. Folding such dimensions would require "shifting"
// the scalable flag onto some other fixed-width dim (e.g. vector<[1]x4xf32> ->
// vector<[4]xf32>). This could be implemented in the future.
static VectorType dropNonScalableUnitDimFromType(VectorType inVecTy) {
  auto inVecShape = inVecTy.getShape();
  SmallVector<int64_t> newShape;
  SmallVector<bool> newScalableDims;
  for (auto [dim, isScalable] :
       llvm::zip_equal(inVecShape, inVecTy.getScalableDims())) {
    if (dim == 1 && !isScalable)
      continue;

    newShape.push_back(dim);
    newScalableDims.push_back(isScalable);
  }
  // All dims have been dropped, return vector<1xeType>.
  if (newShape.empty()) {
    newShape.push_back(1);
    newScalableDims.push_back(false);
  }

  return VectorType::get(newShape, inVecTy.getElementType(), newScalableDims);
}

/// For vectors with at least one unit dim, replaces:
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
///  %mul = arith.mulf %B_row, %A_row : vector<1x[4]xf32>
///  %cast = vector.shape_cast %mul : vector<1x[4]xf32> to vector<[4]xf32>
///
/// gets converted to:
///
///  %B_row_sc = vector.shape_cast %B_row : vector<1x[4]xf32> to vector<[4]xf32>
///  %A_row_sc = vector.shape_cast %A_row : vector<1x[4]xf32> to vector<[4]xf32>
///  %mul = arith.mulf %B_row_sc, %A_row_sc : vector<[4]xf32>
///  %cast_new = vector.shape_cast %mul : vector<[4]xf32> to vector<1x[4]xf32>
///  %cast = vector.shape_cast %cast_new : vector<1x[4]xf32> to vector<[4]xf32>
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

    // Check the operand pre-conditions. For `Elementwise` ops all operands are
    // guaranteed to have identical shapes (with some exceptions such as
    // `arith.select`) and it suffices to only check one of them.
    auto sourceVectorType = dyn_cast<VectorType>(op->getOperand(0).getType());
    if (!sourceVectorType)
      return failure();
    if (sourceVectorType.getRank() < 2)
      return failure();

    SmallVector<Value> newOperands;
    auto loc = op->getLoc();
    for (auto operand : op->getOperands()) {
      auto opVectorType = cast<VectorType>(operand.getType());
      auto newVType = dropNonScalableUnitDimFromType(opVectorType);
      if (newVType == opVectorType)
        return rewriter.notifyMatchFailure(op, "No unit dimension to remove.");

      auto opSC = vector::ShapeCastOp::create(rewriter, loc, newVType, operand);
      newOperands.push_back(opSC);
    }

    VectorType newResultVectorType =
        dropNonScalableUnitDimFromType(resultVectorType);
    // Create an updated elementwise Op without unit dim.
    Operation *elementwiseOp =
        rewriter.create(loc, op->getName().getIdentifier(), newOperands,
                        newResultVectorType, op->getAttrs());

    // Restore the unit dim by applying vector.shape_cast to the result.
    rewriter.replaceOpWithNewOp<ShapeCastOp>(op, resultVectorType,
                                             elementwiseOp->getResult(0));

    return success();
  }
};

/// A pattern to drop unit dims from vector.transpose.
///
/// Example:
///
///  BEFORE:
///  ```mlir
///  %transpose = vector.transpose %vector, [3, 0, 1, 2]
///    : vector<1x1x4x[4]xf32> to vector<[4]x1x1x4xf32>
///  ```
///
///  AFTER:
///  ```mlir
///  %dropDims = vector.shape_cast %vector
///    : vector<1x1x4x[4]xf32> to vector<4x[4]xf32>
///  %transpose = vector.transpose %0, [1, 0]
///    : vector<4x[4]xf32> to vector<[4]x4xf32>
///  %restoreDims = vector.shape_cast %transpose
///    : vector<[4]x4xf32> to vector<[4]x1x1x4xf32>
///  ```
struct DropUnitDimsFromTransposeOp final
    : OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    VectorType sourceType = op.getSourceVectorType();
    VectorType sourceTypeWithoutUnitDims =
        dropNonScalableUnitDimFromType(sourceType);

    if (sourceType == sourceTypeWithoutUnitDims)
      return failure();

    // Construct a map from dimIdx -> number of dims dropped before dimIdx.
    auto sourceDims = llvm::to_vector(vector::getDims(sourceType));
    SmallVector<int64_t> droppedDimsBefore(sourceType.getRank());
    int64_t droppedDims = 0;
    for (auto [i, dim] : llvm::enumerate(sourceDims)) {
      droppedDimsBefore[i] = droppedDims;
      if (dim == std::make_tuple(1, false))
        ++droppedDims;
    }

    // Drop unit dims from transpose permutation.
    ArrayRef<int64_t> perm = op.getPermutation();
    SmallVector<int64_t> newPerm;
    for (int64_t idx : perm) {
      if (sourceDims[idx] == std::make_tuple(1, false))
        continue;
      newPerm.push_back(idx - droppedDimsBefore[idx]);
    }

    // Fixup for `newPerm`. The `sourceTypeWithoutUnitDims` could be vector<1xT>
    // type when the dimensions are unit dimensions. In this case, the newPerm
    // should be [0].
    if (newPerm.empty()) {
      newPerm.push_back(0);
    }

    Location loc = op.getLoc();
    // Drop the unit dims via shape_cast.
    auto dropDimsShapeCast = vector::ShapeCastOp::create(
        rewriter, loc, sourceTypeWithoutUnitDims, op.getVector());
    // Create the new transpose.
    auto transposeWithoutUnitDims =
        vector::TransposeOp::create(rewriter, loc, dropDimsShapeCast, newPerm);
    // Restore the unit dims via shape cast.
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        op, op.getResultVectorType(), transposeWithoutUnitDims);

    return success();
  }
};

/// A pattern to drop unit dims from the iter_args of an scf.for.
///
/// Example:
///
///  BEFORE:
///  ```mlir
///  %res = scf.for ... iter_args(%iter = %init) -> vector<[4]x1x1x4xf32> {
///    ...
///    scf.yield %
///  }
///  ```
///
///  AFTER:
///  ```mlir
///  %drop = vector.shape_cast %init
///    : vector<4x1x1x[4]xf32> to vector<4x[4]xf32>
///  %new_loop = scf.for ... iter_args(%iter = %drop) -> vector<[4]x4xf32> {
///    %new_iter = vector.shape_cast %iter
///      : vector<[4]x4xf32> to vector<[4]x1x1x4xf32>
///    ...
///  }
///  %res = vector.shape_cast %new_loop
///    : vector<[4]x4xf32> to vector<[4]x1x1x4xf32>
///  ```
struct DropUnitDimsFromScfForOp final : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    /// Find the first iter_arg with droppable unit dims. Further applications
    /// of this pattern will apply to later arguments.
    for (OpOperand &operand : forOp.getInitArgsMutable()) {
      auto vectorType = dyn_cast<VectorType>(operand.get().getType());
      if (!vectorType)
        continue;

      VectorType newVectorType = dropNonScalableUnitDimFromType(vectorType);
      if (vectorType == newVectorType)
        continue;

      // Create a new ForOp with that iter operand replaced.
      auto castFn = [](OpBuilder &b, Location loc, Type type, Value source) {
        return vector::ShapeCastOp::create(b, loc, type, source);
      };

      Value replacement =
          castFn(rewriter, forOp.getLoc(), newVectorType, operand.get());
      rewriter.replaceOp(forOp,
                         replaceAndCastForOpIterArg(rewriter, forOp, operand,
                                                    replacement, castFn));
      return success();
    }
    return failure();
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

    auto newAdd = arith::AddFOp::create(rewriter, vAdd.getLoc(),
                                        addLhs.getLhs(), vAdd.getRhs());
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
      extractedElem = vector::ExtractOp::create(rewriter, loc, op.getVector(),
                                                static_cast<int64_t>(idx));

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

/// Fold `mulf(tr(broadcast(A)), broadcast(B))` into `vector.outerproduct(A,
/// B)`.
/// Example:
///  %lhsBcast = vector.broadcast %lhs : vector<4xi32> to vector<4x4xi32>
///  %lhsT = vector.transpose %lhsBcast, [1, 0] : vector<4x4xi32> to
///  vector<4x4xi32> %rhsBcast = vector.broadcast %rhs : vector<4xi32> to
///  vector<4x4xi32> %mul = arith.muli %lhsT, %rhsBcast : vector<4x4xi32>
///
/// Becomes :
///
///  %res = vector.outerproduct %lhs, %rhs : vector<4xi32>, vector<4xi32>
///
/// Supports only 1D-to-2D broadcasts. The following cases are not supported.
/// %ex1 = vector.broadcast %lhsCast : vector<1x4xf32> to vector<4x4xf32>
/// %ex2 = vector.broadcast %lhsCast : f32 to vector<4x4xf32>
/// %ex3 = vector.broadcast %lhsCast : vector<1x1xf32> to vector<4x4xf32>
template <typename MulOpType>
struct FoldArithToVectorOuterProduct : public OpRewritePattern<MulOpType> {
  using OpRewritePattern<MulOpType>::OpRewritePattern;
  // Returns whether a vector.broadcast matches requirements for an outerproduct
  // pattern. aka a 1D-to-2D broadcastOp without broadcasted unit dimension.
  bool isValidBroadcastSource(vector::BroadcastOp broadcastOp) const {
    // Fail if it is not a 1-to-2 dimension to broadcast to avoid generating
    // shape_casts/broadcasts which does not belong in this pattern.
    if (!broadcastOp.computeBroadcastedUnitDims().empty())
      return false;
    // Avoid broadcast like f32 or vector<f32> -> ResType
    auto srcType = dyn_cast<VectorType>(broadcastOp.getSourceType());
    return srcType && srcType.getRank() != 2;
  }

  LogicalResult matchAndRewrite(MulOpType mulOp,
                                PatternRewriter &rewriter) const override {
    auto resType = llvm::dyn_cast<VectorType>(mulOp.getResult().getType());
    if (!resType)
      return failure();
    if (resType.getRank() != 2)
      return failure();
    /// If operandA can be written as tr(broadcast(A)) and operandB as
    /// broadcast(B) where broadcasts are 1D-to-2D, create and return
    /// vector.outerproduct(A, B). Returns failure() otherwise.
    auto matchOuterProduct =
        [&](Value operandA,
            Value operandB) -> FailureOr<vector::OuterProductOp> {
      auto transposedLhs = operandA.getDefiningOp<vector::TransposeOp>();
      if (!transposedLhs)
        return failure();
      // Fail unless this is a true 2-D matrix transpose.
      ArrayRef<int64_t> permutation = transposedLhs.getPermutation();
      if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0)
        return failure();

      auto broadcastedLhs =
          transposedLhs.getVector().getDefiningOp<vector::BroadcastOp>();
      if (!broadcastedLhs || !isValidBroadcastSource(broadcastedLhs))
        return failure();

      auto broadcastedRhs = operandB.getDefiningOp<vector::BroadcastOp>();
      if (!broadcastedRhs || !isValidBroadcastSource(broadcastedRhs))
        return failure();

      return vector::OuterProductOp::create(
          rewriter, mulOp->getLoc(), resType, broadcastedLhs.getSource(),
          broadcastedRhs.getSource(), Value(), vector::CombiningKind::ADD);
    };

    Value lhs = mulOp->getOperand(0), rhs = mulOp->getOperand(1);
    auto maybeOuterP = matchOuterProduct(lhs, rhs);
    // Handle commutativity, the transposed op is the outerproduct LHS.
    if (failed(maybeOuterP))
      maybeOuterP = matchOuterProduct(rhs, lhs);
    if (failed(maybeOuterP))
      return failure();
    rewriter.replaceOp(mulOp, maybeOuterP->getResult());
    return success();
  }
};

} // namespace

void mlir::vector::populateFoldArithExtensionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldArithExtIntoContractionOp<arith::ExtFOp>,
               FoldArithExtIntoContractionOp<arith::ExtSIOp>>(
      patterns.getContext());
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

void mlir::vector::populateDropUnitDimWithShapeCastPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<DropUnitDimFromElementwiseOps, DropUnitDimsFromScfForOp,
               DropUnitDimsFromTransposeOp>(patterns.getContext(), benefit);
}

void mlir::vector::populateBubbleVectorBitCastOpPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<BubbleDownVectorBitCastForExtract,
               BubbleDownBitCastForStridedSliceExtract,
               BubbleUpBitCastForInsert, BubbleUpBitCastForStridedSliceInsert>(
      patterns.getContext(), benefit);
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
  patterns.add<MultiReduceToContract, CombineContractBroadcastMask,
               CombineContractABTranspose, CombineContractResultTranspose>(
      patterns.getContext(), benefit);
}

void mlir::vector::populateDropInnerMostUnitDimsXferOpPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<DropInnerMostUnitDimsTransferRead,
               DropInnerMostUnitDimsTransferWrite>(patterns.getContext(),
                                                   benefit);
}

void mlir::vector::populateSinkVectorOpsPatterns(RewritePatternSet &patterns,
                                                 PatternBenefit benefit) {
  patterns.add<ReorderElementwiseOpsOnTranspose, ReorderCastOpsOnBroadcast,
               ReorderElementwiseOpsOnBroadcast, ExtractOpFromElementwise>(
      patterns.getContext(), benefit);
}

void mlir::vector::populateSinkVectorMemOpsPatterns(RewritePatternSet &patterns,
                                                    PatternBenefit benefit) {
  // TODO: Consider converting these patterns to canonicalizations.
  patterns.add<ExtractOpFromLoad, StoreOpFromSplatOrBroadcast>(
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

void mlir::vector::populateElementwiseToVectorOpsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldArithToVectorOuterProduct<arith::MulFOp>,
               FoldArithToVectorOuterProduct<arith::MulIOp>>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// TableGen'd enum attribute definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/VectorTransformsEnums.cpp.inc"
