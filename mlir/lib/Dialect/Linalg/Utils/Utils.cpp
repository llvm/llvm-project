//===- Utils.cpp - Utilities to support the Linalg dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-utils"

using namespace mlir;
using namespace presburger;
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::scf;

namespace {

// Helper visitor to determine whether an AffineExpr is tiled.
// This is achieved by traversing every AffineDimExpr with position `pos` and
// checking whether the corresponding `tileSizes[pos]` is non-zero.
// This also enforces only positive coefficients occur in multiplications.
//
// Example:
//   `d0 + 2 * d1 + d3` is tiled by [0, 0, 0, 2] but not by [0, 0, 2, 0]
//
struct TileCheck : public AffineExprVisitor<TileCheck> {
  TileCheck(ArrayRef<OpFoldResult> tileSizes) : tileSizes(tileSizes) {}

  void visitDimExpr(AffineDimExpr expr) {
    isTiled |= !isZeroInteger(tileSizes[expr.getPosition()]);
  }
  void visitAffineBinaryOpExpr(AffineBinaryOpExpr expr) {
    visit(expr.getLHS());
    visit(expr.getRHS());
    if (expr.getKind() == mlir::AffineExprKind::Mul)
      assert(cast<AffineConstantExpr>(expr.getRHS()).getValue() > 0 &&
             "nonpositive multiplying coefficient");
  }
  bool isTiled = false;
  ArrayRef<OpFoldResult> tileSizes;
};

} // namespace

static bool isTiled(AffineExpr expr, ArrayRef<OpFoldResult> tileSizes) {
  if (!expr)
    return false;
  TileCheck t(tileSizes);
  t.visit(expr);
  return t.isTiled;
}

// Checks whether the `map  varies with respect to a non-zero `tileSize`.
static bool isTiled(AffineMap map, ArrayRef<OpFoldResult> tileSizes) {
  if (!map)
    return false;
  for (unsigned r = 0; r < map.getNumResults(); ++r)
    if (isTiled(map.getResult(r), tileSizes))
      return true;
  return false;
}

std::optional<RegionMatcher::BinaryOpKind>
RegionMatcher::matchAsScalarBinaryOp(GenericOp op) {
  auto &region = op.getRegion();
  if (!region.hasOneBlock())
    return std::nullopt;

  Block &block = region.front();
  if (block.getNumArguments() != 2 ||
      !block.getArgument(0).getType().isSignlessIntOrFloat() ||
      !block.getArgument(1).getType().isSignlessIntOrFloat())
    return std::nullopt;

  auto &ops = block.getOperations();
  if (!llvm::hasSingleElement(block.without_terminator()))
    return std::nullopt;

  using mlir::matchers::m_Val;
  auto a = m_Val(block.getArgument(0));
  auto b = m_Val(block.getArgument(1));

  auto addPattern = m_Op<linalg::YieldOp>(m_Op<arith::AddIOp>(a, b));
  if (addPattern.match(&ops.back()))
    return BinaryOpKind::IAdd;

  return std::nullopt;
}

/// Explicit instantiation of loop nest generator for different loop types.
template struct mlir::linalg::GenerateLoopNest<scf::ForOp>;
template struct mlir::linalg::GenerateLoopNest<scf::ParallelOp>;
template struct mlir::linalg::GenerateLoopNest<AffineForOp>;

/// Given a list of subview ranges, extract individual values for lower, upper
/// bounds and steps and put them into the corresponding vectors.
static void unpackRanges(OpBuilder &builder, Location loc,
                         ArrayRef<Range> ranges, SmallVectorImpl<Value> &lbs,
                         SmallVectorImpl<Value> &ubs,
                         SmallVectorImpl<Value> &steps) {
  for (Range range : ranges) {
    lbs.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.offset));
    ubs.emplace_back(getValueOrCreateConstantIndexOp(builder, loc, range.size));
    steps.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.stride));
  }
}

//===----------------------------------------------------------------------===//
// General utilities
//===----------------------------------------------------------------------===//
//
/// The permutation can be obtained from two permutations:
///   a) Compute the permutation vector to move the last `numPackedDims` into
///      the `innerPosDims` of a shape of rank `rank`.
///   b) Compute the permutation vector to move outer dims if the
///      `outerPerm` parameter is not empty.
/// Apply (b) permutation on (a) permutation to get the final permutation.
static SmallVector<int64_t>
computePackUnPackPerm(int64_t rank, ArrayRef<int64_t> &innerDimsPos,
                      ArrayRef<int64_t> &outerPerm,
                      PackingMetadata &packingMetadata) {
  int64_t numPackedDims = innerDimsPos.size();
  auto lastDims =
      llvm::to_vector(llvm::seq<int64_t>(rank - numPackedDims, rank));
  packingMetadata = computePackingMetadata(rank, innerDimsPos);
  SmallVector<int64_t> innerPositionsPerm =
      computePermutationVector(rank, lastDims, packingMetadata.insertPositions);

  SmallVector<int64_t> outerPos = packingMetadata.outerPositions;
  if (!outerPerm.empty())
    applyPermutationToVector(outerPos, outerPerm);
  SmallVector<int64_t> outerPositionPerm =
      computePermutationVector(rank, packingMetadata.outerPositions, outerPos);

  SmallVector<int64_t> packInverseDestPermutation = innerPositionsPerm;
  applyPermutationToVector(packInverseDestPermutation, outerPositionPerm);
  return packInverseDestPermutation;
}

namespace mlir {
namespace linalg {

SmallVector<int64_t> getPackInverseDestPerm(PackOp packOp) {

  PackingMetadata pMetadata;
  int64_t packedRank = packOp.getDestType().getRank();
  ArrayRef<int64_t> innerDimPos = packOp.getInnerDimsPos();
  ArrayRef<int64_t> outerPerm = packOp.getOuterDimsPerm();
  SmallVector<int64_t> packInvDestPerm =
      computePackUnPackPerm(packedRank, innerDimPos, outerPerm, pMetadata);
  return packInvDestPerm;
}

SmallVector<int64_t> getUnPackInverseSrcPerm(UnPackOp unpackOp) {
  PackingMetadata metadata;
  return getUnPackInverseSrcPerm(unpackOp, metadata);
}

SmallVector<int64_t> getUnPackInverseSrcPerm(UnPackOp unpackOp,
                                             PackingMetadata &metadata) {
  int64_t unpackRank = unpackOp.getSourceType().getRank();
  ArrayRef<int64_t> innerDimPos = unpackOp.getInnerDimsPos();
  ArrayRef<int64_t> outerPerm = unpackOp.getOuterDimsPerm();
  SmallVector<int64_t> unpackInvSrcPerm =
      computePackUnPackPerm(unpackRank, innerDimPos, outerPerm, metadata);
  return unpackInvSrcPerm;
}

bool allIndexingsAreProjectedPermutation(LinalgOp op) {
  return llvm::all_of(op.getIndexingMapsArray(), [](AffineMap m) {
    return m.isProjectedPermutation(/*allowZeroInResults=*/true);
  });
}

bool hasOnlyScalarElementwiseOp(Region &r) {
  if (!r.hasOneBlock())
    return false;
  for (Operation &op : r.front()) {
    if (!(isa<arith::ConstantOp, func::ConstantOp, tensor::ExtractOp,
              linalg::YieldOp, linalg::IndexOp, AffineApplyOp>(op) ||
          OpTrait::hasElementwiseMappableTraits(&op)) ||
        llvm::any_of(op.getResultTypes(),
                     [](Type type) { return !type.isIntOrIndexOrFloat(); }))
      return false;
  }
  return true;
}

bool isElementwise(LinalgOp op) {
  if (op.getNumLoops() != op.getNumParallelLoops())
    return false;

  if (!allIndexingsAreProjectedPermutation(op))
    return false;

  // TODO: relax the restrictions on indexing map.
  for (OpOperand &opOperand : op.getDpsInitsMutable()) {
    if (!op.getMatchingIndexingMap(&opOperand).isPermutation())
      return false;
  }
  return hasOnlyScalarElementwiseOp(op->getRegion(0));
}

bool isParallelIterator(utils::IteratorType iteratorType) {
  return iteratorType == utils::IteratorType::parallel;
}

bool isReductionIterator(utils::IteratorType iteratorType) {
  return iteratorType == utils::IteratorType::reduction;
}

//===----------------------------------------------------------------------===//
// Convolution matcher utilities
//===----------------------------------------------------------------------===//

/// Returns the BlockArgument that leads to `val`, if any. Traverses optional
/// ext* ops.
static BlockArgument getBlockArgumentWithOptionalExtOps(Value val) {
  BlockArgument blockArg;
  if (!(blockArg = dyn_cast<BlockArgument>(val))) {
    Operation *defOp = val.getDefiningOp();
    if (!dyn_cast_if_present<arith::ExtFOp>(defOp) &&
        !dyn_cast_if_present<arith::ExtSIOp>(defOp) &&
        !dyn_cast_if_present<arith::ExtUIOp>(defOp)) {
      return nullptr;
    }
    blockArg = dyn_cast<BlockArgument>(defOp->getOperand(0));
  }
  return blockArg;
}

/// Utility to match block body for matmul-like ops.
static bool bodyMatcherForMatmulLikeOps(Value yieldVal, Block *body) {
  Operation *addOp = yieldVal.getDefiningOp();
  if (!isa_and_present<arith::AddIOp, arith::AddFOp>(addOp))
    return false;

  Operation *mulOp = addOp->getOperand(1).getDefiningOp();
  if (!isa_and_present<arith::MulIOp, arith::MulFOp>(mulOp))
    return false;

  BlockArgument lhsBlockArg =
      getBlockArgumentWithOptionalExtOps(mulOp->getOperand(0));
  BlockArgument rhsBlockArg =
      getBlockArgumentWithOptionalExtOps(mulOp->getOperand(1));
  BlockArgument outBlockArg =
      getBlockArgumentWithOptionalExtOps(addOp->getOperand(0));
  if (!lhsBlockArg || !rhsBlockArg || !outBlockArg ||
      lhsBlockArg.getOwner() != body || rhsBlockArg.getOwner() != body ||
      outBlockArg.getOwner() != body || lhsBlockArg.getArgNumber() != 0 ||
      rhsBlockArg.getArgNumber() != 1 || outBlockArg.getArgNumber() != 2)
    return false;
  return true;
}

/// Utility to match block body for linalg.pool* ops.
template <typename... OpTypes>
static bool bodyMatcherForPoolOps(Value yieldVal, Block *body) {
  Operation *defOp = yieldVal.getDefiningOp();
  if (!(isa_and_present<OpTypes>(defOp) || ...))
    return false;

  BlockArgument lhsArg = dyn_cast<BlockArgument>(defOp->getOperand(0));
  BlockArgument rhsArg = dyn_cast<BlockArgument>(defOp->getOperand(1));
  if (!lhsArg || !rhsArg || lhsArg.getOwner() != body ||
      rhsArg.getOwner() != body || lhsArg.getArgNumber() != 2 ||
      rhsArg.getArgNumber() != 0)
    return false;
  return true;
}

static bool bodyMatcherForMaxSignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MaximumFOp, arith::MaxSIOp>(yieldVal,
                                                                  body);
}

// max_unsigned ops should not allow float data type.
// TODO: Retire OPDSL logic. Refer to :
//       https://github.com/llvm/llvm-project/issues/164800
static bool bodyMatcherForMaxUnsignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MaximumFOp, arith::MaxUIOp>(yieldVal,
                                                                  body);
}

static bool bodyMatcherForMinSignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MinimumFOp, arith::MinSIOp>(yieldVal,
                                                                  body);
}

// min_unsigned ops should not allow float data type.
// TODO: Retire OPDSL logic. Refer to :
//       https://github.com/llvm/llvm-project/issues/164800
static bool bodyMatcherForMinUnsignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MinimumFOp, arith::MinUIOp>(yieldVal,
                                                                  body);
}

static bool bodyMatcherForSumPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::AddIOp, arith::AddFOp>(yieldVal, body);
}

static mlir::AffineExpr getAffineMapDim(ArrayAttr indexingMaps,
                                        uint32_t mapIndex, uint32_t dimIndex) {
  auto affineMap = cast<AffineMapAttr>(indexingMaps[mapIndex]).getValue();
  if (dimIndex < affineMap.getNumResults())
    return affineMap.getResult(dimIndex);
  return nullptr;
}

// Check if `expr` is either:
// - a dimension expr alone (implying *1), or
// - a multiplication of dimension expr by constant.
static bool isDimTimesConstantOrDimOnly(AffineExpr expr, AffineExpr &dim,
                                        int64_t &constantValue) {
  if (auto dExpr = dyn_cast<AffineDimExpr>(expr)) {
    dim = dExpr;
    constantValue = 1;
    return true;
  }

  auto mulExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!mulExpr || mulExpr.getKind() != AffineExprKind::Mul)
    return false;

  AffineExpr lhs = mulExpr.getLHS();
  AffineExpr rhs = mulExpr.getRHS();

  if (auto dExpr = dyn_cast<AffineDimExpr>(lhs)) {
    if (auto cst = dyn_cast<AffineConstantExpr>(rhs)) {
      dim = dExpr;
      constantValue = cst.getValue();
      return true;
    }
  }
  if (auto cst = dyn_cast<AffineConstantExpr>(lhs)) {
    if (auto dExpr = dyn_cast<AffineDimExpr>(rhs)) {
      dim = dExpr;
      constantValue = cst.getValue();
      return true;
    }
  }
  return false;
}

/// Given an array of AffineMaps `indexingMaps` verify the following :-
///   indexingMaps[0].getResult(iDim) ==
///         indexingMaps[1].getResult(fDim) * <CST_1> +
///         indexingMaps[n-1].getResult(oDim) * <CST_2>
///  where, CST_1 and CST_2 can be any constant.
static bool matchConvDimAddExprPattern(ArrayAttr indexingMaps, unsigned iDim,
                                       unsigned fDim, unsigned oDim,
                                       int64_t &dilation, int64_t &stride) {
  unsigned inputMapIdx = 0, filterMapIdx = 1,
           outputMapIdx = indexingMaps.size() - 1;
  AffineExpr inpExpr = getAffineMapDim(indexingMaps, inputMapIdx, iDim);
  auto addExpr = dyn_cast<AffineBinaryOpExpr>(inpExpr);
  if (!addExpr || addExpr.getKind() != AffineExprKind::Add)
    return false;

  AffineExpr dim0, dim1;
  int64_t c0, c1;

  if (isDimTimesConstantOrDimOnly(addExpr.getLHS(), dim0, c0) &&
      isDimTimesConstantOrDimOnly(addExpr.getRHS(), dim1, c1)) {
    // Pattern matched with dims and constants extracted.
    AffineExpr fExpr = getAffineMapDim(indexingMaps, filterMapIdx, fDim);
    AffineExpr oExpr = getAffineMapDim(indexingMaps, outputMapIdx, oDim);
    if (dim0 == fExpr && dim1 == oExpr) {
      dilation = c0;
      stride = c1;
      return true;
    } else if (dim1 == fExpr && dim0 == oExpr) {
      dilation = c1;
      stride = c0;
      return true;
    }
  }
  return false;
}

/// Given an array of AffineMaps `indexingMaps` verify the following :-
///   indexingMaps[aIndex].getResult(aDim) ==
///   indexingMaps[bIndex].getResult(bDim)
static bool matchConvDimExprPattern(ArrayAttr indexingMaps, unsigned aIndex,
                                    unsigned aDim, unsigned bIndex,
                                    unsigned bDim) {
  return getAffineMapDim(indexingMaps, aIndex, aDim) ==
         getAffineMapDim(indexingMaps, bIndex, bDim);
}

/// Give an array of AffineMaps, verify each map to be of the corresponding
/// `expectedSize`.
static bool verifyConvIndexingMapSizes(ArrayAttr indexingMaps,
                                       ArrayRef<int64_t> expectedSizes) {
  if (indexingMaps.size() != expectedSizes.size())
    return false;

  for (auto [indexingMap, expectedSize] :
       llvm::zip_equal(indexingMaps, expectedSizes)) {
    auto affineMap = cast<AffineMapAttr>(indexingMap).getValue();
    if (affineMap.getNumResults() != expectedSize)
      return false;
  }
  return true;
}

/// Utility to update `dilations` and `strides` by copy the corresponding data
/// from `tempDilations` and `tempStrides`.
static void updateConvDilationsAndStrides(SmallVector<int64_t> *dilations,
                                          SmallVector<int64_t> *strides,
                                          ArrayRef<int64_t> tempDilations,
                                          ArrayRef<int64_t> tempStrides) {
  if (!(dilations && strides))
    return;
  for (auto [dilation, stride] : llvm::zip(tempDilations, tempStrides)) {
    dilations->push_back(dilation);
    strides->push_back(stride);
  }
  return;
}

// ---------------------------------------------
// Matchers for specific convolution operation.
// ---------------------------------------------

// #inputMap = affine_map<(N, W, C, w) -> (N, W + w, C)>
// #filterMap = affine_map<(N, W, C, w) -> (w, C)>
// #outputMap = affine_map<(N, W, C, w) -> (N, W, C)>
template <>
bool isaConvolutionOpOfType<linalg::DepthwiseConv1DNwcWcOp>(
    LinalgOp op, SmallVector<int64_t> *dilations,
    SmallVector<int64_t> *strides) {
  if (isa<linalg::DepthwiseConv1DNwcWcOp>(op))
    return true;

  assert(isaConvolutionOpInterface(op) &&
         "expected linalgOp to implement ConvolutionOpInterface");

  ArrayAttr indexingMaps = op.getIndexingMaps();
  if (!verifyConvIndexingMapSizes(indexingMaps, {3, 2, 3}))
    return false;

  Block *body = op.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  unsigned inputMapIdx = 0, filterMapIdx = 1, outputMapIdx = 2;

  SmallVector<int64_t> tempDilations(1, 1);
  SmallVector<int64_t> tempStrides(1, 1);
  bool returnVal =
      (matchConvDimExprPattern(indexingMaps, inputMapIdx, 0, outputMapIdx, 0) &&
       matchConvDimExprPattern(indexingMaps, inputMapIdx, 2, filterMapIdx, 1) &&
       matchConvDimExprPattern(indexingMaps, inputMapIdx, 2, outputMapIdx, 2) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0,
                                  /*oDim=*/1, tempDilations[0],
                                  tempStrides[0]) &&
       bodyMatcherForMatmulLikeOps(yieldVal, body));
  if (returnVal)
    updateConvDilationsAndStrides(dilations, strides, tempDilations,
                                  tempStrides);
  return returnVal;
}

// #inputMap = affine_map<(N, H, W, C, h, w) -> (N, C, H + h, W + w)>
// #filterMap = affine_map<(N, H, W, C, h, w) -> (C, h, w)>
// #outputMap = affine_map<(N, H, W, C, h, w) -> (N, C, H, W)>
template <>
bool isaConvolutionOpOfType<linalg::DepthwiseConv2DNchwChwOp>(
    LinalgOp op, SmallVector<int64_t> *dilations,
    SmallVector<int64_t> *strides) {
  if (isa<linalg::DepthwiseConv2DNchwChwOp>(op))
    return true;

  assert(isaConvolutionOpInterface(op) &&
         "expected linalgOp to implement ConvolutionOpInterface");

  ArrayAttr indexingMaps = op.getIndexingMaps();
  if (!verifyConvIndexingMapSizes(indexingMaps, {4, 3, 4}))
    return false;

  Block *body = op.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  unsigned inputMapIdx = 0, filterMapIdx = 1, outputMapIdx = 2;

  SmallVector<int64_t> tempDilations(2, 1);
  SmallVector<int64_t> tempStrides(2, 1);
  bool returnVal =
      (matchConvDimExprPattern(indexingMaps, inputMapIdx, 0, outputMapIdx, 0) &&
       matchConvDimExprPattern(indexingMaps, inputMapIdx, 1, filterMapIdx, 0) &&
       matchConvDimExprPattern(indexingMaps, inputMapIdx, 1, outputMapIdx, 1) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1,
                                  /*oDim=*/2, tempDilations[0],
                                  tempStrides[0]) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/3, /*fDim=*/2,
                                  /*oDim=*/3, tempDilations[1],
                                  tempStrides[1]) &&
       bodyMatcherForMatmulLikeOps(yieldVal, body));
  if (returnVal)
    updateConvDilationsAndStrides(dilations, strides, tempDilations,
                                  tempStrides);
  return returnVal;
}

// #inputMap = affine_map<(N, D, H, W, CM, d, h, w, C)
//                    -> (N, D + d, H + h, W + w, C)>
// #filterMap = affine_map<(N, D, H, W, CM, d, h, w, C)
//                    -> (d, h, w, C, CM)>
// #outputMap = affine_map<(N, D, H, W, CM, d, h, w, C)
//                    -> (N, D, H, W, C, CM)>
template <>
bool isaConvolutionOpOfType<linalg::DepthwiseConv3DNdhwcDhwcmOp>(
    LinalgOp op, SmallVector<int64_t> *dilations,
    SmallVector<int64_t> *strides) {
  if (isa<linalg::DepthwiseConv3DNdhwcDhwcmOp>(op))
    return true;

  assert(isaConvolutionOpInterface(op) &&
         "expected linalgOp to implement ConvolutionOpInterface");

  ArrayAttr indexingMaps = op.getIndexingMaps();
  if (!verifyConvIndexingMapSizes(indexingMaps, {5, 5, 6}))
    return false;

  Block *body = op.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  unsigned inputMapIdx = 0, filterMapIdx = 1, outputMapIdx = 2;

  SmallVector<int64_t> tempDilations(3, 1);
  SmallVector<int64_t> tempStrides(3, 1);
  bool returnVal =
      (matchConvDimExprPattern(indexingMaps, inputMapIdx, 0, outputMapIdx, 0) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0,
                                  /*oDim=*/1, tempDilations[0],
                                  tempStrides[0]) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1,
                                  /*oDim=*/2, tempDilations[1],
                                  tempStrides[1]) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/3, /*fDim=*/2,
                                  /*oDim=*/3, tempDilations[2],
                                  tempStrides[2]) &&
       matchConvDimExprPattern(indexingMaps, inputMapIdx, 4, filterMapIdx, 3) &&
       matchConvDimExprPattern(indexingMaps, inputMapIdx, 4, outputMapIdx, 4) &&
       matchConvDimExprPattern(indexingMaps, filterMapIdx, 4, outputMapIdx,
                               5) &&
       bodyMatcherForMatmulLikeOps(yieldVal, body));
  if (returnVal)
    updateConvDilationsAndStrides(dilations, strides, tempDilations,
                                  tempStrides);
  return returnVal;
}

// #inputMap = affine_map<(N, H, W, C, h, w) -> (N, H + h, W + w, C)>
// #filterMap = affine_map<(N, H, W, C, h, w) -> (h, w)>
// #outputMap = affine_map<(N, H, W, C, h, w) -> (N, H, W, C)>
template <>
bool isaConvolutionOpOfType<linalg::PoolingNhwcMaxOp>(
    LinalgOp op, SmallVector<int64_t> *dilations,
    SmallVector<int64_t> *strides) {
  if (isa<linalg::PoolingNhwcMaxOp>(op))
    return true;

  assert(isaConvolutionOpInterface(op) &&
         "expected linalgOp to implement ConvolutionOpInterface");

  ArrayAttr indexingMaps = op.getIndexingMaps();
  if (!verifyConvIndexingMapSizes(indexingMaps, {4, 2, 4}))
    return false;

  Block *body = op.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  unsigned inputMapIdx = 0, outputMapIdx = 2;

  SmallVector<int64_t> tempDilations(2, 1);
  SmallVector<int64_t> tempStrides(2, 1);
  bool returnVal =
      (matchConvDimExprPattern(indexingMaps, inputMapIdx, 0, outputMapIdx, 0) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0,
                                  /*oDim=*/1, tempDilations[0],
                                  tempStrides[0]) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1,
                                  /*oDim=*/2, tempDilations[1],
                                  tempStrides[1]) &&
       matchConvDimExprPattern(indexingMaps, inputMapIdx, 3, outputMapIdx, 3) &&
       bodyMatcherForMaxSignedPoolOps(yieldVal, body));
  if (returnVal)
    updateConvDilationsAndStrides(dilations, strides, tempDilations,
                                  tempStrides);
  return returnVal;
}

// #inputMap = affine_map<(N, H, W, C, h, w) -> (N, H + h, W + w, C)>
// #filterMap = affine_map<(N, H, W, C, h, w) -> (h, w)>
// #outputMap = affine_map<(N, H, W, C, h, w) -> (N, H, W, C)>
template <>
bool isaConvolutionOpOfType<linalg::PoolingNhwcMinOp>(
    LinalgOp op, SmallVector<int64_t> *dilations,
    SmallVector<int64_t> *strides) {
  if (isa<linalg::PoolingNhwcMinOp>(op))
    return true;

  assert(isaConvolutionOpInterface(op) &&
         "expected linalgOp to implement ConvolutionOpInterface");

  ArrayAttr indexingMaps = op.getIndexingMaps();
  if (!verifyConvIndexingMapSizes(indexingMaps, {4, 2, 4}))
    return false;

  Block *body = op.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  unsigned inputMapIdx = 0, outputMapIdx = 2;

  SmallVector<int64_t> tempDilations(2, 1);
  SmallVector<int64_t> tempStrides(2, 1);
  bool returnVal =
      (matchConvDimExprPattern(indexingMaps, inputMapIdx, 0, outputMapIdx, 0) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0,
                                  /*oDim=*/1, tempDilations[0],
                                  tempStrides[0]) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1,
                                  /*oDim=*/2, tempDilations[1],
                                  tempStrides[1]) &&
       matchConvDimExprPattern(indexingMaps, inputMapIdx, 3, outputMapIdx, 3) &&
       bodyMatcherForMinSignedPoolOps(yieldVal, body));
  if (returnVal)
    updateConvDilationsAndStrides(dilations, strides, tempDilations,
                                  tempStrides);
  return returnVal;
}

// #inputMap = affine_map<(N, H, W, C, h, w) -> (N, H + h, W + w, C)>
// #filterMap = affine_map<(N, H, W, C, h, w) -> (h, w)>
// #outputMap = affine_map<(N, H, W, C, h, w) -> (N, H, W, C)>
template <>
bool isaConvolutionOpOfType<linalg::PoolingNhwcSumOp>(
    LinalgOp op, SmallVector<int64_t> *dilations,
    SmallVector<int64_t> *strides) {
  if (isa<linalg::PoolingNhwcSumOp>(op))
    return true;

  assert(isaConvolutionOpInterface(op) &&
         "expected linalgOp to implement ConvolutionOpInterface");

  ArrayAttr indexingMaps = op.getIndexingMaps();
  if (!verifyConvIndexingMapSizes(indexingMaps, {4, 2, 4}))
    return false;

  Block *body = op.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  unsigned inputMapIdx = 0, outputMapIdx = 2;

  SmallVector<int64_t> tempDilations(2, 1);
  SmallVector<int64_t> tempStrides(2, 1);
  bool returnVal =
      (matchConvDimExprPattern(indexingMaps, inputMapIdx, 0, outputMapIdx, 0) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0,
                                  /*oDim=*/1, tempDilations[0],
                                  tempStrides[0]) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1,
                                  /*oDim=*/2, tempDilations[1],
                                  tempStrides[1]) &&
       matchConvDimExprPattern(indexingMaps, inputMapIdx, 3, outputMapIdx, 3) &&
       bodyMatcherForSumPoolOps(yieldVal, body));
  if (returnVal)
    updateConvDilationsAndStrides(dilations, strides, tempDilations,
                                  tempStrides);
  return returnVal;
}

// #inputMap = affine_map<(N, H, W, C, h, w) -> (N, H + h, W + w, C)>
// #filterMap = affine_map<(N, H, W, C, h, w) -> (h, w)>
// #outputMap = affine_map<(N, H, W, C, h, w) -> (N, H, W, C)>
template <>
bool isaConvolutionOpOfType<linalg::PoolingNhwcMaxUnsignedOp>(
    LinalgOp op, SmallVector<int64_t> *dilations,
    SmallVector<int64_t> *strides) {
  if (isa<linalg::PoolingNhwcMaxUnsignedOp>(op))
    return true;

  assert(isaConvolutionOpInterface(op) &&
         "expected linalgOp to implement ConvolutionOpInterface");

  ArrayAttr indexingMaps = op.getIndexingMaps();
  if (!verifyConvIndexingMapSizes(indexingMaps, {4, 2, 4}))
    return false;

  Block *body = op.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  unsigned inputMapIdx = 0, outputMapIdx = 2;

  SmallVector<int64_t> tempDilations(2, 1);
  SmallVector<int64_t> tempStrides(2, 1);
  bool returnVal =
      (matchConvDimExprPattern(indexingMaps, inputMapIdx, 0, outputMapIdx, 0) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0,
                                  /*oDim=*/1, tempDilations[0],
                                  tempStrides[0]) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1,
                                  /*oDim=*/2, tempDilations[1],
                                  tempStrides[1]) &&
       matchConvDimExprPattern(indexingMaps, inputMapIdx, 3, outputMapIdx, 3) &&
       bodyMatcherForMaxUnsignedPoolOps(yieldVal, body));
  if (returnVal)
    updateConvDilationsAndStrides(dilations, strides, tempDilations,
                                  tempStrides);
  return returnVal;
}

// #inputMap = affine_map<(N, H, W, C, h, w) -> (N, H + h, W + w, C)>
// #filterMap = affine_map<(N, H, W, C, h, w) -> (h, w)>
// #outputMap = affine_map<(N, H, W, C, h, w) -> (N, H, W, C)>
template <>
bool isaConvolutionOpOfType<linalg::PoolingNhwcMinUnsignedOp>(
    LinalgOp op, SmallVector<int64_t> *dilations,
    SmallVector<int64_t> *strides) {
  if (isa<linalg::PoolingNhwcMinUnsignedOp>(op))
    return true;

  assert(isaConvolutionOpInterface(op) &&
         "expected linalgOp to implement ConvolutionOpInterface");

  ArrayAttr indexingMaps = op.getIndexingMaps();
  if (!verifyConvIndexingMapSizes(indexingMaps, {4, 2, 4}))
    return false;

  Block *body = op.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  unsigned inputMapIdx = 0, outputMapIdx = 2;

  SmallVector<int64_t> tempDilations(2, 1);
  SmallVector<int64_t> tempStrides(2, 1);
  bool returnVal =
      (matchConvDimExprPattern(indexingMaps, inputMapIdx, 0, outputMapIdx, 0) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0,
                                  /*oDim=*/1, tempDilations[0],
                                  tempStrides[0]) &&
       matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1,
                                  /*oDim=*/2, tempDilations[1],
                                  tempStrides[1]) &&
       matchConvDimExprPattern(indexingMaps, inputMapIdx, 3, outputMapIdx, 3) &&
       bodyMatcherForMinUnsignedPoolOps(yieldVal, body));
  if (returnVal)
    updateConvDilationsAndStrides(dilations, strides, tempDilations,
                                  tempStrides);
  return returnVal;
}

Value makeComposedPadHighOp(OpBuilder &b, Location loc, RankedTensorType type,
                            Value source, Value pad, bool nofold,
                            ValueRange typeDynDims) {
  // Exit if `source` is not defined by an ExtractSliceOp.
  auto sliceOp = source.getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Search the `source` use-def chain for padded LinalgOps.
  Value current = sliceOp.getSource();
  while (current) {
    auto linalgOp = current.getDefiningOp<LinalgOp>();
    if (!linalgOp)
      break;
    OpResult opResult = cast<OpResult>(current);
    current = linalgOp.getDpsInitOperand(opResult.getResultNumber())->get();
  }
  auto padOp = current ? current.getDefiningOp<tensor::PadOp>() : nullptr;

  // Exit if the search fails to match a tensor::PadOp at the end of the matched
  // LinalgOp sequence.
  if (!padOp)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Exit if the padded result type does not match.
  if (sliceOp.getSource().getType() != type)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Exit if the LinalgOps are not high padded.
  if (llvm::any_of(padOp.getMixedLowPad(), [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) != static_cast<int64_t>(0);
      }))
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Exit if `padOpSliceOp`, which defines the slice used by
  // `padOp`, is rank-reducing.
  auto padOpSliceOp = padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!padOpSliceOp ||
      sliceOp.getMixedSizes().size() != padOpSliceOp.getMixedSizes().size())
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Exit if the sizes of the dynamic sizes of `sliceOp` do not match the size
  // of the slice padded by `padOp`.
  if (llvm::any_of(
          llvm::zip(sliceOp.getMixedSizes(), padOpSliceOp.getMixedSizes()),
          [](std::tuple<OpFoldResult, OpFoldResult> it) {
            return !isEqualConstantIntOrValue(std::get<0>(it), std::get<1>(it));
          }))
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Exit if the padding values do not match.
  Attribute padOpPadAttr, padAttr;
  Value padOpPad = padOp.getConstantPaddingValue();
  if (!padOpPad || !matchPattern(padOpPad, m_Constant(&padOpPadAttr)) ||
      !matchPattern(pad, m_Constant(&padAttr)) || padOpPadAttr != padAttr)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   typeDynDims);

  // Return the padded result if the padding values and sizes match.
  return sliceOp.getSource();
}

GenericOp makeMemRefCopyOp(OpBuilder &b, Location loc, Value from, Value to) {
  auto memrefTypeTo = cast<MemRefType>(to.getType());
#ifndef NDEBUG
  auto memrefTypeFrom = cast<MemRefType>(from.getType());
  assert(memrefTypeFrom.getRank() == memrefTypeTo.getRank() &&
         "`from` and `to` memref must have the same rank");
#endif // NDEBUG

  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(memrefTypeTo.getRank(),
                                                 utils::IteratorType::parallel);
  return linalg::GenericOp::create(
      b, loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::ArrayRef({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        linalg::YieldOp::create(b, loc, args.front());
      });
}

/// Specialization to build an scf "for" nest.
template <>
void GenerateLoopNest<scf::ForOp>::doit(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    ArrayRef<linalg::ProcInfo> procInfo) {
  assert((procInfo.empty() || (procInfo.size() == loopRanges.size())) &&
         "expected as many entries for proc info as number of loops, even if "
         "they are null entries");
  SmallVector<Value> iterArgInitValues;
  if (!linalgOp.hasPureBufferSemantics())
    llvm::append_range(iterArgInitValues, linalgOp.getDpsInits());
  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(b, loc, loopRanges, lbs, ubs, steps);
  LoopNest loopNest = mlir::scf::buildLoopNest(
      b, loc, lbs, ubs, steps, iterArgInitValues,
      [&](OpBuilder &b, Location loc, ValueRange ivs, ValueRange iterArgs) {
        assert(iterArgs.size() == iterArgInitValues.size() &&
               "expect the number of output tensors and iter args to match");
        SmallVector<Value> operandValuesToUse = linalgOp->getOperands();
        if (!iterArgs.empty()) {
          operandValuesToUse = linalgOp.getDpsInputs();
          operandValuesToUse.append(iterArgs.begin(), iterArgs.end());
        }
        return bodyBuilderFn(b, loc, ivs, operandValuesToUse);
      });

  if (loopNest.loops.empty() || procInfo.empty())
    return;

  // Filter out scf.for loops that were created out of parallel dimensions.
  for (const auto &loop : llvm::enumerate(loopNest.loops)) {
    if (procInfo[loop.index()].distributionMethod ==
        DistributionMethod::Cyclic) {
      mapLoopToProcessorIds(loop.value(), procInfo[loop.index()].procId,
                            procInfo[loop.index()].nprocs);
    }
  }
}

/// Specialization to build affine "for" nest.
template <>
void GenerateLoopNest<AffineForOp>::doit(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    ArrayRef<linalg::ProcInfo> /*procInfo*/) {
  SmallVector<Value> iterArgInitValues;
  if (!linalgOp.hasPureBufferSemantics())
    llvm::append_range(iterArgInitValues, linalgOp.getDpsInits());
  assert(iterArgInitValues.empty() && "unexpected AffineForOp init values");
  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(b, loc, loopRanges, lbs, ubs, steps);

  // Affine loops require constant steps.
  SmallVector<int64_t, 4> constantSteps;
  constantSteps.reserve(steps.size());
  for (Value v : steps) {
    auto constVal = getConstantIntValue(v);
    assert(constVal.has_value() && "Affine loops require constant steps");
    constantSteps.push_back(constVal.value());
  }

  affine::buildAffineLoopNest(b, loc, lbs, ubs, constantSteps,
                              [&](OpBuilder &b, Location loc, ValueRange ivs) {
                                bodyBuilderFn(b, loc, ivs,
                                              linalgOp->getOperands());
                              });
}

/// Update the `lb`, `ub` and `step` to get per processor `lb`, `ub` and `step`.
void updateBoundsForCyclicDistribution(OpBuilder &b, Location loc, Value procId,
                                       Value nprocs, Value &lb, Value &ub,
                                       Value &step) {
  AffineExpr d0, d1;
  bindDims(b.getContext(), d0, d1);
  AffineExpr s0 = getAffineSymbolExpr(0, b.getContext());
  lb =
      affine::makeComposedAffineApply(b, loc, d0 + d1 * s0, {lb, procId, step});
  step = affine::makeComposedAffineApply(b, loc, d0 * s0, {nprocs, step});
}

/// Generates a loop nest consisting of scf.parallel and scf.for, depending
/// on the `iteratorTypes.` Consecutive parallel loops create a single
/// scf.parallel operation; each sequential loop creates a new scf.for
/// operation. The body of the innermost loop is populated by
/// `bodyBuilderFn` that accepts a range of induction variables for all
/// loops. `ivStorage` is used to store the partial list of induction
/// variables.
// TODO: this function can be made iterative instead. However, it
// will have at most as many recursive calls as nested loops, which rarely
// exceeds 10.
static void generateParallelLoopNest(
    OpBuilder &b, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps, ArrayRef<utils::IteratorType> iteratorTypes,
    ArrayRef<linalg::ProcInfo> procInfo,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn,
    SmallVectorImpl<Value> &ivStorage) {
  assert(lbs.size() == ubs.size());
  assert(lbs.size() == steps.size());
  assert(lbs.size() == iteratorTypes.size());
  assert(procInfo.empty() || (lbs.size() == procInfo.size()));

  // If there are no (more) loops to be generated, generate the body and be
  // done with it.
  if (iteratorTypes.empty()) {
    bodyBuilderFn(b, loc, ivStorage);
    return;
  }

  // If there are no outer parallel loops, generate one sequential loop and
  // recurse.
  if (!isParallelIterator(iteratorTypes.front())) {
    LoopNest singleLoop = buildLoopNest(
        b, loc, lbs.take_front(), ubs.take_front(), steps.take_front(),
        [&](OpBuilder &b, Location loc, ValueRange ivs) {
          ivStorage.append(ivs.begin(), ivs.end());
          generateParallelLoopNest(
              b, loc, lbs.drop_front(), ubs.drop_front(), steps.drop_front(),
              iteratorTypes.drop_front(),
              procInfo.empty() ? procInfo : procInfo.drop_front(),
              bodyBuilderFn, ivStorage);
        });
    return;
  }

  unsigned nLoops = iteratorTypes.size();
  unsigned numProcessed = 0;
  DistributionMethod distributionMethod = DistributionMethod::None;
  if (procInfo.empty()) {
    numProcessed = nLoops - iteratorTypes.drop_while(isParallelIterator).size();
  } else {
    distributionMethod = procInfo.front().distributionMethod;
    numProcessed =
        nLoops - procInfo
                     .drop_while([&](linalg::ProcInfo p) {
                       return p.distributionMethod == distributionMethod;
                     })
                     .size();
  }

  auto remainderProcInfo =
      procInfo.empty() ? procInfo : procInfo.drop_front(numProcessed);
  switch (distributionMethod) {
  case DistributionMethod::None: {
    // Generate a single parallel loop-nest operation for all outermost
    // parallel loops and recurse.
    scf::ParallelOp::create(
        b, loc, lbs.take_front(numProcessed), ubs.take_front(numProcessed),
        steps.take_front(numProcessed),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange localIvs) {
          ivStorage.append(localIvs.begin(), localIvs.end());
          generateParallelLoopNest(
              nestedBuilder, nestedLoc, lbs.drop_front(numProcessed),
              ubs.drop_front(numProcessed), steps.drop_front(numProcessed),
              iteratorTypes.drop_front(numProcessed), remainderProcInfo,
              bodyBuilderFn, ivStorage);
        });
    return;
  }
  case DistributionMethod::Cyclic: {
    // Generate a single parallel loop-nest operation for all outermost
    // parallel loops and recurse.
    scf::ParallelOp::create(
        b, loc, lbs.take_front(numProcessed), ubs.take_front(numProcessed),
        steps.take_front(numProcessed),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange localIvs) {
          ivStorage.append(localIvs.begin(), localIvs.end());
          generateParallelLoopNest(
              nestedBuilder, nestedLoc, lbs.drop_front(numProcessed),
              ubs.drop_front(numProcessed), steps.drop_front(numProcessed),
              iteratorTypes.drop_front(numProcessed), remainderProcInfo,
              bodyBuilderFn, ivStorage);
        });
    return;
  }
  case DistributionMethod::CyclicNumProcsGeNumIters: {
    // Check (for the processed loops) that the iteration is in-bounds.
    ArithBuilder ab(b, loc);
    Value cond = ab.slt(lbs[0], ubs[0]);
    for (unsigned i = 1; i < numProcessed; ++i)
      cond = ab._and(cond, ab.slt(lbs[i], ubs[i]));
    ivStorage.append(lbs.begin(), std::next(lbs.begin(), numProcessed));
    scf::IfOp::create(b, loc, cond, [&](OpBuilder &b, Location loc) {
      generateParallelLoopNest(b, loc, lbs.drop_front(numProcessed),
                               ubs.drop_front(numProcessed),
                               steps.drop_front(numProcessed),
                               iteratorTypes.drop_front(numProcessed),
                               remainderProcInfo, bodyBuilderFn, ivStorage);
      scf::YieldOp::create(b, loc, ValueRange{});
    });
    return;
  }
  case DistributionMethod::CyclicNumProcsEqNumIters:
    // No check/loops needed here. Set the `%iv` to be the `%lb` and proceed
    // with inner loop generation.
    ivStorage.append(lbs.begin(), std::next(lbs.begin(), numProcessed));
    generateParallelLoopNest(
        b, loc, lbs.drop_front(numProcessed), ubs.drop_front(numProcessed),
        steps.drop_front(numProcessed), iteratorTypes.drop_front(numProcessed),
        remainderProcInfo, bodyBuilderFn, ivStorage);
    return;
  }
}

/// Specialization for generating a mix of parallel and sequential scf loops.
template <>
void GenerateLoopNest<scf::ParallelOp>::doit(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    ArrayRef<linalg::ProcInfo> procInfo) {
  SmallVector<Value> iterArgInitValues;
  if (!linalgOp.hasPureBufferSemantics())
    llvm::append_range(iterArgInitValues, linalgOp.getDpsInits());
  assert(iterArgInitValues.empty() && "unexpected ParallelOp init values");
  // This function may be passed more iterator types than ranges.
  assert(iteratorTypes.size() >= loopRanges.size() &&
         "expected iterator type for all ranges");
  assert((procInfo.empty() || (procInfo.size() == loopRanges.size())) &&
         "expected proc information for all loops when present");
  iteratorTypes = iteratorTypes.take_front(loopRanges.size());
  SmallVector<Value, 8> lbsStorage, ubsStorage, stepsStorage, ivs;
  unsigned numLoops = iteratorTypes.size();
  ivs.reserve(numLoops);
  lbsStorage.reserve(numLoops);
  ubsStorage.reserve(numLoops);
  stepsStorage.reserve(numLoops);

  // Get the loop lb, ub, and step.
  unpackRanges(b, loc, loopRanges, lbsStorage, ubsStorage, stepsStorage);

  // Modify the lb, ub, and step based on the distribution options.
  for (const auto &it : llvm::enumerate(procInfo)) {
    if (it.value().distributionMethod != linalg::DistributionMethod::None) {
      updateBoundsForCyclicDistribution(
          b, loc, it.value().procId, it.value().nprocs, lbsStorage[it.index()],
          ubsStorage[it.index()], stepsStorage[it.index()]);
    }
  }
  ValueRange lbs(lbsStorage), ubs(ubsStorage), steps(stepsStorage);
  generateParallelLoopNest(
      b, loc, lbs, ubs, steps, iteratorTypes, procInfo,
      [&](OpBuilder &b, Location loc, ValueRange ivs) {
        bodyBuilderFn(b, loc, ivs, linalgOp->getOperands());
      },
      ivs);

  assert(ivs.size() == iteratorTypes.size() && "did not generate enough loops");
}

static Operation *materializeTiledShape(OpBuilder &builder, Location loc,
                                        Value valueToTile,
                                        const SliceParameters &sliceParams) {
  auto shapedType = dyn_cast<ShapedType>(valueToTile.getType());
  auto *sliceOp = TypeSwitch<ShapedType, Operation *>(shapedType)
                      .Case([&](MemRefType) {
                        return memref::SubViewOp::create(
                            builder, loc, valueToTile, sliceParams.offsets,
                            sliceParams.sizes, sliceParams.strides);
                      })
                      .Case([&](RankedTensorType) {
                        return tensor::ExtractSliceOp::create(
                            builder, loc, valueToTile, sliceParams.offsets,
                            sliceParams.sizes, sliceParams.strides);
                      })
                      .DefaultUnreachable("Unexpected shaped type");
  return sliceOp;
}

Operation *makeTiledShape(OpBuilder &builder, Location loc, Value valueToTile,
                          ArrayRef<OpFoldResult> tileSizes, AffineMap map,
                          ArrayRef<OpFoldResult> lbs,
                          ArrayRef<OpFoldResult> ubs,
                          ArrayRef<OpFoldResult> subShapeSizes,
                          bool omitPartialTileCheck) {
  SliceParameters sliceParams =
      computeSliceParameters(builder, loc, valueToTile, tileSizes, map, lbs,
                             ubs, subShapeSizes, omitPartialTileCheck);
  return materializeTiledShape(builder, loc, valueToTile, sliceParams);
}

SliceParameters
computeSliceParameters(OpBuilder &builder, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> tileSizes, AffineMap map,
                       ArrayRef<OpFoldResult> lbs, ArrayRef<OpFoldResult> ubs,
                       ArrayRef<OpFoldResult> subShapeSizes,
                       bool omitPartialTileCheck) {
  auto shapedType = dyn_cast<ShapedType>(valueToTile.getType());
  assert(shapedType && "only shaped types can be tiled");
  ArrayRef<int64_t> shape = shapedType.getShape();
  int64_t rank = shapedType.getRank();

  // Compute offsets/sizes/strides for the tile.
  SliceParameters sliceParams;
  sliceParams.offsets.reserve(rank);
  sliceParams.sizes.reserve(rank);
  sliceParams.strides.reserve(rank);
  for (unsigned r = 0; r < rank; ++r) {
    LLVM_DEBUG(llvm::dbgs() << "computeSliceParameters: for dim#" << r);
    if (!isTiled(map.getSubMap({r}), tileSizes)) {
      sliceParams.offsets.push_back(builder.getIndexAttr(0));
      OpFoldResult dim = createFoldedDimOp(builder, loc, valueToTile, r);
      sliceParams.sizes.push_back(dim);
      sliceParams.strides.push_back(builder.getIndexAttr(1));
      LLVM_DEBUG(llvm::dbgs() << ": not tiled: use size: " << dim << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subsize...\n");

    // Tiling creates a new slice at the proper index, the slice step is 1
    // (i.e. the op does not subsample, stepping occurs in the loop).
    auto m = map.getSubMap({r});
    LLVM_DEBUG(llvm::dbgs() << "computeSliceParameters: submap: " << m << "\n");
    IRRewriter rewriter(builder);
    // The offset of the slice is m(lbs) - m(0).
    SmallVector<Attribute> zeros(lbs.size(), rewriter.getIndexAttr(0));
    SmallVector<Attribute> mAtZero;
    [[maybe_unused]] auto res = m.constantFold(zeros, mAtZero);
    assert(succeeded(res) && "affine_map must be evaluatable (not symbols)");
    int64_t mAtZeroInt =
        cast<IntegerAttr>(mAtZero[0]).getValue().getSExtValue();
    OpFoldResult offset = makeComposedFoldedAffineApply(
        rewriter, loc, m.getResult(0) - mAtZeroInt, lbs);
    sliceParams.offsets.push_back(offset);

    OpFoldResult closedIntSize =
        makeComposedFoldedAffineApply(rewriter, loc, m, subShapeSizes);
    // Resulting size needs to be made half open interval again.
    AffineExpr s0 = getAffineSymbolExpr(0, builder.getContext());
    OpFoldResult size =
        makeComposedFoldedAffineApply(rewriter, loc, s0 + 1, closedIntSize);
    LLVM_DEBUG(llvm::dbgs()
               << "computeSliceParameters: raw size: " << size << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "computeSliceParameters: new offset: " << offset << "\n");
    sliceParams.strides.push_back(builder.getIndexAttr(1));

    if (omitPartialTileCheck) {
      // We statically know that the partial/boundary tile condition is
      // unnecessary.
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << size << "\n");
      sliceParams.sizes.push_back(size);
      continue;
    }

    // The size of the subview / extract_slice should be trimmed to avoid
    // out-of-bounds accesses, unless:
    // a. We statically know the subshape size divides the shape size evenly.
    // b. The subshape size is 1. According to the way the loops are set up,
    //    tensors with "0" dimensions would never be constructed.
    int64_t shapeSize = shape[r];
    std::optional<int64_t> sizeCst = getConstantIntValue(size);
    auto hasTileSizeOne = sizeCst == 1;
    auto dividesEvenly = sizeCst && ShapedType::isStatic(shapeSize) &&
                         ((shapeSize % *sizeCst) == 0);
    if (!hasTileSizeOne && !dividesEvenly) {
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: shapeSize=" << shapeSize
                              << ", size: " << size
                              << ": make sure in bound with affine.min\n");

      AffineExpr dim0, dim1, dim2;
      MLIRContext *context = builder.getContext();
      bindDims(context, dim0, dim1, dim2);

      // Get the dimension size for this dimension. We need to first calculate
      // the max index and then plus one. This is important because for
      // convolution ops, we have its input window dimension's affine map of the
      // form `(d0 * s0 + d1)`, where `d0`/`d1 is an output/filter window
      // dimension and `s0` is stride. Directly use the dimension size of
      // output/filer window dimensions will cause incorrect calculation.
      AffineMap minusOneMap = AffineMap::inferFromExprList(
                                  {ArrayRef<AffineExpr>{dim0 - 1}}, context)
                                  .front();
      AffineMap plusOneMap = AffineMap::inferFromExprList(
                                 {ArrayRef<AffineExpr>{dim0 + 1}}, context)
                                 .front();
      SmallVector<OpFoldResult> maxIndices =
          llvm::to_vector(llvm::map_range(ubs, [&](OpFoldResult ub) {
            return makeComposedFoldedAffineApply(rewriter, loc, minusOneMap,
                                                 {ub});
          }));
      OpFoldResult maxIndex =
          makeComposedFoldedAffineApply(rewriter, loc, m, maxIndices);
      OpFoldResult d =
          makeComposedFoldedAffineApply(rewriter, loc, plusOneMap, {maxIndex});

      // Compute min(dim - offset, size) to avoid out-of-bounds accesses.
      AffineMap minMap = AffineMap::inferFromExprList(
                             {ArrayRef<AffineExpr>{dim1 - dim2, dim0}}, context)
                             .front();
      size =
          makeComposedFoldedAffineMin(rewriter, loc, minMap, {size, d, offset});
    }
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << size << "\n");
    sliceParams.sizes.push_back(size);
  }
  return sliceParams;
}

SmallVector<OpFoldResult> computeTileOffsets(OpBuilder &b, Location loc,
                                             ArrayRef<OpFoldResult> ivs,
                                             ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<OpFoldResult> offsets;
  for (unsigned idx = 0, idxIvs = 0, e = tileSizes.size(); idx < e; ++idx) {
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for loop#" << idx << "\n");
    bool isTiled = !isZeroInteger(tileSizes[idx]);
    offsets.push_back(isTiled ? ivs[idxIvs++] : b.getIndexAttr(0));
    LLVM_DEBUG(llvm::dbgs()
               << "computeTileOffsets: " << offsets.back() << "\n");
  }
  return offsets;
}

SmallVector<OpFoldResult> computeTileSizes(OpBuilder &b, Location loc,
                                           ArrayRef<OpFoldResult> tileSizes,
                                           ArrayRef<OpFoldResult> sizeBounds) {
  SmallVector<OpFoldResult> sizes;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx) {
    bool isTiled = !isZeroInteger(tileSizes[idx]);
    // Before composing, we need to make range a closed interval.
    OpFoldResult size = isTiled ? tileSizes[idx] : sizeBounds[idx];
    AffineExpr d0 = getAffineDimExpr(0, b.getContext());
    IRRewriter rewriter(b);
    sizes.push_back(makeComposedFoldedAffineApply(rewriter, loc, d0 - 1, size));
    LLVM_DEBUG(llvm::dbgs() << "computeTileSizes: " << sizes.back() << "\n");
  }
  return sizes;
}

SmallVector<Type> getTensorOutputTypes(LinalgOp op, ValueRange operands) {
  if (op.hasPureBufferSemantics())
    return {};
  return llvm::to_vector(
      llvm::map_range(op.getDpsInitsMutable(), [&](OpOperand &opOperand) {
        return operands[opOperand.getOperandNumber()].getType();
      }));
}

SmallVector<Value> insertSlicesBack(OpBuilder &builder, Location loc,
                                    LinalgOp op, ValueRange operands,
                                    ValueRange results) {
  if (op.hasPureBufferSemantics())
    return {};
  SmallVector<Value> tensorResults;
  tensorResults.reserve(results.size());
  // Insert a insert_slice for each output tensor.
  unsigned resultIdx = 0;
  for (OpOperand &opOperand : op.getDpsInitsMutable()) {
    // TODO: use an interface/adaptor to avoid leaking position in
    // `tiledOperands`.
    Value outputTensor = operands[opOperand.getOperandNumber()];
    if (auto sliceOp = outputTensor.getDefiningOp<tensor::ExtractSliceOp>()) {
      Value inserted = tensor::InsertSliceOp::create(
          builder, loc, sliceOp.getSource().getType(), results[resultIdx],
          sliceOp.getSource(), sliceOp.getOffsets(), sliceOp.getSizes(),
          sliceOp.getStrides(), sliceOp.getStaticOffsets(),
          sliceOp.getStaticSizes(), sliceOp.getStaticStrides());
      tensorResults.push_back(inserted);
    } else {
      tensorResults.push_back(results[resultIdx]);
    }
    ++resultIdx;
  }
  return tensorResults;
}

SmallVector<std::optional<SliceParameters>>
computeAllSliceParameters(OpBuilder &builder, Location loc, LinalgOp linalgOp,
                          ValueRange valuesToTile, ArrayRef<OpFoldResult> ivs,
                          ArrayRef<OpFoldResult> tileSizes,
                          ArrayRef<OpFoldResult> sizeBounds,
                          bool omitPartialTileCheck) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](OpFoldResult v) { return !isZeroInteger(v); })) &&
         "expected as many ivs as non-zero sizes");

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<OpFoldResult> lbs =
      computeTileOffsets(builder, loc, ivs, tileSizes);
  SmallVector<OpFoldResult> subShapeSizes =
      computeTileSizes(builder, loc, tileSizes, sizeBounds);

  assert(static_cast<int64_t>(valuesToTile.size()) <=
             linalgOp->getNumOperands() &&
         "more value to tile than operands.");
  SmallVector<std::optional<SliceParameters>> allSliceParams;
  allSliceParams.reserve(valuesToTile.size());
  for (auto [opOperand, val] :
       llvm::zip(linalgOp->getOpOperands(), valuesToTile)) {
    Value shapedOp = val;
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for operand " << shapedOp);
    AffineMap map = linalgOp.getMatchingIndexingMap(&opOperand);
    // Use `opOperand` as is if it is not tiled and not an output tensor. Having
    // an extract/insert slice pair for all output tensors simplifies follow up
    // transformations such as padding and bufferization since the
    // extract/insert slice pairs make the accessed iteration argument
    // subdomains explicit.

    Type operandType = opOperand.get().getType();
    if (!isTiled(map, tileSizes) && !(isa<RankedTensorType>(operandType) &&
                                      linalgOp.isDpsInit(&opOperand))) {
      allSliceParams.push_back(std::nullopt);
      LLVM_DEBUG(llvm::dbgs()
                 << ": not tiled: use shape: " << operandType << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subshape...\n");

    allSliceParams.push_back(computeSliceParameters(
        builder, loc, shapedOp, tileSizes, map, lbs, sizeBounds, subShapeSizes,
        omitPartialTileCheck));
  }

  return allSliceParams;
}

SmallVector<Value> makeTiledShapes(OpBuilder &builder, Location loc,
                                   LinalgOp linalgOp, ValueRange valuesToTile,
                                   ArrayRef<OpFoldResult> ivs,
                                   ArrayRef<OpFoldResult> tileSizes,
                                   ArrayRef<OpFoldResult> sizeBounds,
                                   bool omitPartialTileCheck) {
  SmallVector<std::optional<SliceParameters>> allSliceParameter =
      computeAllSliceParameters(builder, loc, linalgOp, valuesToTile, ivs,
                                tileSizes, sizeBounds, omitPartialTileCheck);
  SmallVector<Value> tiledShapes;
  for (auto item : llvm::zip(valuesToTile, allSliceParameter)) {
    Value valueToTile = std::get<0>(item);
    std::optional<SliceParameters> sliceParams = std::get<1>(item);
    tiledShapes.push_back(
        sliceParams.has_value()
            ? materializeTiledShape(builder, loc, valueToTile, *sliceParams)
                  ->getResult(0)
            : valueToTile);
  }
  return tiledShapes;
}

void offsetIndices(OpBuilder &b, LinalgOp linalgOp,
                   ArrayRef<OpFoldResult> offsets) {
  IRRewriter rewriter(b);
  offsetIndices(rewriter, linalgOp, offsets);
}

void offsetIndices(RewriterBase &b, LinalgOp linalgOp,
                   ArrayRef<OpFoldResult> offsets) {
  if (!linalgOp.hasIndexSemantics())
    return;

  for (IndexOp indexOp : linalgOp.getBlock()->getOps<IndexOp>()) {
    if (indexOp.getDim() >= offsets.size() || !offsets[indexOp.getDim()])
      continue;
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(indexOp);
    AffineExpr index, offset;
    bindDims(b.getContext(), index, offset);
    OpFoldResult applied = makeComposedFoldedAffineApply(
        b, indexOp.getLoc(), index + offset,
        {getAsOpFoldResult(indexOp.getResult()), offsets[indexOp.getDim()]});
    Value materialized =
        getValueOrCreateConstantIndexOp(b, indexOp.getLoc(), applied);
    b.replaceUsesWithIf(indexOp, materialized, [&](OpOperand &use) {
      return use.getOwner() != materialized.getDefiningOp();
    });
  }
}

/// Get the reassociation maps to fold the result of a extract_slice (or source
/// of a insert_slice) operation with given offsets, and sizes to its
/// rank-reduced version. This is only done for the cases where the size is 1
/// and offset is 0. Strictly speaking the offset 0 is not required in general,
/// but non-zero offsets are not handled by SPIR-V backend at this point (and
/// potentially cannot be handled).
std::optional<SmallVector<ReassociationIndices>>
getReassociationMapForFoldingUnitDims(ArrayRef<OpFoldResult> mixedSizes) {
  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices curr;
  for (const auto &it : llvm::enumerate(mixedSizes)) {
    auto dim = it.index();
    auto size = it.value();
    curr.push_back(dim);
    auto attr = llvm::dyn_cast_if_present<Attribute>(size);
    if (attr && cast<IntegerAttr>(attr).getInt() == 1)
      continue;
    reassociation.emplace_back(ReassociationIndices{});
    std::swap(reassociation.back(), curr);
  }
  // When the reassociations are not empty, then fold the remaining
  // unit-dimensions into the last dimension.  If the reassociations so far is
  // empty, then leave it emtpy. This will fold everything to a rank-0 tensor.
  if (!curr.empty() && !reassociation.empty())
    reassociation.back().append(curr.begin(), curr.end());
  return reassociation;
}

} // namespace linalg
} // namespace mlir
