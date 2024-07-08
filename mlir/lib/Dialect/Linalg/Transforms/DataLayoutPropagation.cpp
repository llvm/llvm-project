//===- DataLayoutPropagation.cpp -----------------------------------------===///
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_LINALGDATALAYOUTPROPAGATION
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-data-layout-propagation"

namespace {

static bool hasGatherSemantics(linalg::GenericOp genericOp) {
  for (Operation &op : genericOp.getBody()->getOperations())
    if (isa<tensor::ExtractOp, linalg::IndexOp>(op))
      return true;
  return false;
}

// The struct contains the infomation about mapping packing information to
// the iteration domain of Linalg ops.
struct PackInfo {
  int64_t getNumTiledLoops() const { return tileToPointMapping.size(); };
  // InnerDimsPos on iteration domain, which follows the order in pack ops.
  SmallVector<int64_t> tiledDimsPos;
  // The sizes of tiling data dimensions on iteration domain.
  llvm::DenseMap<int64_t, OpFoldResult> domainDimAndTileMapping;
  // The mapping from a dimension of iteration domain to the corresponding inner
  // tiling dimension on iteration domain.
  llvm::DenseMap<int64_t, int64_t> tileToPointMapping;
  // The permutation of outer dims (on domain).
  SmallVector<int64_t> outerDimsOnDomainPerm;
};

template <typename OpTy>
static FailureOr<PackInfo>
getPackingInfoFromOperand(OpOperand *opOperand, linalg::GenericOp genericOp,
                          OpTy packOrUnPackOp) {
  static_assert(llvm::is_one_of<OpTy, tensor::PackOp, tensor::UnPackOp>::value,
                "applies to only pack or unpack operations");
  LLVM_DEBUG(
      { llvm::dbgs() << "--- Construct PackInfo From an operand ---\n"; });

  AffineMap indexingMap = genericOp.getMatchingIndexingMap(opOperand);
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iterators =
      genericOp.getIteratorTypesArray();

  PackInfo packInfo;
  int64_t origNumDims = indexingMap.getNumDims();
  SmallVector<AffineExpr> exprs(indexingMap.getResults());
  ArrayRef<int64_t> innerDimsPos = packOrUnPackOp.getInnerDimsPos();
  for (auto [index, innerDimPos, tileSize] :
       llvm::zip_equal(llvm::seq<unsigned>(0, innerDimsPos.size()),
                       innerDimsPos, packOrUnPackOp.getMixedTiles())) {
    auto expr = exprs[innerDimPos];
    if (!isa<AffineDimExpr>(expr))
      return failure();
    int64_t domainDimPos =
        cast<AffineDimExpr>(exprs[innerDimPos]).getPosition();
    if (!isParallelIterator(iterators[domainDimPos]))
      return failure();
    packInfo.tiledDimsPos.push_back(domainDimPos);
    packInfo.domainDimAndTileMapping[domainDimPos] = tileSize;
    packInfo.tileToPointMapping[domainDimPos] = origNumDims + index;
    LLVM_DEBUG({
      llvm::dbgs() << "map innerDimPos=" << innerDimPos
                   << " to iteration dimension (d" << domainDimPos << ", d"
                   << packInfo.tileToPointMapping[domainDimPos]
                   << "), which has size=("
                   << packInfo.domainDimAndTileMapping[domainDimPos] << ")\n";
    });
  }

  // Bail out if a tiled dimension is present in a map but not as an affine dim
  // expression.
  auto areAllAffineDimExpr = [&](int dim) {
    for (AffineMap map : indexingMaps) {
      if (llvm::any_of(map.getResults(), [dim](AffineExpr expr) {
            return expr.isFunctionOfDim(dim) && !isa<AffineDimExpr>(expr);
          })) {
        return false;
      }
    }
    return true;
  };
  for (int64_t i : packInfo.tiledDimsPos)
    if (!areAllAffineDimExpr(i))
      return failure();

  // Get the outer dims perm on the iteration domain. Start by identifying the
  // set of domain dims affected by the outer permutation along with the
  // permuted ordering for those dims. Then the full outer dims permutation can
  // be constructed by replacing the affected dims with the permuted result in a
  // numLoops-rank identity. e.g.
  //   outerDimsPerm = [1, 2, 0]
  //   indexingMap = (d0, d1, d2, d3, d4) -> (d1, d4, d3)
  //
  //   permutedOuterDims =        [4,    3, 1]
  //   outerDimsOnDomainPerm = [0, 4, 2, 3, 1]
  //
  // Non-affine dim expressions must not be permuted by the outer dims
  // permutation.
  SmallVector<int64_t> permutedOuterDims;
  for (auto [index, dim] : llvm::enumerate(packOrUnPackOp.getOuterDimsPerm())) {
    auto permutedExpr = indexingMap.getResult(dim);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(permutedExpr)) {
      permutedOuterDims.push_back(dimExpr.getPosition());
      continue;
    }

    // TODO: Allow propagation with transposes on non affine dim expressions,
    // e.g. d0 + d1 which implies transposing both dims simultaneously while
    // maintaining the relative position between them.
    if (static_cast<int64_t>(index) != dim)
      return failure();
  }
  if (!permutedOuterDims.empty()) {
    int64_t outerDimIndex = 0;
    llvm::DenseSet<int64_t> permutedDomainDims(permutedOuterDims.begin(),
                                               permutedOuterDims.end());
    for (int i = 0, e = indexingMap.getNumDims(); i < e; i++)
      packInfo.outerDimsOnDomainPerm.push_back(
          permutedDomainDims.contains(i) ? permutedOuterDims[outerDimIndex++]
                                         : i);
    LLVM_DEBUG({
      llvm::dbgs() << "map outer dimsDimsPerm to ";
      for (auto dim : packInfo.outerDimsOnDomainPerm)
        llvm::dbgs() << dim << " ";
      llvm::dbgs() << "\n";
    });
  }

  return packInfo;
}

static SmallVector<int64_t> computeOuterDims(ArrayRef<int64_t> perm,
                                             ArrayRef<AffineExpr> exprs) {
  // Compute `outer_dims_perm`. See example:
  // current exprs      : (d0, d1, d2, d3) -> (d2, d3)
  // perm               : [0, 3, 1, 2]
  // First map d2, d3 with their position in the array as:
  // currentPositionTileLoops: dim | pos
  //                           d2  | 0
  //                           d3  | 1
  // then scan `perm` in order and get the `outer_dims_perm`
  // to be used, here it would be [1, 0].
  assert(!perm.empty() && "expect perm not to be empty");
  assert(!exprs.empty() && "expect exprs not to be empty");
  if (exprs.size() == 1)
    return {};
  SmallVector<int64_t> outerDimsPerm;
  DenseMap<int64_t, int64_t> currentPositionTileLoops;
  for (auto [pos, expr] : llvm::enumerate(exprs)) {
    // Here we rely on the assumption that the outer dims permutation
    // when propagating currently requires that non-affine dim expressions
    // are not permuted, thus allowing the identity assignment below.
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
      currentPositionTileLoops[dimExpr.getPosition()] = pos;
    else
      currentPositionTileLoops[pos] = pos;
  }
  for (int64_t loopIdx : perm) {
    if (currentPositionTileLoops.count(loopIdx))
      outerDimsPerm.push_back(currentPositionTileLoops.lookup(loopIdx));
  }
  return outerDimsPerm;
}

/// Returns a tuple for packed operand and indexing_map with the assumptions:
///   1) The generic op is the producer of the pack op.
///   2) The generic op has only one result.
/// If the operand is a scalar or packing dimensions are all irrelevant to the
/// operand, the operand and the updated indexing map will be returned.
/// Otherwise, it returns the packed operand and the updated indexing map. E.g.,
///
///   #map0 = affine_map<(d0, d1) -> (d0, d1)>
///   #map1 = affine_map<(d0, d1) -> (d0)>
///   #map2 = affine_map<(d0, d1) -> (d1)>
///   %0 = linalg.generic {indexing_maps = [#map1, #map2, #map0],
///                        iterator_types = ["parallel", "parallel"]}
///      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
///      outs(%init : tensor<?x?xf32>) {
///    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
///      %4 = arith.addf %arg3, %arg4 : f32
///      linalg.yield %4 : f32
///  } -> tensor<?x?xf32>
///  %1 = tensor.pack %0
///    inner_dims_pos = [0, 1]
///    inner_tiles = [8, 2]
///    into %dest : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///
///  Taking the first input operand as an example, the inner tile size of d1 is
///  8. Thus, the below operation and `affine_map<(d0, d1, d2, d3)> ->
///  affine_map<(d1, d3)>` will be returned.
///
///  %pack = tensor.pack %arg0
///    inner_dims_pos = [0]
///    inner_tiles = [8]
///    into %init : tensor<?xf32> -> tensor<?x8xf32>
static std::tuple<Value, AffineMap>
getOrCreatePackedViewOfOperand(OpBuilder &b, Location loc, PackInfo packInfo,
                               GenericOp genericOp, OpOperand *opOperand) {
  int64_t numOrigLoops = genericOp.getNumLoops();
  int64_t numInnerLoops = packInfo.getNumTiledLoops();
  int64_t numLoops = numOrigLoops + numInnerLoops;
  AffineMap origIndexingMap = genericOp.getMatchingIndexingMap(opOperand);
  llvm::DenseMap<int64_t, int64_t> domainDimToOperandDim;
  SmallVector<AffineExpr> exprs(origIndexingMap.getResults());

  // If the OpOperand is a scalar or a zero-rank tensor, no need to pack.
  if (genericOp.isScalar(opOperand) || exprs.empty())
    return std::make_tuple(opOperand->get(),
                           AffineMap::get(numLoops, 0, exprs, b.getContext()));

  // Step 1. Construct the information of packing data dimensions; append inner
  // dimensions to the indexing maps for the operand.
  for (auto [index, expr] : llvm::enumerate(exprs)) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      int64_t dimPos = dimExpr.getPosition();
      domainDimToOperandDim[dimPos] = index;
      continue;
    }
  }
  SmallVector<int64_t> innerDimsPos;
  SmallVector<OpFoldResult> innerTileSizes;
  for (auto dimPos : packInfo.tiledDimsPos) {
    if (!domainDimToOperandDim.count(dimPos))
      continue;
    int64_t index = domainDimToOperandDim[dimPos];
    innerTileSizes.push_back(packInfo.domainDimAndTileMapping[dimPos]);
    innerDimsPos.push_back(index);
    exprs.push_back(b.getAffineDimExpr(packInfo.tileToPointMapping[dimPos]));
  }

  // Step 2. Handle outer dim permutations.
  SmallVector<int64_t> outerDimsPerm;
  if (!packInfo.outerDimsOnDomainPerm.empty()) {
    outerDimsPerm = computeOuterDims(packInfo.outerDimsOnDomainPerm, exprs);

    // Step 2.1: Fold transpose into the linalg.generic.
    SmallVector<int64_t> inversedOuterPerm =
        invertPermutationVector(packInfo.outerDimsOnDomainPerm);
    for (auto i : llvm::seq<unsigned>(0, origIndexingMap.getNumResults())) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(exprs[i])) {
        int64_t dimPos = dimExpr.getPosition();
        exprs[i] = b.getAffineDimExpr(inversedOuterPerm[dimPos]);
        continue;
      }
      assert(isa<AffineConstantExpr>(exprs[i]) &&
             "Attempted to permute non-constant and non-affine dim expression");
    }
    // Step 2.2: Undo the transposition on `exprs` and propagate the
    // transposition on the pack using outerDimsPerm.
    if (!outerDimsPerm.empty()) {
      SmallVector<AffineExpr> auxVec = exprs;
      for (const auto &en : enumerate(outerDimsPerm))
        auxVec[en.index()] = exprs[en.value()];
      exprs = auxVec;
    }
  }
  auto indexingMap = AffineMap::get(numLoops, 0, exprs, b.getContext());

  // The operand does not have dimensions that relates to pack op.
  if (innerDimsPos.empty() && outerDimsPerm.empty())
    return std::make_tuple(opOperand->get(), indexingMap);

  auto empty = tensor::PackOp::createDestinationTensor(
      b, loc, opOperand->get(), innerTileSizes, innerDimsPos, outerDimsPerm);
  auto packedOperand = b.create<tensor::PackOp>(
      loc, opOperand->get(), empty, innerDimsPos, innerTileSizes,
      /*padding=*/std::nullopt, outerDimsPerm);
  return std::make_tuple(packedOperand, indexingMap);
}

/// Pack a genericOp and return it.
static GenericOp packGenericOp(RewriterBase &rewriter, GenericOp genericOp,
                               Value dest, AffineMap packedOutIndexingMap,
                               const PackInfo &packInfo) {
  Location loc = genericOp.getLoc();
  SmallVector<Value> inputOperands;
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand *inputOperand : genericOp.getDpsInputOperands()) {
    auto [packedOperand, packedIndexingMap] = getOrCreatePackedViewOfOperand(
        rewriter, loc, packInfo, genericOp, inputOperand);
    inputOperands.push_back(packedOperand);
    indexingMaps.push_back(packedIndexingMap);
  }

  int64_t numInnerLoops = packInfo.getNumTiledLoops();
  SmallVector<utils::IteratorType> iterTypes =
      genericOp.getIteratorTypesArray();
  iterTypes.append(numInnerLoops, utils::IteratorType::parallel);

  indexingMaps.push_back(packedOutIndexingMap);

  auto newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, dest.getType(), inputOperands, dest, indexingMaps, iterTypes,
      /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));
  rewriter.cloneRegionBefore(genericOp.getRegion(), newGenericOp.getRegion(),
                             newGenericOp.getRegion().begin());
  return newGenericOp;
}

/// Bubbles up tensor.pack op through a producer generic op. This
/// swap pack(generic) to generic(pack). The new generic op works on packed
/// domain; pack ops are created for input and output operands. E.g.,
///
///     #map0 = affine_map<(d0, d1) -> (d0, d1)>
///     %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///     %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
///     %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
///     %3 = linalg.generic {indexing_maps = [#map0, #map0],
///                          iterator_types = ["parallel", "parallel"]}
///         ins(%arg0 : tensor<?x?xf32>)
///         outs(%2 : tensor<?x?xf32>) {
///       ^bb0(%arg3: f32, %arg4: f32):
///         %4 = arith.addf %arg3, %arg3 : f32
///         linalg.yield %4 : f32
///     } -> tensor<?x?xf32>
///     %4 = tensor.pack %3
///       inner_dims_pos = [0, 1]
///       inner_tiles = [8, 2]
///       into %dest : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///
/// will be converted to
///
///     #map = affine_map<()[s0] -> (s0 ceildiv 8)>
///     #map1 = affine_map<()[s0] -> (s0 ceildiv 2)>
///     #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///     %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///     %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
///     %0 = affine.apply #map()[%dim]
///     %1 = affine.apply #map1()[%dim_0]
///     %2 = tensor.empty(%0, %1) : tensor<?x?x8x2xf32>
///     %pack = tensor.pack %arg0
///       inner_dims_pos = [0, 1]
///       inner_tiles = [8, 2]
///       into %2 : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///     %3 = linalg.generic {indexing_maps = [#map2, #map2],
///       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
///       ins(%pack : tensor<?x?x8x2xf32>)
///       outs(%arg1 : tensor<?x?x8x2xf32>) {
///     ^bb0(%in: f32, %out: f32):
///       %4 = arith.addf %in, %in : f32
///       linalg.yield %4 : f32
///     } -> tensor<?x?x8x2xf32>
static FailureOr<GenericOp>
bubbleUpPackOpThroughGenericOp(RewriterBase &rewriter, tensor::PackOp packOp,
                               const ControlPropagationFn &controlFn) {
  auto genericOp = packOp.getSource().getDefiningOp<GenericOp>();
  if (!genericOp)
    return failure();

  // User controlled propagation function.
  if (!controlFn(genericOp))
    return failure();

  // TODO: Enable propagation in the presence of linalg.index and
  // tensor.extract, likely as a separate pattern as the pack information and
  // propagation decision needs to be inferred from the region of the generic.
  if (hasGatherSemantics(genericOp))
    return failure();

  // TODO: Relax the restriction. We are able to bubble up the pack op through
  // multi-result generic op. It just needs more work.
  if (genericOp.getNumResults() != 1)
    return failure();

  // Bail-out if the result of the generic has multiple uses, as bubbling up
  // creates recomputation if the generic has multiple users.
  // TODO: Enable the case where every use is an identical pack op as no
  // recomputation is needed in that case.
  if (!genericOp->getResult(0).hasOneUse())
    return failure();

  // We want to move the pack not the generic.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(genericOp);

  // We need to handle two cases:
  // 1) The tensor.pack destination is a tensor.empty. If this is the case, we
  // create a new tensor.empty to avoid breaking dominance, as we are moving the
  // tensor.pack above the linalg.generic.
  // 2) The destination is not a tensor.empty. In this case we can replace only
  // if the destination of the tensor.pack dominates the linalg.generic.
  Value packOpDest = packOp.getDest();
  if (!packOpDest.hasOneUse())
    return failure();
  if (auto emptyOp = packOpDest.getDefiningOp<tensor::EmptyOp>()) {
    packOpDest = rewriter.create<tensor::EmptyOp>(
        genericOp->getLoc(), emptyOp.getMixedSizes(),
        emptyOp.getType().getElementType());
  } else {
    DominanceInfo dom(genericOp);
    if (!dom.properlyDominates(packOpDest, genericOp))
      return failure();
  }

  // TODO: Add an option for allowing padding values. It could introduce
  // undefined behavior if we unconditionally propagate pack op through all
  // the ops. E.g., if the padding value is zero and there are division ops in
  // a generic op. Some values of padding area could be NaN (0/0).
  if (packOp.getPaddingValue())
    return failure();

  OpOperand *opOperand = genericOp.getDpsInitOperand(0);
  auto packInfo = getPackingInfoFromOperand(opOperand, genericOp, packOp);
  if (failed(packInfo))
    return failure();

  // Rebuild the indexing map for the corresponding init operand.
  auto [packedOutOperand, packedOutIndexingMap] =
      getOrCreatePackedViewOfOperand(rewriter, genericOp.getLoc(), *packInfo,
                                     genericOp, opOperand);

  // If the dps init operand of the generic is a tensor.empty forward the pack
  // op destination.
  Value dest = packedOutOperand;
  if (auto initTensor = genericOp.getDpsInitOperand(0)
                            ->get()
                            .getDefiningOp<tensor::EmptyOp>()) {
    dest = packOpDest;
  }
  return packGenericOp(rewriter, genericOp, dest, packedOutIndexingMap,
                       *packInfo);
}

/// Wrapper pattern that applies bubbleUpPackOpThroughGenericOp method.
struct BubbleUpPackOpThroughGenericOpPattern
    : public OpRewritePattern<tensor::PackOp> {
public:
  BubbleUpPackOpThroughGenericOpPattern(MLIRContext *context,
                                        ControlPropagationFn fun)
      : OpRewritePattern<tensor::PackOp>(context), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto genericOp =
        bubbleUpPackOpThroughGenericOp(rewriter, packOp, controlFn);
    if (failed(genericOp))
      return failure();
    rewriter.replaceOp(packOp, genericOp->getResults());
    return success();
  }

private:
  ControlPropagationFn controlFn;
};

/// Propagate a tensor.pack operation up through a tensor.pad. The idea is to
/// add as many zero padding dimensions in `high` and `low` based on the number
/// of point loops.
class BubbleUpPackThroughPadOp final : public OpRewritePattern<tensor::PackOp> {
public:
  BubbleUpPackThroughPadOp(MLIRContext *context, ControlPropagationFn fun)
      : OpRewritePattern<tensor::PackOp>(context), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto padOp = packOp.getSource().getDefiningOp<tensor::PadOp>();
    if (!padOp)
      return failure();

    // User controlled propagation function.
    if (!controlFn(padOp))
      return failure();

    // TODO: Enable padding when the padding values are the same.
    if (packOp.getPaddingValue())
      return failure();

    // Fail for non-constant padding values. The body of the pad could
    // depend on the padding indices and/or properties of the padded
    // tensor so for now we fail.
    // TODO: Support non-constant padding values.
    Value paddingVal = padOp.getConstantPaddingValue();
    if (!paddingVal)
      return failure();

    if (!packOp.getDest().getDefiningOp<tensor::EmptyOp>())
      return failure();

    ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();

    // Bail out if one of the padded dimension is a tiled one.
    llvm::SmallBitVector paddedDims = padOp.getPaddedDims();
    llvm::SmallBitVector innerDims(paddedDims.size());
    for (int64_t dim : innerDimsPos)
      innerDims.flip(dim);
    if (paddedDims.anyCommon(innerDims))
      return failure();

    Location loc = padOp->getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(padOp);

    ArrayRef<int64_t> outerDimsPerm = packOp.getOuterDimsPerm();
    SmallVector<OpFoldResult> mixedTiles = packOp.getMixedTiles();
    auto empty = tensor::PackOp::createDestinationTensor(
        rewriter, loc, padOp.getSource(), mixedTiles, innerDimsPos,
        outerDimsPerm);
    auto sourcePack = rewriter.create<tensor::PackOp>(
        loc, padOp.getSource(), empty, innerDimsPos, mixedTiles,
        /*padding=*/std::nullopt, outerDimsPerm);

    // If we have `outer_dims_perms` we need to adjust the padded dimensions.
    SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highPad = padOp.getMixedHighPad();
    if (!outerDimsPerm.empty()) {
      applyPermutationToVector<OpFoldResult>(lowPad, outerDimsPerm);
      applyPermutationToVector<OpFoldResult>(highPad, outerDimsPerm);
    }
    // The tiled dimensions were verified to be unpadded above, so here we
    // just append 0 for the inner tile dimensions.
    size_t pointLoopsSize = innerDimsPos.size();
    lowPad.append(pointLoopsSize, rewriter.getIndexAttr(0));
    highPad.append(pointLoopsSize, rewriter.getIndexAttr(0));

    auto newPadOp = rewriter.create<tensor::PadOp>(
        loc, /*result=*/Type(), sourcePack, lowPad, highPad, paddingVal,
        padOp.getNofold());

    // If the pad has more than one user, create an unpack on the new pad to
    // replace the other uses.
    if (!padOp->hasOneUse()) {
      auto unpackEmpty = tensor::UnPackOp::createDestinationTensor(
          rewriter, loc, newPadOp, mixedTiles, innerDimsPos, outerDimsPerm);
      Value unpackedPad = rewriter.create<tensor::UnPackOp>(
          loc, newPadOp, unpackEmpty, innerDimsPos, mixedTiles, outerDimsPerm);
      rewriter.replaceAllUsesExcept(padOp, unpackedPad, sourcePack);
    }

    // Replace the pack with the new pad.
    rewriter.replaceOp(packOp, newPadOp.getResult());

    return success();
  }

private:
  ControlPropagationFn controlFn;
};

/// Project dimsPos to the inner-most non-unit dim pos with reassocIndices.
///
/// For example, given dimsPos [0, 2], reassocIndices [[0, 1], [2, 3]], and
/// targetShape [16, 16, 32, 1], it returns [1, 2]. Because for pos 0, the
/// inner-most projected dim in pos [0, 1] is 1. And for pos 2, the inner-most
/// non-unit projected dims in pos [2, 3] is 2.
///
/// If all candidates in a reassociation are unit dims, it chooses the
/// inner-most dim pos.
static SmallVector<int64_t>
projectToInnerMostNonUnitDimsPos(ArrayRef<int64_t> dimsPos,
                                 ArrayRef<ReassociationIndices> reassocIndices,
                                 ArrayRef<int64_t> targetShape) {
  SmallVector<int64_t> projectedDimsPos;
  for (auto pos : dimsPos) {
    // In the case all dims are unit, this will return the inner-most one.
    int64_t projectedPos = reassocIndices[pos].back();
    for (auto i : llvm::reverse(reassocIndices[pos])) {
      int64_t dim = targetShape[i];
      if (dim > 1 || ShapedType::isDynamic(dim)) {
        projectedPos = i;
        break;
      }
    }
    projectedDimsPos.push_back(projectedPos);
  }
  return projectedDimsPos;
}

/// Check if all dims in dimsPos are divisible by the corresponding tile sizes.
static bool isDimsDivisibleByTileSizes(ArrayRef<int64_t> dimsPos,
                                       ArrayRef<int64_t> shape,
                                       ArrayRef<int64_t> tileSizes) {
  for (auto [pos, tileSize] : llvm::zip_equal(dimsPos, tileSizes)) {
    int64_t dim = shape[pos];
    if (ShapedType::isDynamic(dim) || (dim % tileSize) != 0)
      return false;
  }
  return true;
}

/// Permutate the reassociation indices and reindex them in the sequence order.
/// Returns the next dim pos in the sequence.
///
/// For example, given reassocIndices [[0, 1], [2]] and permutation [1, 0], it
/// applies the permutation to get [[2], [0, 1]] and reindexes the indices into
/// [[0], [1, 2]].
static int64_t applyPermutationAndReindexReassoc(
    SmallVector<ReassociationIndices> &reassocIndices,
    ArrayRef<int64_t> permutation) {
  if (!permutation.empty())
    applyPermutationToVector<ReassociationIndices>(reassocIndices, permutation);
  int64_t nextPos = 0;
  for (ReassociationIndices &indices : reassocIndices) {
    for (auto &index : indices) {
      index = nextPos;
      nextPos += 1;
    }
  }
  return nextPos;
}

/// Bubble up pack op through collapse shape op when the packed dims can be
/// projected to the dims before collapsing. This is possible when the inner
/// tile sizes can divide the projected dims.
///
/// For example:
///
/// %collapsed = tensor.collapse_shape %in [[0, 1], 2]
///     : tensor<?x16x4xf32> into tensor<?x4xf32>
/// %pack = tensor.pack %collapsed outer_dims_perm = [0, 1]
///     inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %empty
///     : tensor<?x4xf32> -> tensor<?x4x8x1xf32>
///
/// can be transformed into:
///
/// %pack = tensor.pack %in outer_dims_perm = [1, 2]
///     inner_dims_pos = [1, 2] inner_tiles = [8, 1] into %empty
///     : tensor<?x16x4xf32> -> tensor<?x2x4x8x1xf32>
/// %collapsed = tensor.collapse_shape %pack [[0, 1], 2, 3, 4]
///     : tensor<?x2x4x8x1xf32> into tensor<?x4x8x1>
static LogicalResult
bubbleUpPackOpThroughCollapseShape(tensor::CollapseShapeOp collapseOp,
                                   tensor::PackOp packOp,
                                   PatternRewriter &rewriter) {
  SmallVector<int64_t> innerTileSizes = packOp.getStaticTiles();
  ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();
  ArrayRef<int64_t> outerDimsPerm = packOp.getOuterDimsPerm();

  ArrayRef<int64_t> srcShape = collapseOp.getSrcType().getShape();
  SmallVector<ReassociationIndices> reassocIndices =
      collapseOp.getReassociationIndices();
  // Project inner tile pos to the dim pos before collapsing. For example, if
  // dims [x, y] is collapsed into [z], packing on dim z can be projected back
  // to pack on dim y.
  //
  // Project to inner-most non-unit dims to increase the chance that they can be
  // divided by the inner tile sizes. This is correct because for [..., x, 1],
  // packing on dim 1 is equivalent to packing on dim x.
  SmallVector<int64_t> projectedInnerDimsPos =
      projectToInnerMostNonUnitDimsPos(innerDimsPos, reassocIndices, srcShape);

  if (!isDimsDivisibleByTileSizes(projectedInnerDimsPos, srcShape,
                                  innerTileSizes)) {
    return failure();
  }
  // Expand the outer dims permutation with the associated source dims for the
  // new permutation after bubbling. This is because moving a collapsed dim is
  // equivalent to moving the associated source dims together.
  SmallVector<int64_t> newOuterDimsPerm;
  for (auto outerPos : outerDimsPerm) {
    newOuterDimsPerm.insert(newOuterDimsPerm.end(),
                            reassocIndices[outerPos].begin(),
                            reassocIndices[outerPos].end());
  }

  auto emptyOp = tensor::PackOp::createDestinationTensor(
      rewriter, packOp.getLoc(), collapseOp.getSrc(), packOp.getMixedTiles(),
      projectedInnerDimsPos, newOuterDimsPerm);
  auto newPackOp = rewriter.create<tensor::PackOp>(
      packOp.getLoc(), collapseOp.getSrc(), emptyOp, projectedInnerDimsPos,
      packOp.getMixedTiles(), packOp.getPaddingValue(), newOuterDimsPerm);

  SmallVector<ReassociationIndices> newReassocIndices = reassocIndices;
  // First apply the permutation on the reassociations of the outer dims.
  // For example given the permutation [1, 0], the reassociations [[0, 1], [2]]
  // -> [[0], [1, 2]]
  int64_t nextPos =
      applyPermutationAndReindexReassoc(newReassocIndices, outerDimsPerm);
  // Then add direct mapping for the inner tile dims.
  for (size_t i = 0; i < innerDimsPos.size(); ++i) {
    newReassocIndices.push_back({nextPos});
    nextPos += 1;
  }

  auto newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
      collapseOp.getLoc(), packOp.getType(), newPackOp, newReassocIndices);
  rewriter.replaceOp(packOp, newCollapseOp);

  return success();
}

/// Project dimsPos to their collapsed positions in the reassocIndices.
///
/// For example, given dimsPos [0, 1, 2, 4], and matching reassocIndices
/// [[0], [1, 2], [3], [4]], it returns [0, 1, 1, 3]. Because for pos 0,
/// the reassoc dim [0] is 0. For pos 1 and 2, the reassoc dim in pos
/// [1, 2] is 1. And for pos 4, the reassoc dim [4] is 3.
static SmallVector<int64_t>
projectDimsPosIntoReassocPos(ArrayRef<int64_t> dimsPos,
                             ArrayRef<ReassociationIndices> reassocIndices) {
  SmallVector<int64_t> projectedPos;

  // Map each dimension to the position of corresponding reassociation index.
  for (auto pos : dimsPos) {
    for (auto [idx, indices] : llvm::enumerate(reassocIndices)) {
      // If the dimension is present in the current indices group, the group
      // position within the reassociation map is the desired projected
      // dimension position.
      if (llvm::any_of(indices,
                       [&](int64_t expandDim) { return expandDim == pos; })) {
        projectedPos.push_back(idx);
        break;
      }
    }
  }
  assert(projectedPos.size() == dimsPos.size() && "Invalid dim pos projection");

  return projectedPos;
}

/// Bubble up pack op through expand shape op.
///
/// For example:
///
/// %expand = tensor.expand_shape %in [[0], [1, 2]]
///     : tensor<?x64xf32> into tensor<?x4x16xf32>
/// %pack = tensor.pack %expand outer_dims_perm = [0, 1]
///     inner_dims_pos = [2] inner_tiles = [8] into %empty
///     : tensor<?x4x16xf32> -> tensor<?x4x2x8xf32>
///
/// can be transformed into:
///
/// %pack = tensor.pack %in outer_dims_perm = [1, 2]
///     inner_dims_pos = [1] inner_tiles = [8] into %empty
///     : tensor<?x64xf32> -> tensor<?x8x8xf32>
/// %expand = tensor.expand_shape %pack [[0], [1, 2], [3]]
///     : tensor<?x8x8xf32> into tensor<?x4x2x8xf32>
static LogicalResult
bubbleUpPackOpThroughExpandShape(tensor::ExpandShapeOp expandOp,
                                 tensor::PackOp packOp,
                                 PatternRewriter &rewriter) {
  // Outer dimensions permutation is not supported currently.
  // TODO: Handle outer_dims_perm variants.
  ArrayRef<int64_t> outerDimsPerm = packOp.getOuterDimsPerm();
  if (!outerDimsPerm.empty() && !isIdentityPermutation(outerDimsPerm)) {
    return rewriter.notifyMatchFailure(packOp,
                                       "non-identity outer dims perm NYI");
  }

  // Validate dimensions' relations between shape expansion and packing.
  SmallVector<ReassociationIndices, 4> reassoc =
      expandOp.getReassociationIndices();
  ArrayRef<int64_t> packInnerDims = packOp.getInnerDimsPos();
  llvm::SetVector<int64_t> packDimsPos(packInnerDims.begin(),
                                       packInnerDims.end());

  for (auto [idx, indices] : llvm::enumerate(reassoc)) {
    // For each expand_shape reassociation, figure out which dimensions get
    // packed if any.
    llvm::SetVector<int64_t> expandDimPos(indices.begin(), indices.end());
    llvm::SetVector<int64_t> packedDims =
        llvm::set_intersection(packDimsPos, expandDimPos);

    // The expanded dimension is not packed so, it does not affect moving pack
    // before shape expansion - simply continue.
    if (packedDims.empty())
      continue;
    // Shape expansion cannot be propagated when multiple expanded dimension are
    // packed - in this case operation reordering would affect final element
    // positions and/or shapes can no longer be projected.
    if (packedDims.size() != 1)
      return rewriter.notifyMatchFailure(
          packOp, "only one of the expanded dimensions can be packed");
    // Only the inner-most expanded dimension should be packed. Otherwise,
    // elements order will be affected after operation reordering.
    if (packedDims.front() != indices.back())
      return rewriter.notifyMatchFailure(
          packOp, "can only pack the inner-most expanded dimension");
  }

  // Project pack.inner_dims_pos to positions before shape expansion.
  SmallVector<int64_t> projectedInnerDimsPos =
      projectDimsPosIntoReassocPos(packInnerDims, reassoc);

  // Project the shape expansion to new packed shape.
  // The pack.outer_dims_perm is restricted to identity so, the permutation can
  // be omitted for simplicity.
  // TODO: Account for outer dimensions permutation.
  //
  // If reassociation is not possible, then reordering cannot happen.
  // This can be caused by pack padding affecting previously expanded
  // dimensions or packing extending dimensions.
  RankedTensorType newPackType = tensor::PackOp::inferPackedType(
      expandOp.getSrcType(), packOp.getStaticInnerTiles(),
      projectedInnerDimsPos, /*outerDimsPerm=*/SmallVector<int64_t>{});
  auto reassocExpand =
      getReassociationIndicesForReshape(newPackType, packOp.getDestType());
  if (!reassocExpand)
    return rewriter.notifyMatchFailure(
        packOp, "could not reassociate dims after bubbling up");

  Value destTensor = tensor::PackOp::createDestinationTensor(
      rewriter, packOp.getLoc(), expandOp.getSrc(), packOp.getMixedTiles(),
      projectedInnerDimsPos, /*outerDimsPerm=*/SmallVector<int64_t>{});
  Value packedVal = rewriter.create<tensor::PackOp>(
      packOp.getLoc(), expandOp.getSrc(), destTensor, projectedInnerDimsPos,
      packOp.getMixedTiles(), packOp.getPaddingValue(),
      /*outerDimsPerm=*/SmallVector<int64_t>{});

  Value newExpandOp = rewriter.create<tensor::ExpandShapeOp>(
      packOp.getLoc(), packOp.getDestType(), packedVal, *reassocExpand);
  rewriter.replaceOp(packOp, newExpandOp);

  return success();
}

class BubbleUpPackOpThroughReshapeOp final
    : public OpRewritePattern<tensor::PackOp> {
public:
  BubbleUpPackOpThroughReshapeOp(MLIRContext *context, ControlPropagationFn fun)
      : OpRewritePattern<tensor::PackOp>(context), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    Operation *srcOp = packOp.getSource().getDefiningOp();
    // Currently only support when the pack op is the only user.
    if (!srcOp || !(srcOp->getNumResults() == 1) ||
        !srcOp->getResult(0).hasOneUse()) {
      return failure();
    }
    // Currently only support static inner tile sizes.
    if (llvm::any_of(packOp.getStaticTiles(), [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return failure();
    }

    // User controlled propagation function.
    if (!controlFn(srcOp))
      return failure();

    return TypeSwitch<Operation *, LogicalResult>(srcOp)
        .Case([&](tensor::CollapseShapeOp op) {
          return bubbleUpPackOpThroughCollapseShape(op, packOp, rewriter);
        })
        .Case([&](tensor::ExpandShapeOp op) {
          return bubbleUpPackOpThroughExpandShape(op, packOp, rewriter);
        })
        .Default([](Operation *) { return failure(); });
  }

private:
  ControlPropagationFn controlFn;
};

/// Push down unpack op through expand shape op when the packed dims can be
/// projected to the dims after expanding. This is possible when the inner tile
/// sizes can divide the projected dims.
///
/// For example:
///
/// %unpack = tensor.unpack %in outer_dims_perm = [0, 1]
///     inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %empty
///     : tensor<?x32x8x8xf32> -> tensor<?x256xf32>
/// %expanded = tensor.expand_shape %unpack [[0, 1], [2]]
///     : tensor<?x256xf32> into tensor<?x256x256xf32>
///
/// can be transformed into:
///
/// %expanded = tensor.expand_shape %ain [[0, 1], [2], [3], [4]]
///     : tensor<?x32x8x8xf32> into tensor<?x32x32x8x8xf32>
/// %unpack = tensor.unpack %expanded outer_dims_perm = [0, 1, 2]
///     inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %empty
///     : tensor<?x32x32x8x8xf32> -> tensor<?x256x256xf32>
static LogicalResult
pushDownUnPackOpThroughExpandShape(tensor::UnPackOp unPackOp,
                                   tensor::ExpandShapeOp expandOp,
                                   PatternRewriter &rewriter) {
  SmallVector<int64_t> innerTileSizes = unPackOp.getStaticTiles();
  ArrayRef<int64_t> innerDimsPos = unPackOp.getInnerDimsPos();
  ArrayRef<int64_t> outerDimsPerm = unPackOp.getOuterDimsPerm();

  auto expandTy = dyn_cast<RankedTensorType>(expandOp.getType());
  if (!expandTy)
    return failure();
  ArrayRef<int64_t> dstShape = expandTy.getShape();
  SmallVector<ReassociationIndices> reassocIndices =
      expandOp.getReassociationIndices();
  // Project inner tile pos to the dim pos after expanding. For example, if dims
  // [z] is expanded into [x, y], unpacking on dim z can be projected to unpack
  // on dim y.
  //
  // Project to inner-most non-unit dims to increase the chance that they can be
  // divided by the inner tile sizes. This is correct because for [..., x, 1],
  // unpacking on dim 1 is equivalent to unpacking on dim x.
  SmallVector<int64_t> projectedInnerDimsPos =
      projectToInnerMostNonUnitDimsPos(innerDimsPos, reassocIndices, dstShape);

  if (!isDimsDivisibleByTileSizes(projectedInnerDimsPos, dstShape,
                                  innerTileSizes)) {
    return failure();
  }
  // Expand the outer dims permutation with the associated expanded dims for the
  // new permutation after pushing. This is because moving a source dim is
  // equivalent to moving the associated expanded dims together.
  SmallVector<int64_t> newOuterDimsPerm;
  for (auto outerPos : outerDimsPerm) {
    newOuterDimsPerm.insert(newOuterDimsPerm.end(),
                            reassocIndices[outerPos].begin(),
                            reassocIndices[outerPos].end());
  }

  SmallVector<ReassociationIndices> newReassocIndices = reassocIndices;
  // First apply the permutation on the reassociations of the outer dims.
  // For example given the permutation [1, 0], the reassociations [[0, 1], [2]]
  // -> [[0], [1, 2]]
  int64_t nextPos =
      applyPermutationAndReindexReassoc(newReassocIndices, outerDimsPerm);
  // Then add direct mapping for the inner tile dims.
  for (size_t i = 0; i < innerDimsPos.size(); ++i) {
    newReassocIndices.push_back({nextPos});
    nextPos += 1;
  }

  RankedTensorType newExpandType = tensor::PackOp::inferPackedType(
      expandTy, innerTileSizes, projectedInnerDimsPos, newOuterDimsPerm);
  auto newExpandOp = rewriter.create<tensor::ExpandShapeOp>(
      expandOp.getLoc(), newExpandType, unPackOp.getSource(),
      newReassocIndices);

  auto emptyOp = tensor::UnPackOp::createDestinationTensor(
      rewriter, unPackOp.getLoc(), newExpandOp, unPackOp.getMixedTiles(),
      projectedInnerDimsPos, newOuterDimsPerm);
  auto newUnPackOp = rewriter.create<tensor::UnPackOp>(
      unPackOp.getLoc(), newExpandOp.getResult(), emptyOp,
      projectedInnerDimsPos, unPackOp.getMixedTiles(), newOuterDimsPerm);
  rewriter.replaceOp(expandOp, newUnPackOp);

  return success();
}

class PushDownUnPackOpThroughReshapeOp final
    : public OpRewritePattern<tensor::UnPackOp> {
public:
  PushDownUnPackOpThroughReshapeOp(MLIRContext *context,
                                   ControlPropagationFn fun)
      : OpRewritePattern<tensor::UnPackOp>(context), controlFn(std::move(fun)) {
  }

  LogicalResult matchAndRewrite(tensor::UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    Value result = unPackOp.getResult();
    // Currently only support unpack op with the single user.
    if (!result.hasOneUse()) {
      return failure();
    }
    // Currently only support static inner tile sizes.
    if (llvm::any_of(unPackOp.getStaticTiles(), [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return failure();
    }

    Operation *consumerOp = *result.user_begin();
    // User controlled propagation function.
    if (!controlFn(consumerOp))
      return failure();

    return TypeSwitch<Operation *, LogicalResult>(consumerOp)
        .Case([&](tensor::ExpandShapeOp op) {
          return pushDownUnPackOpThroughExpandShape(unPackOp, op, rewriter);
        })
        .Default([](Operation *) { return failure(); });
  }

private:
  ControlPropagationFn controlFn;
};

// TODO: Relax this restriction. We should unpack a generic op also
// in the presence of multiple unpack ops as producers.
/// Return the unpacked operand, if present, for the current generic op.
static FailureOr<OpOperand *> getUnPackedOperand(GenericOp genericOp) {
  OpOperand *unPackedOperand = nullptr;
  for (OpOperand &operand : genericOp->getOpOperands()) {
    auto unPackOp = operand.get().getDefiningOp<tensor::UnPackOp>();
    if (!unPackOp)
      continue;
    if (unPackedOperand)
      return failure();
    unPackedOperand = &operand;
  }
  if (!unPackedOperand)
    return failure();
  return unPackedOperand;
}

/// Push down a tensor.unpack op through a generic op.
/// The new generic op works on packed domain; pack ops are created for input
/// and output operands. A tensor.unpack op is inserted right after the packed
/// generic. E.g.
///
/// #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///
/// %arg0 = tensor<12x2x56x56x32xf32> // packed arg.
///
/// %0 = tensor.empty() : tensor<12x56x56x64xf32>
/// %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2]
///                          inner_dims_pos = [3] inner_tiles = [32] into %0
/// %2 = linalg.generic {indexing_maps = [#map],
///      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
///      outs(%1 : tensor<12x56x56x64xf32>) {
///      ^bb0(%out : f32):
///         linalg.yield %out : f32
///      } -> tensor<12x56x56x64xf32>
///
/// will be converted to
///
/// #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
///
/// %0 = tensor.empty() : tensor<12x56x56x64xf32>
/// %1 = linalg.generic {indexing_maps = [#map],
///      iterator_types = ["parallel", "parallel", "parallel",
///                        "parallel", "parallel"]}
///      outs(%arg0 : tensor<12x2x56x56x32xf32>) {
///      ^bb0(%out : f32):
///         linalg.yield %out : f32
///      } -> tensor<12x2x56x56x32xf32>
/// %2 = tensor.unpack %1 outer_dims_perm = [0, 3, 1, 2]
///                       inner_dims_pos = [3] inner_tiles = [32] into %0
///
static FailureOr<std::tuple<GenericOp, Value>>
pushDownUnPackOpThroughGenericOp(RewriterBase &rewriter, GenericOp genericOp) {
  if (genericOp.getNumResults() != 1)
    return failure();

  if (hasGatherSemantics(genericOp))
    return failure();

  // Collect the unPacked operand, if present.
  auto maybeUnPackedOperand = getUnPackedOperand(genericOp);
  if (failed(maybeUnPackedOperand))
    return failure();
  OpOperand *unPackedOperand = *(maybeUnPackedOperand);

  // Extract packing information.
  tensor::UnPackOp producerUnPackOp =
      unPackedOperand->get().getDefiningOp<tensor::UnPackOp>();
  assert(producerUnPackOp && "expect a valid UnPackOp");
  auto packInfo =
      getPackingInfoFromOperand(unPackedOperand, genericOp, producerUnPackOp);
  if (failed(packInfo))
    return failure();

  // Rebuild the indexing map for the corresponding init operand.
  auto [packedOutOperand, packedOutIndexingMap] =
      getOrCreatePackedViewOfOperand(rewriter, genericOp.getLoc(), *packInfo,
                                     genericOp, genericOp.getDpsInitOperand(0));
  auto destPack = packedOutOperand.getDefiningOp<tensor::PackOp>();

  // If the dps init operand of the generic is a tensor.empty, do not pack it
  // and forward the new tensor.empty as a destination.
  Value dest = packedOutOperand;
  if (auto initTensor = genericOp.getDpsInitOperand(0)
                            ->get()
                            .getDefiningOp<tensor::EmptyOp>()) {
    if (destPack)
      dest = destPack.getDest();
  }

  // Pack the genericOp.
  GenericOp newGenericOp =
      packGenericOp(rewriter, genericOp, dest, packedOutIndexingMap, *packInfo);
  Value newResult =
      newGenericOp.getTiedOpResult(newGenericOp.getDpsInitOperand(0));

  // If the output is unaffected, no need to unpack.
  if (!destPack)
    return std::make_tuple(newGenericOp, newResult);

  auto mixedTiles = destPack.getMixedTiles();
  auto innerDimsPos = destPack.getInnerDimsPos();
  auto outerDimsPerm = destPack.getOuterDimsPerm();

  // If the output type for the generic differs from the source
  // unpack op, we need to create a new destination tensor. In the
  // dynamic case we always need a new destination.
  auto loc = genericOp.getLoc();
  Value unPackDest = producerUnPackOp.getDest();
  auto genericOutType =
      cast<RankedTensorType>(genericOp.getDpsInitOperand(0)->get().getType());
  if (producerUnPackOp.getDestType() != genericOutType ||
      !genericOutType.hasStaticShape()) {
    unPackDest = tensor::UnPackOp::createDestinationTensor(
        rewriter, loc, newResult, mixedTiles, innerDimsPos, outerDimsPerm);
  }

  // Insert an unPackOp right after the packed generic.
  Value unPackOpRes =
      rewriter
          .create<tensor::UnPackOp>(loc, newResult, unPackDest, innerDimsPos,
                                    mixedTiles, outerDimsPerm)
          .getResult();

  return std::make_tuple(newGenericOp, unPackOpRes);
}

// Wrapper pattern that applies pushDownUnPackOpThroughGenericOp method.
struct PushDownUnPackOpThroughGenericOp : public OpRewritePattern<GenericOp> {
public:
  PushDownUnPackOpThroughGenericOp(MLIRContext *context,
                                   ControlPropagationFn fun)
      : OpRewritePattern<GenericOp>(context), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!controlFn(genericOp))
      return failure();

    auto genericAndRepl = pushDownUnPackOpThroughGenericOp(rewriter, genericOp);
    if (failed(genericAndRepl))
      return failure();
    rewriter.replaceOp(genericOp, std::get<1>(*genericAndRepl));
    return success();
  }

private:
  ControlPropagationFn controlFn;
};

/// Propagate a tensor.unpack operation through a tensor.pad. The idea is to
/// add as many zero padding dimensions in `high` and `low` based on the number
/// of point loops.
struct PushDownUnPackThroughPadOp : public OpRewritePattern<tensor::PadOp> {
  PushDownUnPackThroughPadOp(MLIRContext *context, ControlPropagationFn fun)
      : OpRewritePattern<tensor::PadOp>(context), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    tensor::UnPackOp unpackOp =
        padOp.getSource().getDefiningOp<tensor::UnPackOp>();
    if (!unpackOp)
      return failure();

    if (!controlFn(padOp))
      return failure();

    Location loc = padOp.getLoc();
    // Bail out if one of the padded dimension is a tiled one.
    llvm::SmallBitVector paddedDims = padOp.getPaddedDims();
    ArrayRef<int64_t> innerDimsPos = unpackOp.getInnerDimsPos();
    llvm::SmallBitVector innerDims(paddedDims.size());
    for (int64_t dim : innerDimsPos)
      innerDims.flip(dim);
    if (paddedDims.anyCommon(innerDims))
      return failure();

    Value paddingVal = padOp.getConstantPaddingValue();
    if (!paddingVal)
      return failure();

    // If we have `outer_dims_perms` we need to adjust the padded dimensions.
    ArrayRef<int64_t> outerDimsPerm = unpackOp.getOuterDimsPerm();
    SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highPad = padOp.getMixedHighPad();
    if (!outerDimsPerm.empty()) {
      applyPermutationToVector<OpFoldResult>(lowPad, outerDimsPerm);
      applyPermutationToVector<OpFoldResult>(highPad, outerDimsPerm);
    }
    // Add zero padding for the point loops.
    size_t pointLoopsSize = innerDimsPos.size();
    lowPad.append(pointLoopsSize, rewriter.getIndexAttr(0));
    highPad.append(pointLoopsSize, rewriter.getIndexAttr(0));

    auto newPadOp = rewriter.create<tensor::PadOp>(
        loc, /*result=*/Type(), unpackOp.getSource(), lowPad, highPad,
        paddingVal, padOp.getNofold());

    // Inject the tensor.unpack right after the packed padOp.
    Value outputUnPack = rewriter.create<tensor::EmptyOp>(
        loc, padOp.getResultType().getShape(),
        padOp.getResultType().getElementType());

    Value replacement = rewriter.create<tensor::UnPackOp>(
        loc, newPadOp.getResult(), outputUnPack, innerDimsPos,
        unpackOp.getMixedTiles(), outerDimsPerm);
    rewriter.replaceOp(padOp, replacement);
    return success();
  }

private:
  ControlPropagationFn controlFn;
};

} // namespace

void mlir::linalg::populateDataLayoutPropagationPatterns(
    RewritePatternSet &patterns,
    const ControlPropagationFn &controlPackUnPackPropagation) {
  patterns
      .insert<BubbleUpPackOpThroughGenericOpPattern, BubbleUpPackThroughPadOp,
              BubbleUpPackOpThroughReshapeOp, PushDownUnPackOpThroughGenericOp,
              PushDownUnPackThroughPadOp, PushDownUnPackOpThroughReshapeOp>(
          patterns.getContext(), controlPackUnPackPropagation);
}
