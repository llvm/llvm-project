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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
static PackInfo getPackingInfoFromOperand(AffineMap indexingMap,
                                          OpTy packOrUnPackOp) {
  static_assert(llvm::is_one_of<OpTy, tensor::PackOp, tensor::UnPackOp>::value,
                "applies to only pack or unpack operations");
  LLVM_DEBUG(
      { llvm::dbgs() << "--- Construct PackInfo From an operand ---\n"; });
  PackInfo packInfo;
  int64_t origNumDims = indexingMap.getNumDims();
  SmallVector<AffineExpr> exprs(indexingMap.getResults());
  ArrayRef<int64_t> innerDimsPos = packOrUnPackOp.getInnerDimsPos();
  for (auto [index, innerDimPos, tileSize] :
       llvm::zip_equal(llvm::seq<unsigned>(0, innerDimsPos.size()),
                       innerDimsPos, packOrUnPackOp.getMixedTiles())) {
    int64_t domainDimPos =
        exprs[innerDimPos].template cast<AffineDimExpr>().getPosition();
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

  for (auto dim : packOrUnPackOp.getOuterDimsPerm())
    packInfo.outerDimsOnDomainPerm.push_back(indexingMap.getDimPosition(dim));
  if (!packInfo.outerDimsOnDomainPerm.empty()) {
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
    unsigned posInDomain = expr.cast<AffineDimExpr>().getPosition();
    currentPositionTileLoops[posInDomain] = pos;
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
  if (genericOp.isScalar(opOperand))
    return std::make_tuple(opOperand->get(),
                           AffineMap::get(numLoops, 0, exprs, b.getContext()));

  // Step 1. Construct the information of packing data dimensions; append inner
  // dimensions to the indexing maps for the operand.
  for (auto [index, expr] : llvm::enumerate(exprs)) {
    int64_t dimPos = expr.cast<AffineDimExpr>().getPosition();
    domainDimToOperandDim[dimPos] = index;
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
      int64_t dimPos = exprs[i].cast<AffineDimExpr>().getPosition();
      exprs[i] = b.getAffineDimExpr(inversedOuterPerm[dimPos]);
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
  if (innerDimsPos.empty())
    return std::make_tuple(opOperand->get(), indexingMap);

  auto empty = tensor::PackOp::createDestinationTensor(
      b, loc, opOperand->get(), innerTileSizes, innerDimsPos, outerDimsPerm);
  auto packedOperand = b.create<tensor::PackOp>(
      loc, opOperand->get(), empty, innerDimsPos, innerTileSizes,
      /*padding=*/std::nullopt, outerDimsPerm);
  return std::make_tuple(packedOperand, indexingMap);
}

/// Pack an element-wise genericOp and return it.
static GenericOp packElementWiseOp(RewriterBase &rewriter, GenericOp genericOp,
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

/// Bubbles up tensor.pack op through elementwise generic op. This
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
bubbleUpPackOpThroughElemGenericOp(RewriterBase &rewriter,
                                   tensor::PackOp packOp) {
  auto genericOp = packOp.getSource().getDefiningOp<GenericOp>();
  if (!genericOp)
    return failure();

  if (!isElementwise(genericOp))
    return failure();

  // TODO: Relax the restriction. We are able to bubble up the pack op through
  // multi-result generic op. It just needs more work.
  if (genericOp.getNumResults() != 1)
    return failure();

  // TODO: Add an option for allowing padding values. It could introduce
  // undefined behavior if we unconditionally propagate pack op through all
  // the ops. E.g., if the padding value is zero and there are division ops in
  // a generic op. Some values of padding area could be NaN (0/0).
  if (packOp.getPaddingValue())
    return failure();

  OpOperand *opOperand = genericOp.getDpsInitOperand(0);
  auto packInfo = getPackingInfoFromOperand(
      genericOp.getMatchingIndexingMap(opOperand), packOp);

  // Rebuild the indexing map for the corresponding init operand.
  auto [packedOutOperand, packedOutIndexingMap] =
      getOrCreatePackedViewOfOperand(rewriter, genericOp.getLoc(), packInfo,
                                     genericOp, opOperand);

  // We'll replace the init operand with the destination of pack op if the init
  // operand has not users in the body of the linalg.generic (pure elementwise).
  // If it has users we need to pack the init operand too and replace the init
  // with the packing result.
  Value dest = (genericOp.getRegionOutputArgs()[0].use_empty())
                   ? packOp.getDest()
                   : packedOutOperand;

  return packElementWiseOp(rewriter, genericOp, dest, packedOutIndexingMap,
                           packInfo);
}

/// Wrapper pattern that applies bubbleUpPackOpThroughElemGenericOp method.
struct BubbleUpPackOpThroughElemGenericOpPattern
    : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto genericOp = bubbleUpPackOpThroughElemGenericOp(rewriter, packOp);
    if (failed(genericOp))
      return failure();
    rewriter.replaceOp(packOp, genericOp->getResults());
    return success();
  }
};

// TODO: Relax this restriction. We should unpack an elementwise also
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

/// Push down a tensor.unpack op through elementwise generic op.
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
pushDownUnPackOpThroughElemGenericOp(RewriterBase &rewriter,
                                     GenericOp genericOp) {
  if (!isElementwise(genericOp))
    return failure();
  if (genericOp.getNumResults() != 1)
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
  auto packInfo = getPackingInfoFromOperand(
      genericOp.getMatchingIndexingMap(unPackedOperand), producerUnPackOp);

  // Rebuild the indexing map for the corresponding init operand.
  auto [packedOutOperand, packedOutIndexingMap] =
      getOrCreatePackedViewOfOperand(rewriter, genericOp.getLoc(), packInfo,
                                     genericOp, genericOp.getDpsInitOperand(0));

  // If the dps init operand of the generic is a tensor.empty, do not pack it
  // and forward the new tensor.empty as a destination.
  Value dest = packedOutOperand;
  if (auto initTensor = genericOp.getDpsInitOperand(0)
                            ->get()
                            .getDefiningOp<tensor::EmptyOp>()) {
    if (auto packOp = packedOutOperand.getDefiningOp<tensor::PackOp>())
      dest = packOp.getDest();
  }

  // Pack the genericOp.
  GenericOp newGenericOp = packElementWiseOp(rewriter, genericOp, dest,
                                             packedOutIndexingMap, packInfo);

  auto unPackOp = unPackedOperand->get().getDefiningOp<tensor::UnPackOp>();
  // Insert an unPackOp right after the packed generic.
  Value unPackOpRes =
      rewriter
          .create<tensor::UnPackOp>(
              genericOp.getLoc(),
              newGenericOp.getTiedOpResult(newGenericOp.getDpsInitOperand(0)),
              unPackOp.getDest(), producerUnPackOp.getInnerDimsPos(),
              producerUnPackOp.getMixedTiles(),
              producerUnPackOp.getOuterDimsPerm())
          .getResult();

  return std::make_tuple(newGenericOp, unPackOpRes);
}

// Wrapper pattern that applies pushDownUnPackOpThroughElemGenericOp method.
struct PushDownUnPackOpThroughElemGenericOp
    : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    auto genericAndRepl =
        pushDownUnPackOpThroughElemGenericOp(rewriter, genericOp);
    if (failed(genericAndRepl))
      return failure();
    rewriter.replaceOp(genericOp, std::get<1>(*genericAndRepl));
    return success();
  }
};

/// Propagate a tensor.unpack operation through a tensor.pad. The idea is to
/// add as many zero padding dimensions in `high` and `low` based on the number
/// of point loops.
struct PushDownUnPackThroughPadOp : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    tensor::UnPackOp unpackOp =
        padOp.getSource().getDefiningOp<tensor::UnPackOp>();
    if (!unpackOp)
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
};

} // namespace

void mlir::linalg::populateDataLayoutPropagationPatterns(
    RewritePatternSet &patterns) {
  patterns
      .insert<BubbleUpPackOpThroughElemGenericOpPattern,
              PushDownUnPackOpThroughElemGenericOp, PushDownUnPackThroughPadOp>(
          patterns.getContext());
}
