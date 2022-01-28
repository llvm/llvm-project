//===- Tiling.cpp - Implementation of linalg Tiling -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect Tiling pass.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::scf;

#define DEBUG_TYPE "linalg-tiling"

static bool isZero(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() == 0;
  return false;
}

using LoopIndexToRangeIndexMap = DenseMap<int, int>;

// Creates a number of ranges equal to the number of non-zero in `tileSizes`.
// One for each loop of the LinalgOp that is tiled. The `tileSizes` argument has
// one entry per surrounding loop. It uses zero as the convention that a
// particular loop is not tiled. This convention simplifies implementations by
// avoiding affine map manipulations.
// The returned ranges correspond to the loop ranges, in the proper order, that
// are tiled and for which new loops will be created. Also the function returns
// a map from loop indices of the LinalgOp to the corresponding non-empty range
// indices of newly created loops.
static std::tuple<SmallVector<Range, 4>, LoopIndexToRangeIndexMap>
makeTiledLoopRanges(RewriterBase &b, Location loc, AffineMap map,
                    ValueRange allShapeSizes, ValueRange allTileSizes) {
  assert(allTileSizes.size() == map.getNumResults());
  // Apply `map` to get shape sizes in loop order.
  auto shapeSizes = applyMapToValues(b, loc, map, allShapeSizes);
  SmallVector<Value, 4> tileSizes(allTileSizes.begin(), allTileSizes.end());

  // Traverse the tile sizes, which are in loop order, erase zeros everywhere.
  LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  for (int idx = 0, e = tileSizes.size(), zerosCount = 0; idx < e; ++idx) {
    if (isZero(tileSizes[idx - zerosCount])) {
      shapeSizes.erase(shapeSizes.begin() + idx - zerosCount);
      tileSizes.erase(tileSizes.begin() + idx - zerosCount);
      ++zerosCount;
      continue;
    }
    loopIndexToRangeIndex[idx] = idx - zerosCount;
  }

  // Create a new range with the applied tile sizes.
  SmallVector<Range, 4> res;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx)
    res.push_back(Range{b.create<arith::ConstantIndexOp>(loc, 0),
                        shapeSizes[idx], tileSizes[idx]});
  return std::make_tuple(res, loopIndexToRangeIndex);
}

// All indices returned by IndexOp should be invariant with respect to tiling.
// Therefore, if an operation is tiled, we have to transform the indices
// accordingly, i.e. offset them by the values of the corresponding induction
// variables that are captured implicitly in the body of the op.
//
// Example. `linalg.generic` before tiling:
//
// #id_2d = (i, j) -> (i, j)
// #pointwise_2d_trait = {
//   indexing_maps = [#id_2d, #id_2d],
//   iterator_types = ["parallel", "parallel"]
// }
// linalg.generic #pointwise_2d_trait %operand, %result {
//   ^bb0(%operand_in: f32, %result_in: f32):
//     %i = linalg.index 0 : index
//     %j = linalg.index 1 : index
//     <some operations that use %i, %j>
// }: memref<50x100xf32>, memref<50x100xf32>
//
// After tiling pass with tiles sizes 10 and 25:
//
// #strided = (i, j)[s0, s1, s2] -> (i * s1 + s0 + j * s2)
//
// %c1 = arith.constant 1 : index
// %c0 = arith.constant 0 : index
// %c25 = arith.constant 25 : index
// %c10 = arith.constant 10 : index
// operand_dim_0 = dim %operand, 0 : memref<50x100xf32>
// operand_dim_1 = dim %operand, 1 : memref<50x100xf32>
// scf.for %k = %c0 to operand_dim_0 step %c10 {
//   scf.for %l = %c0 to operand_dim_1 step %c25 {
//     %4 = std.subview %operand[%k, %l][%c10, %c25][%c1, %c1]
//       : memref<50x100xf32> to memref<?x?xf32, #strided>
//     %5 = std.subview %result[%k, %l][%c10, %c25][%c1, %c1]
//       : memref<50x100xf32> to memref<?x?xf32, #strided>
//     linalg.generic pointwise_2d_trait %4, %5 {
//     ^bb0(%operand_in: f32, %result_in: f32):
//       %i = linalg.index 0 : index
//       %j = linalg.index 1 : index
//       // Indices `k` and `l` are implicitly captured in the body.
//       %transformed_i = arith.addi %i, %k : index // index `i` is offset by %k
//       %transformed_j = arith.addi %j, %l : index // index `j` is offset by %l
//       // Every use of %i, %j is replaced with %transformed_i, %transformed_j
//       <some operations that use %transformed_i, %transformed_j>
//     }: memref<?x?xf32, #strided>, memref<?x?xf32, #strided>
//   }
// }
//
// TODO: Investigate whether mixing implicit and explicit indices
// does not lead to losing information.
static void
transformIndexOps(RewriterBase &b, LinalgOp op, SmallVectorImpl<Value> &ivs,
                  const LoopIndexToRangeIndexMap &loopIndexToRangeIndex) {
  SmallVector<Value> allIvs(op.getNumLoops(), nullptr);
  for (auto &en : enumerate(allIvs)) {
    auto rangeIndex = loopIndexToRangeIndex.find(en.index());
    if (rangeIndex == loopIndexToRangeIndex.end())
      continue;
    en.value() = ivs[rangeIndex->second];
  }
  addTileLoopIvsToIndexOpResults(b, op, allIvs);
}

// Insert a tile `source` into the destination tensor `dest`. The position at
// which the tile is inserted (as well as size of tile) is taken from a given
// ExtractSliceOp `sliceOp`.
static Value insertSliceIntoTensor(RewriterBase &b, Location loc,
                                   tensor::ExtractSliceOp sliceOp, Value source,
                                   Value dest) {
  return b.create<tensor::InsertSliceOp>(
      loc, sliceOp.source().getType(), source, dest, sliceOp.offsets(),
      sliceOp.sizes(), sliceOp.strides(), sliceOp.static_offsets(),
      sliceOp.static_sizes(), sliceOp.static_strides());
}

template <typename LoopTy>
static FailureOr<TiledLinalgOp>
tileLinalgOpImpl(RewriterBase &b, LinalgOp op, ValueRange tileSizes,
                 const LinalgTilingOptions &options) {
  auto nLoops = op.getNumLoops();
  // Initial tile sizes may be too big, only take the first nLoops.
  tileSizes = tileSizes.take_front(nLoops);

  if (llvm::all_of(tileSizes, isZero)) {
    TiledLinalgOp tiledOp;
    tiledOp.op = cast<LinalgOp>(b.clone(*op.getOperation()));
    tiledOp.tensorResults.assign(tiledOp.op->result_begin(),
                                 tiledOp.op->result_end());
    return tiledOp;
  }

  // 1. Build the tiled loop ranges.
  auto allShapeSizes = op.createFlatListOfOperandDims(b, op.getLoc());
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return failure();

  SmallVector<Range, 4> loopRanges;
  LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  std::tie(loopRanges, loopIndexToRangeIndex) = makeTiledLoopRanges(
      b, op.getLoc(), shapeSizesToLoopsMap, allShapeSizes, tileSizes);

  SmallVector<Attribute, 4> iteratorTypes;
  for (const auto &attr :
       enumerate(op.iterator_types().cast<ArrayAttr>().getValue())) {
    if (loopIndexToRangeIndex.count(attr.index()))
      iteratorTypes.push_back(attr.value());
  }
  // If interchangeVector is empty, use the identity. Build the permutation map
  // otherwise.
  auto invPermutationMap =
      AffineMap::getMultiDimIdentityMap(tileSizes.size(), b.getContext());
  if (!options.interchangeVector.empty()) {
    // Based on the pruned iterations (due to zero tile size), recompute the
    // interchange vector.
    SmallVector<unsigned, 4> interchangeVector;
    interchangeVector.reserve(options.interchangeVector.size());
    for (auto pos : options.interchangeVector) {
      auto it = loopIndexToRangeIndex.find(pos);
      if (it == loopIndexToRangeIndex.end())
        continue;
      interchangeVector.push_back(it->second);
    }
    // Interchange vector is guaranteed to be a permutation,
    // `inversePermutation` must succeed.
    invPermutationMap = inversePermutation(
        AffineMap::getPermutationMap(interchangeVector, b.getContext()));
    assert(invPermutationMap);
    SmallVector<int64_t> permutation(interchangeVector.begin(),
                                     interchangeVector.end());
    applyPermutationToVector(loopRanges, permutation);
    applyPermutationToVector(iteratorTypes, permutation);
  }

  // 2. Create the tiled loops.
  LinalgOp res = op;
  SmallVector<Value, 4> ivs, tensorResults;
  auto tiledLoopBodyBuilder =
      [&](OpBuilder &builder, Location loc, ValueRange localIvs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
    ivs.assign(localIvs.begin(), localIvs.end());

    // When an `interchangeVector` is present, it has been applied to the
    // loop ranges and the iterator types. Apply its inverse to the
    // resulting loop `ivs` to match the op definition.
    SmallVector<Value, 4> interchangedIvs;
    if (!options.interchangeVector.empty())
      interchangedIvs = applyMapToValues(b, loc, invPermutationMap, ivs);
    else
      interchangedIvs.assign(ivs.begin(), ivs.end());

    // Tile the `operandValuesToUse` that either match the `op` operands
    // themselves or the tile loop arguments forwarding them.
    assert(operandValuesToUse.size() ==
               static_cast<size_t>(op.getNumInputsAndOutputs()) &&
           "expect the number of operands and inputs and outputs to match");
    SmallVector<Value> valuesToTile = operandValuesToUse;
    auto sizeBounds =
        applyMapToValues(b, loc, shapeSizesToLoopsMap, allShapeSizes);
    SmallVector<Value, 4> tiledOperands = makeTiledShapes(
        b, loc, op, valuesToTile, interchangedIvs, tileSizes, sizeBounds);

    // TODO: use an interface/adaptor to avoid leaking position in
    // `tiledOperands`.
    SmallVector<Type, 4> resultTensorTypes;
    for (OpOperand *opOperand : op.getOutputTensorOperands())
      resultTensorTypes.push_back(
          tiledOperands[opOperand->getOperandNumber()].getType());

    res = op.clone(b, loc, resultTensorTypes, tiledOperands);

    // Insert a insert_slice for each output tensor.
    unsigned resultIdx = 0;
    for (OpOperand *opOperand : op.getOutputTensorOperands()) {
      // TODO: use an interface/adaptor to avoid leaking position in
      // `tiledOperands`.
      Value outputTensor = tiledOperands[opOperand->getOperandNumber()];
      // TODO: Propagate RewriterBase everywhere.
      IRRewriter rewriter(b);
      if (auto sliceOp = outputTensor.getDefiningOp<tensor::ExtractSliceOp>()) {
        tensorResults.push_back(insertSliceIntoTensor(rewriter, loc, sliceOp,
                                                      res->getResult(resultIdx),
                                                      sliceOp.source()));
      } else {
        tensorResults.push_back(res->getResult(resultIdx));
      }
      ++resultIdx;
    }
    return scf::ValueVector(tensorResults.begin(), tensorResults.end());
  };
  GenerateLoopNest<LoopTy>::doit(b, op.getLoc(), loopRanges, op, iteratorTypes,
                                 tiledLoopBodyBuilder, options.distribution,
                                 options.distributionTypes);

  // 3. Transform IndexOp results w.r.t. the tiling.
  transformIndexOps(b, res, ivs, loopIndexToRangeIndex);

  // 4. Gather the newly created loops and return them with the new op.
  SmallVector<Operation *, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs) {
    if (iv.isa<BlockArgument>()) {
      loops.push_back(iv.cast<BlockArgument>().getOwner()->getParentOp());
      assert(loops.back() && "no owner found for induction variable!");
    } else {
      // TODO: Instead of doing this, try to recover the ops used instead of the
      // loop.
      loops.push_back(nullptr);
    }
  }

  // 5. Get the tensor results from the outermost loop if available. Otherwise
  // use the previously captured `tensorResults`.
  Operation *outermostLoop = nullptr;
  for (Operation *loop : loops)
    if ((outermostLoop = loop))
      break;

  return TiledLinalgOp{
      res, loops, outermostLoop ? outermostLoop->getResults() : tensorResults};
}

template <typename LoopTy>
FailureOr<TiledLinalgOp> static tileLinalgOpImpl(
    RewriterBase &b, LinalgOp op, const LinalgTilingOptions &options) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);

  if (!options.tileSizeComputationFunction)
    return failure();

  // Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  auto nLoops = op.getNumLoops();
  SmallVector<Value, 4> tileSizeVector =
      options.tileSizeComputationFunction(b, op);
  if (tileSizeVector.size() < nLoops) {
    auto zero = b.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    tileSizeVector.append(nLoops - tileSizeVector.size(), zero);
  }

  return tileLinalgOpImpl<LoopTy>(b, op, tileSizeVector, options);
}

FailureOr<TiledLinalgOp>
mlir::linalg::tileLinalgOp(RewriterBase &b, LinalgOp op,
                           const LinalgTilingOptions &options) {
  switch (options.loopType) {
  case LinalgTilingLoopType::Loops:
    return tileLinalgOpImpl<scf::ForOp>(b, op, options);
  case LinalgTilingLoopType::ParallelLoops:
    return tileLinalgOpImpl<scf::ParallelOp>(b, op, options);
  case LinalgTilingLoopType::TiledLoops:
    return tileLinalgOpImpl<linalg::TiledLoopOp>(b, op, options);
  default:;
  }
  return failure();
}

/// Generate a loop nest around a given tensor::PadOp (for tiling). `newPadOp`
/// and `loopNest` are output parameters that return the new (tiled)
/// tensor::PadOp and the loop nest.
static LogicalResult tilePadOp(RewriterBase &builder, tensor::PadOp op,
                               tensor::PadOp &newPadOp, LoopNest &loopNest,
                               const LinalgTilingOptions &options) {
  Location loc = op.getLoc();
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(op);

  // Clone tensor::PadOp so that the existing op can be replaced more easily.
  newPadOp = cast<tensor::PadOp>(builder.clone(*op.getOperation()));
  // Get rank and tile sizes.
  int64_t rank = op.getResultType().getRank();
  SmallVector<Value> tileSizes =
      options.tileSizeComputationFunction(builder, op);
  // Normalize untiled padding dimensions to 0.
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  tileSizes.append(rank - tileSizes.size(), zero);
  // Compute lower and upper bounds of the loop nest.
  TilingInterface tilingInterface =
      dyn_cast<TilingInterface>(op.getOperation());
  SmallVector<Range> ranges = tilingInterface.getIterationDomain(builder);
  SmallVector<Value> lbs, dims, allDims, steps;
  for (int64_t i = 0; i < rank; ++i) {
    allDims.push_back(ranges[i].size);
    if (!isZero(tileSizes[i])) {
      lbs.push_back(ranges[i].offset);
      dims.push_back(ranges[i].size);
      steps.push_back(tileSizes[i]);
    }
  }
  // Generate loop nest: One loop per dimension.
  SmallVector<Value> destOperand =
      tilingInterface.getDestinationOperands(builder);
  loopNest = mlir::scf::buildLoopNest(
      builder, loc, lbs, /*ubs=*/dims, steps, ValueRange(destOperand),
      [&](OpBuilder &b, Location loc, ValueRange localIvs,
          ValueRange iterArgs) -> scf::ValueVector {
        // Compute offsets and sizes of ExtractSliceOp.
        SmallVector<Value> offsets =
            computeTileOffsets(b, loc, localIvs, tileSizes);
        SmallVector<Value> sizes =
            computeTileSizes(b, loc, localIvs, tileSizes, allDims);
        // Create ExtractSliceOp: Extract a tile from the tensor::PadOp.
        // Note: The tensor::PadOp is located outside of the loop nest. It is
        // later moved inside by ExtractSliceOfPadTensorSwapPattern.
        auto map = AffineMap::getMultiDimIdentityMap(rank, b.getContext());
        Value tiledOutput =
            makeTiledShape(b, loc, newPadOp->getResult(0), tileSizes, map,
                           offsets, allDims, sizes);
        auto sliceOp = tiledOutput.getDefiningOp<tensor::ExtractSliceOp>();
        assert(sliceOp && "expected ExtractSliceOp");
        // Insert the tile into the output tensor.
        // TODO: Propagate RewriterBase everywhere.
        IRRewriter rewriter(b);
        Value yieldValue =
            insertSliceIntoTensor(rewriter, loc, sliceOp, sliceOp, iterArgs[0]);
        return scf::ValueVector({yieldValue});
      });
  return success();
}

namespace {
struct PadOpTilingPattern : public OpRewritePattern<tensor::PadOp> {
  PadOpTilingPattern(MLIRContext *ctx, LinalgTilingOptions opt)
      : OpRewritePattern<tensor::PadOp>(ctx), options(std::move(opt)) {}

  LogicalResult matchAndRewrite(tensor::PadOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(LinalgTransforms::kLinalgTransformMarker))
      return failure();
    tensor::PadOp newPadOp;
    LoopNest loopNest;
    if (failed(tilePadOp(rewriter, op, newPadOp, loopNest, options)))
      return failure();
    newPadOp->setAttr(LinalgTransforms::kLinalgTransformMarker,
                      rewriter.getUnitAttr());
    // Replace all uses of the original tensor::PadOp.
    rewriter.replaceOp(op, loopNest.getResults()[0]);
    return success();
  }

  LinalgTilingOptions options;
};
} // namespace

namespace {
/// Helper classes for type list expansion.
template <typename... OpTypes>
class CanonicalizationPatternList;

template <>
class CanonicalizationPatternList<> {
public:
  static void insert(RewritePatternSet &patterns) {}
};

template <typename OpTy, typename... OpTypes>
class CanonicalizationPatternList<OpTy, OpTypes...> {
public:
  static void insert(RewritePatternSet &patterns) {
    OpTy::getCanonicalizationPatterns(patterns, patterns.getContext());
    CanonicalizationPatternList<OpTypes...>::insert(patterns);
  }
};
} // namespace

RewritePatternSet
mlir::linalg::getLinalgTilingCanonicalizationPatterns(MLIRContext *ctx) {
  RewritePatternSet patterns(ctx);
  populateLinalgTilingCanonicalizationPatterns(patterns);
  return patterns;
}

void mlir::linalg::populateLinalgTilingCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  AffineApplyOp::getCanonicalizationPatterns(patterns, ctx);
  AffineForOp::getCanonicalizationPatterns(patterns, ctx);
  AffineMinOp::getCanonicalizationPatterns(patterns, ctx);
  AffineMaxOp::getCanonicalizationPatterns(patterns, ctx);
  arith::ConstantIndexOp::getCanonicalizationPatterns(patterns, ctx);

  memref::SubViewOp::getCanonicalizationPatterns(patterns, ctx);
  memref::ViewOp::getCanonicalizationPatterns(patterns, ctx);

  scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  scf::ParallelOp::getCanonicalizationPatterns(patterns, ctx);

  tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, ctx);

  InitTensorOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::PadOp::getCanonicalizationPatterns(patterns, ctx);
  ctx->getLoadedDialect<LinalgDialect>()->getCanonicalizationPatterns(patterns);

  CanonicalizationPatternList<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >::insert(patterns);
}

/// Populate the given list with patterns that apply Linalg tiling.
static void insertTilingPatterns(RewritePatternSet &patterns,
                                 const LinalgTilingOptions &options) {
  auto *ctx = patterns.getContext();
  LinalgTransformationFilter f(ArrayRef<StringAttr>{},
                               StringAttr::get(ctx, "tiled"));
  TilingPatterns<GenericOp,
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                 >::insert(patterns, options, f);
  patterns.add<PadOpTilingPattern>(ctx, options);
}

void mlir::linalg::populatePadTensorTilingPatterns(
    RewritePatternSet &patterns, const LinalgTilingOptions &options) {
  auto *ctx = patterns.getContext();
  patterns.add<PadOpTilingPattern>(ctx, options);
}

static void applyExtractSliceOfPadTensorSwapPattern(FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<ExtractSliceOfPadTensorSwapPattern>(patterns.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  (void)applyPatternsAndFoldGreedily(
      funcOp, getLinalgTilingCanonicalizationPatterns(ctx));
}

namespace {
struct LinalgTilingPass : public LinalgTilingBase<LinalgTilingPass> {
  LinalgTilingPass() = default;
  LinalgTilingPass(ArrayRef<int64_t> tileSizes, LinalgTilingLoopType loopType,
                   ArrayRef<StringRef> distributionTypes) {
    this->tileSizes = tileSizes;
    this->loopType = "";
    this->loopTypeEnum = loopType;
    this->distributionTypes = llvm::to_vector<2>(llvm::map_range(
        distributionTypes, [](StringRef ref) { return ref.str(); }));
  }

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    LinalgTilingLoopType type =
        llvm::StringSwitch<LinalgTilingLoopType>(loopType)
            .Case("for", LinalgTilingLoopType::Loops)
            .Case("affine", LinalgTilingLoopType::AffineLoops)
            .Case("parallel", LinalgTilingLoopType::ParallelLoops)
            .Case("tiled_loop", LinalgTilingLoopType::TiledLoops)
            .Default(loopTypeEnum);
    auto distTypes = llvm::to_vector<2>(llvm::map_range(
        distributionTypes, [](std::string &str) { return StringRef(str); }));
    auto options = LinalgTilingOptions()
                       .setTileSizes(tileSizes)
                       .setLoopType(type)
                       .setDistributionTypes(distTypes);
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    insertTilingPatterns(patterns, options);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    (void)applyPatternsAndFoldGreedily(
        funcOp, getLinalgTilingCanonicalizationPatterns(ctx));
    // Drop the marker.
    funcOp.walk([](LinalgOp op) {
      op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });

    // Apply swap pattern after generating loop nest and running
    // canonicalizations.
    applyExtractSliceOfPadTensorSwapPattern(funcOp);
  }

  LinalgTilingLoopType loopTypeEnum;
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgTilingPass(ArrayRef<int64_t> tileSizes,
                             linalg::LinalgTilingLoopType loopType,
                             ArrayRef<StringRef> distributionTypes) {
  return std::make_unique<LinalgTilingPass>(tileSizes, loopType,
                                            distributionTypes);
}
