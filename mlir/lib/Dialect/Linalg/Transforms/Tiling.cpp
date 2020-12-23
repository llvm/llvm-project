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

#include "PassDetail.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/FoldedIntrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::scf;

using folded_affine_min = FoldedValueBuilder<AffineMinOp>;

#define DEBUG_TYPE "linalg-tiling"

static bool isZero(Value v) {
  if (auto cst = v.getDefiningOp<ConstantIndexOp>())
    return cst.getValue() == 0;
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
makeTiledLoopRanges(OpBuilder &b, Location loc, AffineMap map,
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
    res.push_back(
        Range{std_constant_index(0), shapeSizes[idx], tileSizes[idx]});
  return std::make_tuple(res, loopIndexToRangeIndex);
}
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
  TileCheck(ValueRange tileSizes) : isTiled(false), tileSizes(tileSizes) {}

  void visitDimExpr(AffineDimExpr expr) {
    isTiled |= !isZero(tileSizes[expr.getPosition()]);
  }
  void visitAffineBinaryOpExpr(AffineBinaryOpExpr expr) {
    visit(expr.getLHS());
    visit(expr.getRHS());
    if (expr.getKind() == mlir::AffineExprKind::Mul)
      assert(expr.getRHS().cast<AffineConstantExpr>().getValue() > 0 &&
             "nonpositive multiplying coefficient");
  }
  bool isTiled;
  ValueRange tileSizes;
};

} // namespace

// IndexedGenericOp explicitly uses induction variables in the loop body. The
// values of the indices that are used in the loop body for any given access of
// input/output memref before `subview` op was applied should be invariant with
// respect to tiling.
//
// Therefore, if the operation is tiled, we have to transform the indices
// accordingly, i.e. offset them by the values of the corresponding induction
// variables that are captured implicitly in the body of the op.
//
// Example. `linalg.indexed_generic` before tiling:
//
// #id_2d = (i, j) -> (i, j)
// #pointwise_2d_trait = {
//   indexing_maps = [#id_2d, #id_2d],
//   iterator_types = ["parallel", "parallel"],
//   n_views = [1, 1]
// }
// linalg.indexed_generic #pointwise_2d_trait %operand, %result {
//   ^bb0(%i: index, %j: index, %operand_in: f32, %result_in: f32):
//     <some operations that use %i, %j>
// }: memref<50x100xf32>, memref<50x100xf32>
//
// After tiling pass with tiles sizes 10 and 25:
//
// #strided = (i, j)[s0, s1, s2] -> (i * s1 + s0 + j * s2)
//
// %c1 = constant 1 : index
// %c0 = constant 0 : index
// %c25 = constant 25 : index
// %c10 = constant 10 : index
// operand_dim_0 = dim %operand, 0 : memref<50x100xf32>
// operand_dim_1 = dim %operand, 1 : memref<50x100xf32>
// scf.for %k = %c0 to operand_dim_0 step %c10 {
//   scf.for %l = %c0 to operand_dim_1 step %c25 {
//     %4 = std.subview %operand[%k, %l][%c10, %c25][%c1, %c1]
//       : memref<50x100xf32> to memref<?x?xf32, #strided>
//     %5 = std.subview %result[%k, %l][%c10, %c25][%c1, %c1]
//       : memref<50x100xf32> to memref<?x?xf32, #strided>
//     linalg.indexed_generic pointwise_2d_trait %4, %5 {
//     ^bb0(%i: index, %j: index, %operand_in: f32, %result_in: f32):
//       // Indices `k` and `l` are implicitly captured in the body.
//       %transformed_i = addi %i, %k : index // index `i` is offset by %k
//       %transformed_j = addi %j, %l : index // index `j` is offset by %l
//       // Every use of %i, %j is replaced with %transformed_i, %transformed_j
//       <some operations that use %transformed_i, %transformed_j>
//     }: memref<?x?xf32, #strided>, memref<?x?xf32, #strided>
//   }
// }
//
// TODO: Investigate whether mixing implicit and explicit indices
// does not lead to losing information.
static void transformIndexedGenericOpIndices(
    OpBuilder &b, LinalgOp op, SmallVectorImpl<Value> &ivs,
    const LoopIndexToRangeIndexMap &loopIndexToRangeIndex) {
  auto indexedGenericOp = dyn_cast<IndexedGenericOp>(op.getOperation());
  if (!indexedGenericOp)
    return;

  // `linalg.indexed_generic` comes in two flavours. One has a region with a
  // single block that defines the loop body. The other has a `fun` attribute
  // that refers to an existing function symbol. The `fun` function call will be
  // inserted in the loop body in that case.
  //
  // TODO: Add support for `linalg.indexed_generic` with `fun` attribute.
  auto &region = indexedGenericOp.region();
  if (region.empty()) {
    indexedGenericOp.emitOpError("expected a region");
    return;
  }
  auto &block = region.front();

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(&block);
  for (unsigned i = 0; i < indexedGenericOp.getNumLoops(); ++i) {
    auto rangeIndex = loopIndexToRangeIndex.find(i);
    if (rangeIndex == loopIndexToRangeIndex.end())
      continue;
    Value oldIndex = block.getArgument(i);
    // Offset the index argument `i` by the value of the corresponding induction
    // variable and replace all uses of the previous value.
    Value newIndex = b.create<AddIOp>(indexedGenericOp.getLoc(), oldIndex,
                                      ivs[rangeIndex->second]);
    for (auto &use : oldIndex.getUses()) {
      if (use.getOwner() == newIndex.getDefiningOp())
        continue;
      use.set(newIndex);
    }
  }
}

static bool isTiled(AffineExpr expr, ValueRange tileSizes) {
  if (!expr)
    return false;
  TileCheck t(tileSizes);
  t.visit(expr);
  return t.isTiled;
}

// Checks whether the `map  varies with respect to a non-zero `tileSize`.
static bool isTiled(AffineMap map, ValueRange tileSizes) {
  if (!map)
    return false;
  for (unsigned r = 0; r < map.getNumResults(); ++r)
    if (isTiled(map.getResult(r), tileSizes))
      return true;
  return false;
}

static SmallVector<Value, 4>
makeTiledShapes(OpBuilder &b, Location loc, LinalgOp linalgOp,
                ValueRange operands, AffineMap map, ValueRange ivs,
                ValueRange tileSizes, ValueRange allShapeSizes) {
  assert(operands.size() == linalgOp.getShapedOperands().size());
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](Value v) { return !isZero(v); })) &&
         "expected as many ivs as non-zero sizes");

  using namespace edsc::op;

  auto shapeSizes = applyMapToValues(b, loc, map, allShapeSizes);
  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<Value, 8> lbs, subShapeSizes;
  for (unsigned idx = 0, idxIvs = 0, e = tileSizes.size(); idx < e; ++idx) {
    bool isTiled = !isZero(tileSizes[idx]);
    lbs.push_back(isTiled ? ivs[idxIvs++] : (Value)std_constant_index(0));
    // Before composing, we need to make range a closed interval.
    Value size = isTiled ? tileSizes[idx] : shapeSizes[idx];
    subShapeSizes.push_back(size - std_constant_index(1));
  }

  auto *op = linalgOp.getOperation();

  SmallVector<Value, 4> res;
  res.reserve(op->getNumOperands());
  for (auto en : llvm::enumerate(operands)) {
    Value shapedOp = en.value();
    ShapedType shapedType = shapedOp.getType().cast<ShapedType>();
    unsigned rank = shapedType.getRank();
    AffineMap map = linalgOp.getIndexingMap(en.index());
    // If the shape is not tiled, we can use it as is.
    if (!isTiled(map, tileSizes)) {
      res.push_back(shapedOp);
      continue;
    }

    // Construct a new subview / subtensor for the tile.
    SmallVector<Value, 4> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (unsigned r = 0; r < rank; ++r) {
      if (!isTiled(map.getSubMap({r}), tileSizes)) {
        offsets.push_back(std_constant_index(0));
        sizes.push_back(std_dim(shapedOp, r));
        strides.push_back(std_constant_index(1));
        continue;
      }

      // Tiling creates a new slice at the proper index, the slice step is 1
      // (i.e. the op does not subsample, stepping occurs in the loop).
      auto m = map.getSubMap({r});
      auto offset = applyMapToValues(b, loc, m, lbs).front();
      offsets.push_back(offset);
      auto closedIntSize = applyMapToValues(b, loc, m, subShapeSizes).front();
      // Resulting size needs to be made half open interval again.
      auto size = closedIntSize + std_constant_index(1);

      // The size of the subview / subtensor should be trimmed to avoid
      // out-of-bounds accesses, unless we statically know the subshape size
      // divides the shape size evenly.
      int64_t shapeSize = shapedType.getDimSize(r);
      auto sizeCst = size.getDefiningOp<ConstantIndexOp>();
      if (ShapedType::isDynamic(shapeSize) || !sizeCst ||
          (shapeSize % sizeCst.getValue()) != 0) {
        // Compute min(size, dim - offset) to avoid out-of-bounds accesses.
        auto minMap = AffineMap::get(
            /*dimCount=*/3, /*symbolCount=*/0,
            {getAffineDimExpr(/*position=*/0, b.getContext()),
             getAffineDimExpr(/*position=*/1, b.getContext()) -
                 getAffineDimExpr(/*position=*/2, b.getContext())},
            b.getContext());
        auto d = std_dim(shapedOp, r);
        size =
            affine_min(b.getIndexType(), minMap, ValueRange{size, d, offset});
      }

      sizes.push_back(size);
      strides.push_back(std_constant_index(1));
    }

    if (shapedType.isa<MemRefType>())
      res.push_back(
          b.create<SubViewOp>(loc, shapedOp, offsets, sizes, strides));
    else
      res.push_back(
          b.create<SubTensorOp>(loc, shapedOp, offsets, sizes, strides));
  }

  return res;
}

template <typename LoopTy>
static Optional<TiledLinalgOp>
tileLinalgOpImpl(OpBuilder &b, LinalgOp op, ValueRange tileSizes,
                 const LinalgTilingOptions &options) {
  auto nLoops = op.getNumLoops();
  // Initial tile sizes may be too big, only take the first nLoops.
  tileSizes = tileSizes.take_front(nLoops);

  if (llvm::all_of(tileSizes, isZero))
    return llvm::None;

  if (auto convOp = dyn_cast<linalg::ConvOp>(op.getOperation())) {
    // For conv op only support tiling along batch dimension (which is the first
    // loop).
    if (convOp.padding() && !llvm::all_of(tileSizes.drop_front(), isZero))
      return llvm::None;
  }

  // 1. Build the tiled loop ranges.
  auto allShapeSizes = op.createFlatListOfOperandDims(b, op.getLoc());
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return llvm::None;

  SmallVector<Range, 4> loopRanges;
  LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  std::tie(loopRanges, loopIndexToRangeIndex) = makeTiledLoopRanges(
      b, op.getLoc(), shapeSizesToLoopsMap, allShapeSizes, tileSizes);
  SmallVector<Attribute, 4> iteratorTypes;
  for (auto attr :
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
    applyPermutationToVector(loopRanges, interchangeVector);
    applyPermutationToVector(iteratorTypes, interchangeVector);
  }

  // 2. Create the tiled loops.
  LinalgOp res = op;
  SmallVector<Value, 4> ivs, tensorResults;
  auto outputTensors = op.getOutputTensors();
  GenerateLoopNest<LoopTy>::doit(
      loopRanges, /*iterArgInitValues*/ outputTensors, iteratorTypes,
      [&](ValueRange localIvs, ValueRange iterArgs) -> scf::ValueVector {
        auto &b = ScopedContext::getBuilderRef();
        auto loc = ScopedContext::getLocation();
        ivs.assign(localIvs.begin(), localIvs.end());

        // When an `interchangeVector` is present, it has been applied to the
        // loop ranges and the iterator types. Apply its inverse to the
        // resulting loop `ivs` to match the op definition.
        SmallVector<Value, 4> interchangedIvs;
        if (!options.interchangeVector.empty())
          interchangedIvs = applyMapToValues(b, loc, invPermutationMap, ivs);
        else
          interchangedIvs.assign(ivs.begin(), ivs.end());

        assert(op.getNumOutputTensors() == iterArgs.size() &&
               "num output tensors must match number of loop iter arguments");

        auto operands = llvm::to_vector<4>(op.getInputs());
        SmallVector<Value, 4> outputBuffers = op.getOutputBuffers();
        // TODO: thanks to simplifying assumption we do not need to worry about
        // order of output buffers and tensors: there is only ever one kind.
        assert(outputBuffers.empty() || iterArgs.empty());
        operands.append(outputBuffers.begin(), outputBuffers.end());
        operands.append(iterArgs.begin(), iterArgs.end());
        SmallVector<Value, 4> tiledOperands =
            makeTiledShapes(b, loc, op, operands, shapeSizesToLoopsMap,
                            interchangedIvs, tileSizes, allShapeSizes);
        auto nonShapedOperands = op.getAssumedNonShapedOperands();
        tiledOperands.append(nonShapedOperands.begin(),
                             nonShapedOperands.end());

        // TODO: use an interface/adaptor to avoid leaking position in
        // `tiledOperands`.
        SmallVector<Type, 4> resultTensorTypes;
        for (OpOperand *opOperand : op.getOutputTensorsOpOperands())
          resultTensorTypes.push_back(
              tiledOperands[opOperand->getOperandNumber()].getType());

        res = op.clone(b, loc, resultTensorTypes, tiledOperands);

        // Insert a subtensor_insert for each output tensor.
        unsigned resultIdx = 0;
        for (OpOperand *opOperand : op.getOutputTensorsOpOperands()) {
          // TODO: use an interface/adaptor to avoid leaking position in
          // `tiledOperands`.
          Value outputTensor = tiledOperands[opOperand->getOperandNumber()];
          if (auto subtensor = outputTensor.getDefiningOp<SubTensorOp>()) {
            tensorResults.push_back(b.create<SubTensorInsertOp>(
                loc, subtensor.source().getType(), res->getResult(resultIdx),
                subtensor.source(), subtensor.offsets(), subtensor.sizes(),
                subtensor.strides(), subtensor.static_offsets(),
                subtensor.static_sizes(), subtensor.static_strides()));
          } else {
            tensorResults.push_back(res->getResult(resultIdx));
          }
          ++resultIdx;
        }
        return scf::ValueVector(tensorResults.begin(), tensorResults.end());
      },
      options.distribution);

  // 3. Transforms index arguments of `linalg.generic` w.r.t. to the tiling.
  transformIndexedGenericOpIndices(b, res, ivs, loopIndexToRangeIndex);

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
Optional<TiledLinalgOp> static tileLinalgOpImpl(
    OpBuilder &b, LinalgOp op, const LinalgTilingOptions &options) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  ScopedContext scope(b, op.getLoc());

  // Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  auto nLoops = op.getNumLoops();
  SmallVector<Value, 4> tileSizeVector =
      options.tileSizeComputationFunction(b, op);
  if (tileSizeVector.size() < nLoops) {
    auto zero = std_constant_index(0);
    tileSizeVector.append(nLoops - tileSizeVector.size(), zero);
  }

  return tileLinalgOpImpl<LoopTy>(b, op, tileSizeVector, options);
}

Optional<TiledLinalgOp>
mlir::linalg::tileLinalgOp(OpBuilder &b, LinalgOp op,
                           const LinalgTilingOptions &options) {
  switch (options.loopType) {
  case LinalgTilingLoopType::Loops:
    return tileLinalgOpImpl<scf::ForOp>(b, op, options);
  case LinalgTilingLoopType::ParallelLoops:
    return tileLinalgOpImpl<scf::ParallelOp>(b, op, options);
  default:;
  }
  return llvm::None;
}

namespace {
/// Helper classes for type list expansion.
template <typename... OpTypes>
class CanonicalizationPatternList;

template <>
class CanonicalizationPatternList<> {
public:
  static void insert(OwningRewritePatternList &patterns, MLIRContext *ctx) {}
};

template <typename OpTy, typename... OpTypes>
class CanonicalizationPatternList<OpTy, OpTypes...> {
public:
  static void insert(OwningRewritePatternList &patterns, MLIRContext *ctx) {
    OpTy::getCanonicalizationPatterns(patterns, ctx);
    CanonicalizationPatternList<OpTypes...>::insert(patterns, ctx);
  }
};

/// Helper classes for type list expansion.
template <typename... OpTypes>
class RewritePatternList;

template <>
class RewritePatternList<> {
public:
  static void insert(OwningRewritePatternList &patterns,
                     const LinalgTilingOptions &options, MLIRContext *ctx) {}
};

template <typename OpTy, typename... OpTypes>
class RewritePatternList<OpTy, OpTypes...> {
public:
  static void insert(OwningRewritePatternList &patterns,
                     const LinalgTilingOptions &options, MLIRContext *ctx) {
    patterns.insert<LinalgTilingPattern<OpTy>>(
        ctx, options, LinalgMarker({}, Identifier::get("tiled", ctx)));
    RewritePatternList<OpTypes...>::insert(patterns, options, ctx);
  }
};
} // namespace

OwningRewritePatternList
mlir::linalg::getLinalgTilingCanonicalizationPatterns(MLIRContext *ctx) {
  OwningRewritePatternList patterns;
  populateLinalgTilingCanonicalizationPatterns(patterns, ctx);
  return patterns;
}

void mlir::linalg::populateLinalgTilingCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  AffineApplyOp::getCanonicalizationPatterns(patterns, ctx);
  AffineForOp::getCanonicalizationPatterns(patterns, ctx);
  AffineMinOp::getCanonicalizationPatterns(patterns, ctx);
  AffineMaxOp::getCanonicalizationPatterns(patterns, ctx);
  scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  scf::ParallelOp::getCanonicalizationPatterns(patterns, ctx);
  ConstantIndexOp::getCanonicalizationPatterns(patterns, ctx);
  SubTensorOp::getCanonicalizationPatterns(patterns, ctx);
  SubViewOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
  ViewOp::getCanonicalizationPatterns(patterns, ctx);
  CanonicalizationPatternList<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >::insert(patterns, ctx);
}

/// Populate the given list with patterns that apply Linalg tiling.
static void insertTilingPatterns(OwningRewritePatternList &patterns,
                                 const LinalgTilingOptions &options,
                                 MLIRContext *ctx) {
  RewritePatternList<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >::insert(patterns, options, ctx);
}

static void applyTilingToLoopPatterns(LinalgTilingLoopType loopType,
                                      FuncOp funcOp,
                                      ArrayRef<int64_t> tileSizes) {
  auto options =
      LinalgTilingOptions().setTileSizes(tileSizes).setLoopType(loopType);
  MLIRContext *ctx = funcOp.getContext();
  OwningRewritePatternList patterns;
  insertTilingPatterns(patterns, options, ctx);
  applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  applyPatternsAndFoldGreedily(funcOp,
                               getLinalgTilingCanonicalizationPatterns(ctx));
  // Drop the marker.
  funcOp.walk([](LinalgOp op) {
    op.removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

namespace {
struct LinalgTilingPass : public LinalgTilingBase<LinalgTilingPass> {
  LinalgTilingPass() = default;
  LinalgTilingPass(ArrayRef<int64_t> sizes) { tileSizes = sizes; }

  void runOnFunction() override {
    applyTilingToLoopPatterns(LinalgTilingLoopType::Loops, getFunction(),
                              tileSizes);
  }
};

struct LinalgTilingToParallelLoopsPass
    : public LinalgTilingToParallelLoopsBase<LinalgTilingToParallelLoopsPass> {
  LinalgTilingToParallelLoopsPass() = default;
  LinalgTilingToParallelLoopsPass(ArrayRef<int64_t> sizes) {
    tileSizes = sizes;
  }

  void runOnFunction() override {
    applyTilingToLoopPatterns(LinalgTilingLoopType::ParallelLoops,
                              getFunction(), tileSizes);
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgTilingPass(ArrayRef<int64_t> tileSizes) {
  return std::make_unique<LinalgTilingPass>(tileSizes);
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgTilingToParallelLoopsPass(ArrayRef<int64_t> tileSizes) {
  return std::make_unique<LinalgTilingToParallelLoopsPass>(tileSizes);
}
