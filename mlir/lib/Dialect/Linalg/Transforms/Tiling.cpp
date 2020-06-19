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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/FoldUtils.h"

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
static std::tuple<SmallVector<SubViewOp::Range, 4>, LoopIndexToRangeIndexMap>
makeTiledLoopRanges(OpBuilder &b, Location loc, AffineMap map,
                    ArrayRef<Value> allViewSizes,
                    ArrayRef<Value> allTileSizes) {
  assert(allTileSizes.size() == map.getNumResults());
  // Apply `map` to get view sizes in loop order.
  auto viewSizes = applyMapToValues(b, loc, map, allViewSizes);
  SmallVector<Value, 4> tileSizes(allTileSizes.begin(), allTileSizes.end());

  // Traverse the tile sizes, which are in loop order, erase zeros everywhere.
  LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  for (int idx = 0, e = tileSizes.size(), zerosCount = 0; idx < e; ++idx) {
    if (isZero(tileSizes[idx - zerosCount])) {
      viewSizes.erase(viewSizes.begin() + idx - zerosCount);
      tileSizes.erase(tileSizes.begin() + idx - zerosCount);
      ++zerosCount;
      continue;
    }
    loopIndexToRangeIndex[idx] = idx - zerosCount;
  }

  // Create a new range with the applied tile sizes.
  SmallVector<SubViewOp::Range, 4> res;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx)
    res.push_back(SubViewOp::Range{std_constant_index(0), viewSizes[idx],
                                   tileSizes[idx]});
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
  TileCheck(ArrayRef<Value> tileSizes) : isTiled(false), tileSizes(tileSizes) {}

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
  ArrayRef<Value> tileSizes;
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
// TODO(pifon, ntv): Investigate whether mixing implicit and explicit indices
// does not lead to losing information.
static void transformIndexedGenericOpIndices(
    OpBuilder &b, LinalgOp op, SmallVectorImpl<Value> &ivs,
    const LoopIndexToRangeIndexMap &loopIndexToRangeIndex) {
  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");
  auto indexedGenericOp = dyn_cast<IndexedGenericOp>(op.getOperation());
  if (!indexedGenericOp)
    return;

  // `linalg.indexed_generic` comes in two flavours. One has a region with a
  // single block that defines the loop body. The other has a `fun` attribute
  // that refers to an existing function symbol. The `fun` function call will be
  // inserted in the loop body in that case.
  //
  // TODO(pifon): Add support for `linalg.indexed_generic` with `fun` attribute.
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

static bool isTiled(AffineExpr expr, ArrayRef<Value> tileSizes) {
  if (!expr)
    return false;
  TileCheck t(tileSizes);
  t.visit(expr);
  return t.isTiled;
}

// Checks whether the view with index `viewIndex` within `linalgOp` varies with
// respect to a non-zero `tileSize`.
static bool isTiled(AffineMap map, ArrayRef<Value> tileSizes) {
  if (!map)
    return false;
  for (unsigned r = 0; r < map.getNumResults(); ++r)
    if (isTiled(map.getResult(r), tileSizes))
      return true;
  return false;
}

static SmallVector<Value, 4> makeTiledViews(OpBuilder &b, Location loc,
                                            LinalgOp linalgOp,
                                            ArrayRef<Value> ivs,
                                            ArrayRef<Value> tileSizes,
                                            ArrayRef<Value> viewSizes) {
  assert(linalgOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](Value v) { return !isZero(v); })) &&
         "expected as many ivs as non-zero sizes");

  using namespace edsc::op;

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subviews.
  SmallVector<Value, 8> lbs, subViewSizes;
  for (unsigned idx = 0, idxIvs = 0, e = tileSizes.size(); idx < e; ++idx) {
    bool isTiled = !isZero(tileSizes[idx]);
    lbs.push_back(isTiled ? ivs[idxIvs++] : (Value)std_constant_index(0));
    subViewSizes.push_back(isTiled ? tileSizes[idx] : viewSizes[idx]);
  }

  auto *op = linalgOp.getOperation();

  SmallVector<Value, 4> res;
  res.reserve(op->getNumOperands());
  auto viewIteratorBegin = linalgOp.getInputsAndOutputBuffers().begin();
  for (unsigned viewIndex = 0; viewIndex < linalgOp.getNumInputsAndOutputs();
       ++viewIndex) {
    Value view = *(viewIteratorBegin + viewIndex);
    auto viewType = view.getType().cast<MemRefType>();
    unsigned rank = viewType.getRank();
    auto mapAttr = linalgOp.indexing_maps()[viewIndex];
    auto map = mapAttr.cast<AffineMapAttr>().getValue();
    // If the view is not tiled, we can use it as is.
    if (!isTiled(map, tileSizes)) {
      res.push_back(view);
      continue;
    }

    // Construct a new subview for the tile.
    SmallVector<Value, 4> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (unsigned r = 0; r < rank; ++r) {
      if (!isTiled(map.getSubMap({r}), tileSizes)) {
        offsets.push_back(std_constant_index(0));
        sizes.push_back(std_dim(view, r));
        strides.push_back(std_constant_index(1));
        continue;
      }

      // Tiling creates a new slice at the proper index, the slice step is 1
      // (i.e. the slice view does not subsample, stepping occurs in the loop).
      auto m = map.getSubMap({r});
      auto offset = applyMapToValues(b, loc, m, lbs).front();
      offsets.push_back(offset);
      auto size = applyMapToValues(b, loc, m, subViewSizes).front();

      // The size of the subview should be trimmed to avoid out-of-bounds
      // accesses, unless we statically know the subview size divides the view
      // size evenly.
      int64_t viewSize = viewType.getDimSize(r);
      auto sizeCst = size.getDefiningOp<ConstantIndexOp>();
      if (ShapedType::isDynamic(viewSize) || !sizeCst ||
          (viewSize % sizeCst.getValue()) != 0) {
        // Compute min(size, dim - offset) to avoid out-of-bounds accesses.
        auto minMap = AffineMap::get(
            /*dimCount=*/3, /*symbolCount=*/0,
            {getAffineDimExpr(/*position=*/0, b.getContext()),
             getAffineDimExpr(/*position=*/1, b.getContext()) -
                 getAffineDimExpr(/*position=*/2, b.getContext())},
            b.getContext());
        auto d = std_dim(view, r);
        size =
            affine_min(b.getIndexType(), minMap, ValueRange{size, d, offset});
      }

      sizes.push_back(size);
      strides.push_back(std_constant_index(1));
    }

    res.push_back(b.create<SubViewOp>(loc, view, offsets, sizes, strides));
  }

  return res;
}

template <typename LoopTy>
Optional<TiledLinalgOp> static tileLinalgOpImpl(
    OpBuilder &b, LinalgOp op, const LinalgTilingOptions &options) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  ScopedContext scope(b, op.getLoc());

  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");
  // 1. Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  auto nLoops = op.getNumLoops();
  SmallVector<Value, 4> tileSizeVector =
      options.tileSizeComputationFunction(b, op);
  if (tileSizeVector.size() < nLoops) {
    auto zero = std_constant_index(0);
    tileSizeVector.append(nLoops - tileSizeVector.size(), zero);
  }

  ArrayRef<Value> tileSizes = tileSizeVector;
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

  // If interchangeVector is empty, use the identity. Build the permutation map
  // otherwise.
  auto invPermutationMap =
      AffineMap::getMultiDimIdentityMap(tileSizes.size(), b.getContext());
  if (!options.interchangeVector.empty())
    invPermutationMap = inversePermutation(AffineMap::getPermutationMap(
        options.interchangeVector, b.getContext()));
  if (!invPermutationMap)
    return llvm::None;

  // 2. Build the tiled loop ranges.
  auto viewSizes = getViewSizes(b, op);
  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (asserted in the inverse calculation).
  auto mapsRange = op.indexing_maps().getAsRange<AffineMapAttr>();
  auto maps = llvm::to_vector<8>(
      llvm::map_range(mapsRange, [](AffineMapAttr a) { return a.getValue(); }));
  auto viewSizesToLoopsMap = inversePermutation(concatAffineMaps(maps));
  if (!viewSizesToLoopsMap)
    return llvm::None;

  SmallVector<SubViewOp::Range, 4> loopRanges;
  LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  std::tie(loopRanges, loopIndexToRangeIndex) = makeTiledLoopRanges(
      b, scope.getLocation(), viewSizesToLoopsMap, viewSizes, tileSizes);
  if (!options.interchangeVector.empty())
    applyPermutationToVector(loopRanges, options.interchangeVector);

  // 3. Create the tiled loops.
  LinalgOp res = op;
  SmallVector<Value, 4> ivs;
  SmallVector<Attribute, 4> iteratorTypes =
      llvm::to_vector<4>(op.iterator_types().cast<ArrayAttr>().getValue());
  if (!options.interchangeVector.empty())
    applyPermutationToVector(iteratorTypes, options.interchangeVector);
  GenerateLoopNest<LoopTy>::doit(
      loopRanges, iteratorTypes, [&](ValueRange localIvs) {
        auto &b = ScopedContext::getBuilderRef();
        auto loc = ScopedContext::getLocation();
        ivs.assign(localIvs.begin(), localIvs.end());
        SmallVector<Value, 4> ivValues(ivs.begin(), ivs.end());

        // If we have to apply a permutation to the tiled loop nest, we have to
        // reorder the induction variables This permutation is the right one
        // assuming that loopRanges have previously been permuted by
        // (i,j,k)->(k,i,j) So this permutation should be the inversePermutation
        // of that one: (d0,d1,d2)->(d2,d0,d1)
        if (!options.interchangeVector.empty())
          ivValues = applyMapToValues(b, loc, invPermutationMap, ivValues);

        auto views = makeTiledViews(b, loc, op, ivValues, tileSizes, viewSizes);
        auto operands = getAssumedNonViewOperands(op);
        views.append(operands.begin(), operands.end());
        res = op.clone(b, loc, views);
      });

  // 4. Transforms index arguments of `linalg.generic` w.r.t. to the tiling.
  transformIndexedGenericOpIndices(b, res, ivs, loopIndexToRangeIndex);

  // 5. Gather the newly created loops and return them with the new op.
  SmallVector<Operation *, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs) {
    loops.push_back(iv.cast<BlockArgument>().getOwner()->getParentOp());
    assert(loops.back() && "no owner found for induction variable!");
  }
  return TiledLinalgOp{res, loops};
}

Optional<TiledLinalgOp>
mlir::linalg::tileLinalgOp(OpBuilder &b, LinalgOp op,
                           const LinalgTilingOptions &options) {
  if (options.loopType == LinalgTilingLoopType::Loops)
    return tileLinalgOpImpl<scf::ForOp>(b, op, options);
  if (options.loopType == LinalgTilingLoopType::ParallelLoops)
    return tileLinalgOpImpl<scf::ParallelOp>(b, op, options);
  // TODO: Impl tiling to affine loops when it makes sense.
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
  AffineApplyOp::getCanonicalizationPatterns(patterns, ctx);
  AffineForOp::getCanonicalizationPatterns(patterns, ctx);
  AffineMinOp::getCanonicalizationPatterns(patterns, ctx);
  AffineMaxOp::getCanonicalizationPatterns(patterns, ctx);
  scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  scf::ParallelOp::getCanonicalizationPatterns(patterns, ctx);
  ConstantIndexOp::getCanonicalizationPatterns(patterns, ctx);
  SubViewOp::getCanonicalizationPatterns(patterns, ctx);
  ViewOp::getCanonicalizationPatterns(patterns, ctx);
  CanonicalizationPatternList<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >::insert(patterns, ctx);
  return patterns;
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
  applyPatternsAndFoldGreedily(funcOp, patterns);
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
