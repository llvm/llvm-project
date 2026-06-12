//===-ElideReinterpretCast.cpp - Expansion patterns for MemRef operations-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cassert>
#include <optional>

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_ELIDEREINTERPRETCASTPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {

static std::optional<SmallVector<int64_t>> getIdentityStrides(MemRefType type) {
  if (!type.getLayout().isIdentity() || !type.hasStaticShape())
    return std::nullopt;

  SmallVector<int64_t> strides(type.getRank(), 1);
  int64_t stride = 1;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    strides[dim] = stride;
    stride *= type.getDimSize(dim);
  }
  return strides;
}

static std::optional<unsigned>
findBaseDimForViewStride(MemRefType baseType, ArrayRef<int64_t> baseStrides,
                         ArrayRef<bool> usedBaseDims, int64_t viewStride,
                         int64_t viewSize) {
  std::optional<unsigned> fallback;
  for (auto [idx, stride] : llvm::enumerate(baseStrides)) {
    if (usedBaseDims[idx] || stride != viewStride ||
        baseType.getDimSize(idx) < viewSize)
      continue;

    // Prefer an exact shape match. Otherwise, use the first dimension large
    // enough to contain the copied logical vector.
    if (baseType.getDimSize(idx) == viewSize)
      return idx;
    if (!fallback)
      fallback = idx;
  }
  return fallback;
}

static std::optional<SmallVector<int64_t>>
delinearizeStaticOffset(int64_t offset, MemRefType baseType,
                        ArrayRef<int64_t> baseStrides) {
  if (offset < 0)
    return std::nullopt;

  SmallVector<int64_t> indices(baseType.getRank(), 0);
  int64_t remainder = offset;
  for (auto [idx, stride] : llvm::enumerate(baseStrides)) {
    indices[idx] = remainder / stride;
    if (indices[idx] >= baseType.getDimSize(idx))
      return std::nullopt;
    remainder %= stride;
  }

  if (remainder != 0)
    return std::nullopt;
  return indices;
}

static std::optional<unsigned> getSingleNonUnitDim(MemRefType type) {
  if (!type.hasStaticShape() || type.getRank() == 0)
    return std::nullopt;

  std::optional<unsigned> nonUnitDim;
  for (auto [idx, dim] : llvm::enumerate(type.getShape())) {
    if (dim == 1)
      continue;
    if (nonUnitDim)
      return std::nullopt;
    nonUnitDim = idx;
  }
  return nonUnitDim;
}

struct CopyLoopDimInfo {
  unsigned viewDim;
  unsigned dstLoopDim;
  int64_t loopSize;
};

struct CopyToLoadStoreInfo {
  SmallVector<CopyLoopDimInfo> loopDims;
  SmallVector<int64_t> staticOffsetIndices;
  std::optional<unsigned> dynamicOffsetDim;
};

/// Builds the index mapping needed to replace a copy into a reinterpret_cast
/// view with scalar stores into the reinterpret_cast base.
///
/// Checklist:
/// - The copy destination must be a `memref.reinterpret_cast`.
/// - The copy source, reinterpret_cast source, and reinterpret_cast result must
///   be ranked memrefs with static shapes.
/// - The reinterpret_cast source/result ranks must match.
/// - The reinterpret_cast source must have static identity layout.
/// - Each non-unit copied view dimension must have a static stride that maps to
///   an identity-layout base dimension.
/// - Static offsets, dynamic only in scalar or effectively-1D copies
///   where the offset can be used directly as one base index.
///
/// Examples that return true:
///
///   // Scalar-shaped copy. There are no copied non-unit dimensions, so
///   // dynamic strides in the scalar view do not affect index mapping.
///   copy memref<1x...x1xf32>
///     to reinterpret_cast memref<base-shape>
///       to memref<1x...x1xf32, strided<[?, ..., ?], offset: ?>>
///
///   // Effectively-1D copy. The single non-unit view dimension is mapped to
///   // an identity-layout base dimension by its static stride.
///   copy memref<1x...xNx...x1xf32>
///     to reinterpret_cast memref<base-shape>
///       to memref<1x...xNx...x1xf32, strided<[..., S, ...]>>
///
///   // Multidimensional copy with static offset. Each non-unit view dimension
///   // is mapped independently by its static stride.
///   copy memref<1x...xNx...xKx...x1xf32>
///     to reinterpret_cast memref<base-shape>
///       to memref<1x...xNx...xKx...x1xf32,
///                 strided<[..., S0, ..., S1, ...], offset: O>>
///
/// Examples that return false:
///
///   // Dynamic stride on a copied view dimension.
///   copy memref<1xNxf32>
///     to reinterpret_cast memref<1xNxMxf32>
///       to memref<1xNxf32, strided<[?, ?]>>
///
///   // Multidimensional copy with dynamic linear offset.
///   copy memref<1xNxKxf32>
///     to reinterpret_cast memref<1xNxMxf32>
///       to memref<1xNxKxf32, strided<[N*M, M, 1], offset: ?>>
static std::optional<CopyToLoadStoreInfo>
getCopyToLoadStoreInfo(memref::CopyOp op, memref::ReinterpretCastOp rc) {
  MemRefType srcType = dyn_cast<MemRefType>(op.getSource().getType());
  MemRefType baseType = dyn_cast<MemRefType>(rc.getSource().getType());
  MemRefType viewType = dyn_cast<MemRefType>(rc.getType());
  // TODO: Support unranked copy sources or reinterpret_cast sources.
  if (!srcType || !baseType || !viewType)
    return std::nullopt;

  // TODO: Support rank-changing reinterpret_casts by converting the
  // destination view indices to base indices. For example, a copy to a
  // memref<2x3xf32> view of memref<6xf32> needs to linearize the view indices
  // as `i * 3 + j`, then combine that with the reinterpret_cast offset before
  // indexing the rank-1 base memref.
  if (baseType.getRank() != viewType.getRank())
    return std::nullopt;

  // TODO: Support dynamic shapes with mixed size operands as loop bounds.
  if (!(srcType.hasStaticShape() && baseType.hasStaticShape() &&
        viewType.hasStaticShape()))
    return std::nullopt;

  assert(srcType.getShape() == viewType.getShape() &&
         "copy source and destination are expected to have the same shape");

  // Store indices are formed in the reinterpret_cast source layout.
  std::optional<SmallVector<int64_t>> baseStrides =
      getIdentityStrides(baseType);
  // TODO: Support non-identity reinterpret_cast source layouts by using the
  // source layout strides as base strides.
  if (!baseStrides)
    return std::nullopt;

  CopyToLoadStoreInfo info;
  SmallVector<bool> usedBaseDims(baseType.getRank(), false);

  // Non-unit view dimensions become loop dimensions in the scalar rewrite.
  for (auto [viewDim, viewSize] : llvm::enumerate(viewType.getShape())) {
    if (viewSize == 1)
      continue;

    // TODO: Support dynamic strides on copied view dimensions.
    if (ShapedType::isDynamic(rc.getStaticStrides()[viewDim]))
      return std::nullopt;

    std::optional<unsigned> dstLoopDim =
        findBaseDimForViewStride(baseType, *baseStrides, usedBaseDims,
                                 rc.getStaticStrides()[viewDim], viewSize);
    assert(dstLoopDim &&
           "static reinterpret_cast stride must map to an identity base "
           "dimension");

    usedBaseDims[*dstLoopDim] = true;
    info.loopDims.push_back(
        CopyLoopDimInfo{static_cast<unsigned>(viewDim), *dstLoopDim, viewSize});
  }

  ArrayRef<int64_t> staticOffsets = rc.getStaticOffsets();
  assert(staticOffsets.size() == 1 && "Expecting single offset");
  if (!ShapedType::isDynamic(staticOffsets[0])) {
    // Static offsets are converted to base indices.
    std::optional<SmallVector<int64_t>> offsetIndices =
        delinearizeStaticOffset(staticOffsets[0], baseType, *baseStrides);
    assert(offsetIndices &&
           "static reinterpret_cast offset must delinearize to in-bounds base "
           "indices");

    for (const CopyLoopDimInfo &loopDim : info.loopDims) {
      assert((*offsetIndices)[loopDim.dstLoopDim] + loopDim.loopSize <=
                 baseType.getDimSize(loopDim.dstLoopDim) &&
             "reinterpret_cast metadata describes an invalid accessible "
             "region");
    }
    info.staticOffsetIndices = std::move(*offsetIndices);
    return info;
  }

  // Dynamic offsets are kept only when they can be used as a single base index.
  // TODO: Support multidimensional dynamic offsets with div/mod
  // delinearization.
  if (info.loopDims.size() > 1)
    return std::nullopt;

  if (info.loopDims.empty()) {
    // TODO: Support scalar dynamic offsets into bases with multiple non-unit
    // dimensions, and all-unit bases with a provably zero offset.
    std::optional<unsigned> nonUnitDim = getSingleNonUnitDim(baseType);
    if (!nonUnitDim)
      return std::nullopt;

    info.dynamicOffsetDim = *nonUnitDim;
    return info;
  }

  unsigned dstLoopDim = info.loopDims.front().dstLoopDim;
  info.dynamicOffsetDim =
      (*baseStrides)[dstLoopDim] == 1 ? dstLoopDim : baseStrides->size() - 1;
  return info;
}

/// Rewrites supported copy operations through `memref.reinterpret_cast` to
/// scalar load/store operations.
///
///   // BEFORE (scalar copy)
///   %view = memref.reinterpret_cast %dst
///     to offset: [O], sizes: [1, ..., 1], strides: [...]
///   memref.copy %src, %view
///
///   // AFTER
///   %v = memref.load %src[0, ..., 0]
///   memref.store %v, %dst[delinearized(O)]
///
///   // BEFORE (effectively-1D copy)
///   %view = memref.reinterpret_cast %dst
///     to offset: [O], sizes: [1, N, 1], strides: [...]
///   memref.copy %src, %view
///
///   // AFTER
///   scf.for %i = 0 to N step 1 {
///     %v = memref.load %src[0, %i, 0]
///     memref.store %v, %dst[delinearized(O) + mapped(%i)]
///   }
///
///   // BEFORE (multidimensional copy with static offset)
///   %view = memref.reinterpret_cast %dst
///     to offset: [O], sizes: [1, N, K], strides: [...]
///   memref.copy %src, %view
///
///   // AFTER
///   scf.for %i = 0 to N step 1 {
///     scf.for %j = 0 to K step 1 {
///       %v = memref.load %src[0, %i, %j]
///       memref.store %v, %dst[delinearized(O) + mapped(%i, %j)]
///     }
///   }
struct CopyToLoadAndStore : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const final {
    Value rcOutput = op.getTarget();
    auto rc = rcOutput.getDefiningOp<memref::ReinterpretCastOp>();
    if (!rc)
      return rewriter.notifyMatchFailure(
          op, "target is not a memref.reinterpret_cast");

    std::optional<CopyToLoadStoreInfo> copyInfo =
        getCopyToLoadStoreInfo(op, rc);
    if (!copyInfo)
      return rewriter.notifyMatchFailure(
          op, "reinterpret_cast does not match scalar or loop copy region");

    Location loc = op.getLoc();
    Value src = op.getSource();
    Value dst = rc.getSource();

    MemRefType srcType = cast<MemRefType>(src.getType());
    MemRefType dstType = cast<MemRefType>(dst.getType());

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one;
    // Reuse common index constants across bounds, steps, and static offsets.
    // Keep `%c1` lazy so scalar copies without loops do not create an unused
    // loop-step constant.
    auto getOrCreateIndexConstant = [&](int64_t value) -> Value {
      if (value == 0)
        return zero;
      if (value == 1) {
        if (!one)
          one = arith::ConstantIndexOp::create(rewriter, loc, 1);
        return one;
      }
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    };

    // Materialize all loop bounds before building the loop nest. Otherwise an
    // inner-loop bound may be created inside an outer loop body.
    SmallVector<Value> upperBounds;
    upperBounds.reserve(copyInfo->loopDims.size());
    for (const CopyLoopDimInfo &loopDim : copyInfo->loopDims)
      upperBounds.push_back(getOrCreateIndexConstant(loopDim.loopSize));

    SmallVector<Value> baseStoreIndices(dstType.getRank(), zero);
    // Static offsets were already delinearized into base indices. Materialize
    // the non-zero starting indices before creating loop bodies.
    if (!copyInfo->staticOffsetIndices.empty()) {
      for (auto [idx, offset] :
           llvm::enumerate(copyInfo->staticOffsetIndices)) {
        if (offset == 0)
          continue;
        baseStoreIndices[idx] = getOrCreateIndexConstant(offset);
      }
    } else if (copyInfo->dynamicOffsetDim) {
      // Supported dynamic offsets are used directly in exactly one base
      // dimension selected by getCopyToLoadStoreInfo.
      SmallVector<OpFoldResult> offsets = rc.getMixedOffsets();
      assert(offsets.size() == 1 && "Expecting single offset");
      baseStoreIndices[*copyInfo->dynamicOffsetDim] =
          getValueOrCreateConstantIndexOp(rewriter, loc, offsets[0]);
    }

    // Scope for OpBuilder::InsertionGuard.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Value step;
      if (!upperBounds.empty())
        step = getOrCreateIndexConstant(1);

      SmallVector<Value> loopIvs;
      loopIvs.reserve(copyInfo->loopDims.size());

      // Build one nested loop per non-unit copied view dimension.
      for (Value upperBound : upperBounds) {
        scf::ForOp loop =
            scf::ForOp::create(rewriter, loc, zero, upperBound, step);
        loopIvs.push_back(loop.getInductionVar());
        rewriter.setInsertionPointToStart(loop.getBody());
      }

      // Load indices are zero except for copied view dimensions, which use the
      // corresponding loop induction variables.
      SmallVector<Value> loadIndices(srcType.getRank(), zero);
      unsigned loopIndex = 0;
      for (const CopyLoopDimInfo &loopDim : copyInfo->loopDims)
        loadIndices[loopDim.viewDim] = loopIvs[loopIndex++];

      // Store indices start from the offset-derived base indices. Add each loop
      // IV to the mapped base dimension.
      SmallVector<Value> storeIndices(baseStoreIndices);
      loopIndex = 0;
      for (const CopyLoopDimInfo &loopDim : copyInfo->loopDims) {
        Value iv = loopIvs[loopIndex++];
        if (storeIndices[loopDim.dstLoopDim] == zero) {
          storeIndices[loopDim.dstLoopDim] = iv;
        } else {
          storeIndices[loopDim.dstLoopDim] = arith::AddIOp::create(
              rewriter, loc, storeIndices[loopDim.dstLoopDim], iv);
        }
      }

      // Emit the scalar load/store at the innermost loop body, or directly at
      // the original copy location for scalar copies.
      Value val = memref::LoadOp::create(rewriter, loc, src, loadIndices);
      memref::StoreOp::create(rewriter, loc, val, dst, storeIndices);
    }

    // If the only user of `rc` is the current Op (which is about to be erased),
    // we can safely erase it.
    bool eraseRc = rcOutput.hasOneUse();
    rewriter.eraseOp(op);
    if (eraseRc)
      rewriter.eraseOp(rc);
    return success();
  }
};

/// Captures info about MemRefs that are effectively 1D (the leading or trailing
/// dims are all 1). The only accepted non-unit dim is either the leading of the
/// trailing dim.
///
/// Examples:
/// memref<1x1x4xf32>, memref<4x1x1xf32>, memref<1x1x1xf32>
///
struct ShapeInfoFor1DMemRef {
  // Are all dims == 1? `false` means that there is exactly one dim != 1.
  bool allOnes = true;
  // If there is a non-unit boundary dim, is it the leading or the trailing dim?
  bool isLeadingDimNonUnit = false;
};

/// Returns information about a MemRef if it contains at most one non-unit
/// dimension.
///
/// The single non-unit dimension, if present, must be on the left or right
/// boundary. Rank-1 non-unit MemRefs are treated as being on both boundaries.
static std::optional<ShapeInfoFor1DMemRef>
getShapeInfoFor1DMemRef(MemRefType type) {
  ArrayRef<int64_t> shape = type.getShape();
  int64_t nonUnitCount =
      llvm::count_if(shape, [](int64_t dim) { return dim != 1; });
  // Return default values if missing non-unit dimension (all-ones MemRef).
  if (nonUnitCount == 0)
    return ShapeInfoFor1DMemRef{};
  // Return no info if MemRef has more non-unit dimensions.
  if (nonUnitCount > 1)
    return std::nullopt;
  // Return no info if MemRef has non-unit dimension in non-boundary positions.
  if (shape.front() == 1 && shape.back() == 1)
    return std::nullopt;

  return ShapeInfoFor1DMemRef{/*allOnes=*/false,
                              /*isLeadingDimNonUnit=*/shape.front() != 1};
}

static bool hasStaticZeroOffset(memref::ReinterpretCastOp rc) {
  ArrayRef<int64_t> offsets = rc.getStaticOffsets();
  // FIXME: Despite what `getStaticOffsets` implies, `reinterpret_cast` takes
  // only a single offset. That should be fixed at the op definition level.
  assert(offsets.size() == 1 && "Expecting single offset");
  return !ShapedType::isDynamic(offsets[0]) && offsets[0] == 0;
}

static std::optional<int64_t> getConstantIndex(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  // Non-constant and dynamic indices
  return std::nullopt;
}

/// Return true if input index is in bounds, i.e. `0 <= idx < upperBound`.
/// Fully dynamic index values (i.e. non-constant) that cannot be analysed are
/// treated as in-bounds.
static bool isConstantIndexExplicitlyOutOfBounds(Value idx,
                                                 int64_t upperBound) {
  // Only statically known `arith.constant` indices are checked here.
  std::optional<int64_t> idxVal = getConstantIndex(idx);
  return idxVal && (*idxVal < 0 || *idxVal >= upperBound);
}

/// Examples accepted by this shape restriction:
///   memref<999xf32>       <-> memref<1x1x999xf32>
///   memref<1x108xf32>     <-> memref<1x1x1x108xf32>
///   memref<100x1xf32>     <-> memref<100x1x1xf32>
///   memref<1>             <-> memref<1x1x1>
///
/// General reinterpret_casts are intentionally rejected.
static bool isPureRankExpansionOrCollapsingRC(memref::ReinterpretCastOp rc) {
  auto inputTy = cast<MemRefType>(rc.getSource().getType());
  auto outputTy = cast<MemRefType>(rc.getResult().getType());

  // Only zero, statically known offsets are accepted. Non-zero or dynamic
  // offsets would require reasoning about storage shifts in the underlying
  // reinterpret_cast, which this helper does not model.
  if (!hasStaticZeroOffset(rc))
    return false;

  // Dynamic sizes/strides prevent precise reasoning about the underlying
  // reinterpret_cast, so only fully static shape metadata is accepted.
  if (llvm::any_of(rc.getStaticSizes(), ShapedType::isDynamic) ||
      llvm::any_of(rc.getStaticStrides(), ShapedType::isDynamic))
    return false;

  // Only shapes with at most one non-unit dimension are accepted. This rules
  // out more general multi-dimensional reinterpret_casts and restricts the
  // helper to unit-dim insertion/removal around a single logical dimension.
  std::optional<ShapeInfoFor1DMemRef> inputNonUnitDim =
      getShapeInfoFor1DMemRef(inputTy);
  std::optional<ShapeInfoFor1DMemRef> outputNonUnitDim =
      getShapeInfoFor1DMemRef(outputTy);
  // Bail out if either type does not satisfy the single-boundary-non-unit-dim
  // restriction described above.
  if (!inputNonUnitDim || !outputNonUnitDim)
    return false;

  // The source and result must either both have a single non-unit dimension
  // or both be all-ones.
  if (inputNonUnitDim->allOnes != outputNonUnitDim->allOnes)
    return false;
  if (inputNonUnitDim->allOnes)
    return true;

  // The preserved non-unit dimension must have the same size.
  if (inputTy.getDimSize(
          inputNonUnitDim->isLeadingDimNonUnit ? 0 : inputTy.getRank() - 1) !=
      outputTy.getDimSize(
          outputNonUnitDim->isLeadingDimNonUnit ? 0 : outputTy.getRank() - 1))
    return false;

  // If both sides have rank > 1, the non-unit dimension must be on the same
  // boundary. Rank-1 MemRefs are accepted against either boundary.
  if (inputTy.getRank() != 1 && outputTy.getRank() != 1 &&
      inputNonUnitDim->isLeadingDimNonUnit !=
          outputNonUnitDim->isLeadingDimNonUnit)
    return false;

  return true;
}

/// Checks statically known and constant indices accessed by a load from a pure
/// rank expansion/collapsing to ensure in-bounds only access. Fully dynamic
/// indices are skipped (there is no way to verify them).
[[maybe_unused]] static bool areIndicesInBounds(memref::LoadOp load) {
  auto rc = load.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
  auto rcOutputTy = cast<MemRefType>(rc.getResult().getType());

  for (auto [pos, idx] : llvm::enumerate(load.getIndices())) {
    // FIXME: This should be ensured by the memref.load semantics.
    // In the long term, this sanity-check may live in the same debug-only
    // checks as `MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS`. This rejects
    // only explicit constant OOB indices. Dynamic/non-constant indices are not
    // filtered here.
    if (isConstantIndexExplicitlyOutOfBounds(idx, rcOutputTy.getDimSize(pos)))
      return false;
  }
  return true;
}

/// Rewrites `memref.load` through a pure rank-only `reinterpret_cast` by
/// mapping the load indices directly onto the source MemRef.

/// Shape restriction gated by isPureRankExpansionOrCollapsingRC().
///
/// BEFORE (rank expansion)
///   %view = memref.reinterpret_cast %src
///     : memref<Nxf32> to memref<1x1xNxf32>
///   %v = memref.load %view[%c0, %c0, %i] : memref<1x1xNxf32>
///
/// AFTER
///   %v = memref.load %src[%i] : memref<Nxf32>
///
/// BEFORE (rank collapsing)
///   %view = memref.reinterpret_cast %src
///     : memref<1x1xNxf32> to memref<Nxf32>
///   %v = memref.load %view[%i] : memref<Nxf32>
///
/// AFTER
///   %c0 = arith.constant 0 : index
///   %v = memref.load %src[%c0, %c0, %i] : memref<1x1xNxf32>
struct RewriteLoadFromReinterpretCast
    : public OpRewritePattern<memref::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto rc = op.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
    if (!rc)
      return rewriter.notifyMatchFailure(
          op, "target is not a memref.reinterpret_cast");
    if (!isPureRankExpansionOrCollapsingRC(rc))
      return rewriter.notifyMatchFailure(
          op, "reinterpret_cast is not a pure rank expansion or collapsing of "
              "a single dimension");

    assert(areIndicesInBounds(op) &&
           "load from reinterpret_cast indexes out of bounds!");

    auto rcOutputTy = cast<MemRefType>(rc.getResult().getType());
    auto rcInputTy = cast<MemRefType>(rc.getSource().getType());

    int64_t rcOutputRank = rcOutputTy.getRank();
    int64_t rcInputRank = rcInputTy.getRank();

    SmallVector<Value> idxs(op.getIndices().begin(), op.getIndices().end());
    SmallVector<Value> rcInputIdxs;
    rcInputIdxs.reserve(rcInputRank);

    // The rewrite only supports reinterpret_casts with at most one non-unit
    // dimension, located at the left or right boundary.
    //
    // The higher-rank side tells which side the reinterpret_cast has
    // expanded/collapsed.
    //
    //   expansion: rcOutput has the higher rank
    //   collapsing : rcInput has the higher rank
    //
    // Example:
    //   memref<999>     -> memref<1x1x999>   : leading extra dims
    //   memref<999x1x1> -> memref<999>       : trailing extra dims
    MemRefType expandedTy =
        rcOutputRank >= rcInputRank ? rcOutputTy : rcInputTy;
    std::optional<ShapeInfoFor1DMemRef> expandedNonUnitDim =
        getShapeInfoFor1DMemRef(expandedTy);
    assert(expandedNonUnitDim && "expected a single boundary non-unit dim");
    bool keepLeadingIndices = expandedNonUnitDim->isLeadingDimNonUnit;

    if (rcOutputRank >= rcInputRank) {
      // Rank expansion:
      //   memref<N>     -> memref<1x1xN> : keep the last rcInputRank indices
      //   memref<N>     -> memref<Nx1x1> : keep the first rcInputRank indices
      //   memref<1>     -> memref<1x1x1> : all indices are zero
      //
      // Any discarded indices are known to be zero from
      // areIndicesInBounds().
      int64_t firstKeptPos =
          keepLeadingIndices ? 0 : rcOutputRank - rcInputRank;
      rcInputIdxs.append(idxs.begin() + firstKeptPos,
                         idxs.begin() + firstKeptPos + rcInputRank);
    } else {
      // Rank collapsing:
      //   memref<1x1xN> -> memref<N>     : reinsert leading zeros
      //   memref<Nx1x1> -> memref<N>     : reinsert trailing zeros
      //   memref<1x1x1> -> memref<1>     : all indices are zero
      //
      // The collapsed-away dimensions are unit dims, so re-adding them with
      // zero indices preserves semantics.
      Value c0 = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
      int64_t rankDiff = rcInputRank - rcOutputRank;

      if (keepLeadingIndices) {
        rcInputIdxs.append(idxs.begin(), idxs.end());
        rcInputIdxs.append(rankDiff, c0);
      } else {
        rcInputIdxs.append(rankDiff, c0);
        rcInputIdxs.append(idxs.begin(), idxs.end());
      }
    }

    assert(rcInputIdxs.size() == static_cast<size_t>(rcInputRank) &&
           "Incorrect number of indices!");

    auto rcInput = rc.getSource();
    // If the only user of rc is the current Op (which is about to be erased),
    // we can safely erase it.
    if (rc.getResult().hasOneUse())
      rewriter.eraseOp(rc);
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, rcInput, rcInputIdxs);
    return success();
  }
};

struct ElideReinterpretCastPass
    : public memref::impl::ElideReinterpretCastPassBase<
          ElideReinterpretCastPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    memref::populateElideReinterpretCastPatterns(patterns);
    ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<memref::CopyOp>([](memref::CopyOp op) {
      auto rc = op.getTarget().getDefiningOp<memref::ReinterpretCastOp>();
      if (!rc)
        return true;
      return !getCopyToLoadStoreInfo(op, rc);
    });
    target.addDynamicallyLegalOp<memref::LoadOp>([](memref::LoadOp op) {
      auto rc = op.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
      if (!rc)
        return true;
      return !isPureRankExpansionOrCollapsingRC(rc);
    });
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           scf::SCFDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::memref::populateElideReinterpretCastPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CopyToLoadAndStore, RewriteLoadFromReinterpretCast>(
      patterns.getContext());
}
