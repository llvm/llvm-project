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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Repeated.h"
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

/// Returns true if `rc` represents a scalar view (all sizes == 1)
/// into a memref that has exactly one non-unit dimension located at
/// either the first or last position (i.e. a "row" or "column").
///
/// Examples that return true:
///
///   // Row-major slice (last dim is non-unit)
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1, 1], strides: [1, 1, 1]
///     : memref<1x1x8xi32> to memref<1x1x1xi32>
///
///   // Column-major slice (first dim is non-unit)
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1], strides: [1, 1]
///     : memref<2x1xf32> to memref<1x1xf32>
///
///   // Random strides
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1], strides: [10, 100]
///     : memref<2x1xf32, strided<[10, 100]>>
///         to memref<1x1xf32>
///
///   // Rank-1 case
///   memref.reinterpret_cast %buf to offset: [%off],
///     sizes: [1], strides: [1]
///     : memref<8xi32> to memref<1xi32>
///
/// Examples that return false:
///
///   // More non-unit dims
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1, 1], strides: [1, 1, 1]
///     : memref<1x2x8xi32> to memref<1x1x1xi32>
///
///   // View is not scalar (size != 1)
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [2, 1], strides: [1, 1]
///     : memref<1x2xf32> to memref<2x1xf32>
///
///   // Base has non-identity layout
///   %buff = memref.alloc() : memref<1x2xf32, strided<[1, 3]>>
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1], strides: [1, 1]
///     : memref<1x2xf32, strided<[1, 3]>> to memref<1x1xf32>
static bool isScalarSlice(memref::ReinterpretCastOp rc) {
  auto rcInputTy = dyn_cast<MemRefType>(rc.getSource().getType());
  auto rcOutputTy = dyn_cast<MemRefType>(rc.getType());

  // Reject strided base - logic for computing linear idx is TODO
  if (!rcInputTy.getLayout().isIdentity())
    return false;

  // Reject non-matching ranks
  unsigned srcRank = rcInputTy.getRank();
  if (srcRank != rcOutputTy.getRank())
    return false;

  ArrayRef<int64_t> sizes = rc.getStaticSizes();

  // View must be scalar: memref<1x...x1>
  if (!llvm::all_of(rcOutputTy.getShape(),
                    [](int64_t dim) { return dim == 1; }))
    return false;

  // Sizes must all be statically 1
  if (!llvm::all_of(sizes, [](int64_t size) {
        return !ShapedType::isDynamic(size) && size == 1;
      }))
    return false;

  // Rank-1 special case
  if (srcRank == 1) {
    // Reject non-scalar output
    if (rcOutputTy.getDimSize(0) > 1)
      return false;
  }

  int nonUnitCount =
      std::count_if(rcInputTy.getShape().begin(), rcInputTy.getShape().end(),
                    [](int dim) { return dim != 1; });
  return nonUnitCount == 1;
}

/// Rewrites `memref.copy` of a 1-element MemRef as a scalar load-store pair
///
/// The pattern matches a reinterpret_cast that creates a scalar view
/// (`sizes = [1, ..., 1]`) into a memref with a single non-unit dimension.
/// Since the view contains only one element, the accessed address is
/// determined solely by the base pointer and the offset.
///
/// Two layouts are supported:
///   * row-major slice  (stride pattern [N, ..., 1])
///   * column-major slice (stride pattern [1, ..., N])
///
/// BEFORE (row-major slice)
///   %view = memref.reinterpret_cast %base
///     to offset: [%off], sizes: [1, ..., 1], strides: [N, ..., 1]
///       : memref<1x...xNxf32>
///         to memref<1x...x1xf32, strided<[N, ..., 1], offset: ?>>
///   memref.copy %src, %view
///     : memref<1x...x1xf32>
///       to memref<1x...x1xf32, strided<[N, ..., 1], offset: ?>>
///
/// AFTER
///   %c0 = arith.constant 0 : index
///   %v  = memref.load %src[%c0, ..., %c0] : memref<1x...x1xf32>
///   memref.store %v, %base[%c0, ..., %off] : memref<1x...xNxf32>
///
/// BEFORE (column-major slice)
///   %view = memref.reinterpret_cast %base
///     to offset: [%off], sizes: [1, ..., 1], strides: [1, ..., N]
///       : memref<Nx...x1xf32>
///         to memref<1x...x1xf32, strided<[1, ..., N], offset: ?>>
///   memref.copy %src, %view
///     : memref<1x...x1xf32>
///       to memref<1x...x1xf32, strided<[1, ..., N], offset: ?>>
///
/// AFTER
///   %c0 = arith.constant 0 : index
///   %v  = memref.load %src[%c0, ..., %c0] : memref<1x...x1xf32>
///   memref.store %v, %base[%off, ..., %c0] : memref<Nx...x1xf32>
struct CopyToScalarLoadAndStore : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const final {
    Value rcOutput = op.getTarget();
    auto rc = rcOutput.getDefiningOp<memref::ReinterpretCastOp>();
    if (!rc)
      return rewriter.notifyMatchFailure(
          op, "target is not a memref.reinterpret_cast");

    if (!isScalarSlice(rc))
      return rewriter.notifyMatchFailure(
          op, "reinterpret_cast does not match scalar slice");

    Location loc = op.getLoc();

    Value src = op.getSource();
    Value dst = rc.getSource();

    auto dstType = cast<MemRefType>(dst.getType());
    unsigned dstRank = dstType.getRank();

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

    auto srcType = cast<MemRefType>(src.getType());
    Repeated<Value> loadIndices(srcType.getRank(), zero);
    auto offsets = rc.getMixedOffsets();
    assert(offsets.size() == 1 && "Expecting single offset");
    OpFoldResult offset = offsets[0];
    Value storeOffset = getValueOrCreateConstantIndexOp(rewriter, loc, offset);
    unsigned offsetDim = dstType.getDimSize(0) == 1 ? dstRank - 1 : 0;
    SmallVector<Value> storeIndices(dstRank, zero);
    storeIndices[offsetDim] = storeOffset;
    // If the only user of `rc` is the current Op (which is about to be erased),
    // we can safely erase it.
    if (rcOutput.hasOneUse())
      rewriter.eraseOp(rc);

    Value val = memref::LoadOp::create(rewriter, loc, src, loadIndices);
    memref::StoreOp::create(rewriter, loc, val, dst, storeIndices);

    rewriter.eraseOp(op);
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
      return !isScalarSlice(rc);
    });
    target.addDynamicallyLegalOp<memref::LoadOp>([](memref::LoadOp op) {
      auto rc = op.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
      if (!rc)
        return true;
      return !isPureRankExpansionOrCollapsingRC(rc);
    });
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::memref::populateElideReinterpretCastPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CopyToScalarLoadAndStore, RewriteLoadFromReinterpretCast>(
      patterns.getContext());
}
