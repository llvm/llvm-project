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

//===----------------------------------------------------------------------===//
// Load Rewrite Helpers
//===----------------------------------------------------------------------===//

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

using NonUnitDimMapping = SmallVector<std::pair<int64_t, int64_t>>;

/// Shape restriction accepting only unit-dim insertion/removal
/// reinterpret_casts.
///
/// Examples accepted:
///   memref<1x1x1x108xf32>    <-> memref<1x108xf32>
///   memref<100x1xf32>        <-> memref<100x1x1xf32>
///   memref<1x33x40xf32>      <-> memref<33x1x1x40xf32>
///   memref<1>                <-> memref<1x1x1>
///
/// Returns the mapping of non-unit dimensions from the source
/// to the result MemRef if the reinterpret_cast preserved sizes and order (no
/// transposition) of these dimensions.
static std::optional<NonUnitDimMapping>
getNonUnitDimMapping(memref::ReinterpretCastOp rc) {
  auto inputTy = cast<MemRefType>(rc.getSource().getType());
  auto outputTy = cast<MemRefType>(rc.getResult().getType());

  // Only zero, statically known offsets are accepted. Non-zero or dynamic
  // offsets would require reasoning about storage shifts in the underlying
  // reinterpret_cast, which this helper does not model.
  if (!hasStaticZeroOffset(rc))
    return std::nullopt;

  // Dynamic sizes/strides prevent precise reasoning about the underlying
  // reinterpret_cast, so only fully static shape metadata is accepted.
  if (llvm::any_of(rc.getStaticSizes(), ShapedType::isDynamic) ||
      llvm::any_of(rc.getStaticStrides(), ShapedType::isDynamic))
    return std::nullopt;

  ArrayRef<int64_t> inputShape = inputTy.getShape();
  ArrayRef<int64_t> outputShape = outputTy.getShape();
  int64_t inputDim = 0;
  int64_t outputDim = 0;
  int64_t inputRank = inputTy.getRank();
  int64_t outputRank = outputTy.getRank();
  NonUnitDimMapping mapping;

  // The preserved non-unit dimensions must have the same static sizes and
  // appear in the same order.
  while (inputDim < inputRank || outputDim < outputRank) {
    if (inputDim < inputRank && inputShape[inputDim] == 1) {
      ++inputDim;
      continue;
    }
    if (outputDim < outputRank && outputShape[outputDim] == 1) {
      ++outputDim;
      continue;
    }

    if (inputDim == inputRank || outputDim == outputRank)
      return std::nullopt;

    if (ShapedType::isDynamic(inputShape[inputDim]) ||
        ShapedType::isDynamic(outputShape[outputDim]) ||
        inputShape[inputDim] != outputShape[outputDim])
      return std::nullopt;

    mapping.push_back({inputDim, outputDim});
    ++inputDim;
    ++outputDim;
  }
  return mapping;
}

/// Checks statically known and constant indices accessed by a load from a
/// unit-dim insertion/removal reinterpret_cast to ensure in-bounds only access.
/// Fully dynamic indices are skipped (there is no way to verify them).
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

/// Rewrites `memref.load` through a reinterpret_cast that only inserts/removes
/// unit dimensions by mapping the load indices directly onto the source MemRef.
///
/// Shape restriction gated by getNonUnitDimMapping().
///
/// BEFORE (rank expansion)
///   %view = memref.reinterpret_cast %src
///     : memref<1xNxMxf32> to memref<Nx1x1xMxf32>
///   %v = memref.load %view[%i, %c0, %c0, %j] : memref<Nx1x1xMxf32>
///
/// AFTER
///   %v = memref.load %src[%c0, %i, %j] : memref<1xNxMxf32>
///
/// BEFORE (rank collapsing)
///   %view = memref.reinterpret_cast %src
///     : memref<Nx1x1xMxf32> to memref<1xNxMxf32>
///   %v = memref.load %view[%c0, %i, %j] : memref<1xNxMxf32>
///
/// AFTER
///   %v = memref.load %src[%i, %c0, %c0, %j] : memref<Nx1x1xMxf32>
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
    std::optional<NonUnitDimMapping> dimMapping = getNonUnitDimMapping(rc);
    if (!dimMapping)
      return rewriter.notifyMatchFailure(
          op, "reinterpret_cast is not a unit-dim insertion/removal preserving "
              "non-unit dimensions");

    assert(areIndicesInBounds(op) &&
           "load from reinterpret_cast indexes out of bounds!");

    auto rcInputTy = cast<MemRefType>(rc.getSource().getType());

    int64_t rcInputRank = rcInputTy.getRank();

    SmallVector<Value> oldIdxs(op.getIndices().begin(), op.getIndices().end());

    // Prefer reusing an explicit constant-zero index from the old load.
    Value zeroIndex;
    for (Value idx : oldIdxs) {
      std::optional<int64_t> idxVal = getConstantIndex(idx);
      if (idxVal && *idxVal == 0) {
        zeroIndex = idx;
        break;
      }
    }
    if (!zeroIndex)
      zeroIndex = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);

    // Initialize new load indices to all 0s.
    SmallVector<Value> rcInputIdxs(rcInputRank, zeroIndex);
    for (auto [inputDim, outputDim] : *dimMapping)
      rcInputIdxs[inputDim] = oldIdxs[outputDim];

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
      return !getNonUnitDimMapping(rc);
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
