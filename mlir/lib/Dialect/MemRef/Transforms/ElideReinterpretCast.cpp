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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include <array>
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

//===----------------------------------------------------------------------===//
// Copy Rewrite Helpers
//===----------------------------------------------------------------------===//

/// Copy-relevant information derived from a reinterpret_cast.
struct ResultNonUnitDimsAndOffsetsForRC {
  // Non-unit dimensions of the reinterpret_cast result.
  SmallVector<unsigned> nonUnitDimsPos;
  // Delinearized offsets to in-bounds reinterpret_cast source indices.
  // Optional since it is only supported for static offsets.
  std::optional<SmallVector<int64_t>> delinearizedOffsets;
};

/// Returns delinearized offset indices for a static reinterpret_cast offset of
/// an identity-layout source.
static std::optional<SmallVector<int64_t>>
delinearizeStaticRCOffset(memref::ReinterpretCastOp rc) {
  ArrayRef<int64_t> rcOffsets = rc.getStaticOffsets();
  MemRefType srcType = dyn_cast<MemRefType>(rc.getSource().getType());
  // FIXME: Despite what `getStaticOffsets` implies, `reinterpret_cast` takes
  // only a single offset. That should be fixed at the op definition level.
  assert(rcOffsets.size() == 1 && "Expecting single offset");

  assert(ShapedType::isStatic(rcOffsets[0]) && "expected static offset");
  assert(rcOffsets[0] >= 0 &&
         "static reinterpret_cast offset must be non-negative");
  assert(srcType.getLayout().isIdentity() &&
         "Expecting identity source layout.");
  if (srcType.getRank() == 0) {
    assert(rcOffsets[0] == 0 &&
           "non-zero static offset is invalid for rank-0 source memref");
    return SmallVector<int64_t>{};
  }

  SmallVector<int64_t> offsetIdxs(srcType.getRank(), 0);
  int64_t remainder = rcOffsets[0];
  SmallVector<int64_t> srcStrides = computeStrides(srcType.getShape());
  // Convert the scalar reinterpret_cast offset to per-dimension source starting
  // indices.
  for (auto [dim, stride] : llvm::enumerate(srcStrides)) {
    offsetIdxs[dim] = remainder / stride;
    assert(offsetIdxs[dim] < srcType.getDimSize(dim) &&
           "static reinterpret_cast offset must delinearize to in-bounds "
           "source indices");
    remainder %= stride;
  }

  assert(remainder == 0 &&
         "Assuming identity source layout, the trailing stride == 1 "
         "so, the remainder should be 0 at the end of index calculation.");
  return offsetIdxs;
}

static bool hasExactlyOneCollapsedNonUnitDim(memref::ReinterpretCastOp rc) {
  MemRefType srcType = dyn_cast<MemRefType>(rc.getSource().getType());
  MemRefType resType = dyn_cast<MemRefType>(rc.getType());
  assert(srcType.hasStaticShape() && resType.hasStaticShape() &&
         "expected static shapes");
  assert(srcType.getRank() == resType.getRank() &&
         "expected rank-preserving reinterpret_cast");

  unsigned collapsedDims = 0;

  for (auto [srcSize, resSize] :
       llvm::zip_equal(srcType.getShape(), resType.getShape())) {
    if (srcSize == resSize)
      continue;

    // Only allow collapsing one non-unit source dim to a unit result dim.
    if (srcSize != 1 && resSize == 1) {
      ++collapsedDims;
      continue;
    }

    // The sizes differ and both of them are non-unit - ATM not supported.
    return false;
  }

  // Make sure there is only one collapsed dimension.
  return collapsedDims == 1;
}

/// Returns the unique non-unit dim or nullopt if # non-unit-dims != 1.
static std::optional<unsigned> getSingleNonUnitDim(MemRefType type) {
  assert(type.hasStaticShape() && "expected static shape");
  ArrayRef<int64_t> shape = type.getShape();

  // Find all non-unit dims
  auto nonUnitDims = llvm::make_filter_range(
      llvm::enumerate(shape), [](auto it) { return it.value() != 1; });

  // Expect single non-unit dims
  if (llvm::range_size(nonUnitDims) != 1)
    return std::nullopt;

  // Return the index of the unique non-unit dim.
  return (*nonUnitDims.begin()).index();
}

/// Returns reinterpret_cast's result non-unit dimensions and, for static
/// offsets, delinearized offset.
///
/// Supports ranked, static-shape, rank-preserving reinterpret_casts from
/// identity-layout sources. In addition:
///     identical to the source identity strides, and exactly one non-unit
///     source
///  * Non-scalar results must have static offsets, static result strides
///     dimension collapsed to unit size
/// Scalar-shaped results may have arbitrary result strides (i.e. for scalars,
/// strides are effectively irrelevant).
///
/// Returns nullopt for unsupported
/// reinterpret_casts.
///
/// Examples that return info:
///
///   reinterpret_cast memref<1xMxNxf32, identity-layout>
///     to memref<1xMx1xf32, strided<[M*N, N, 1], offset: OFF>>
///
///   reinterpret_cast memref<1xMxf32, identity-layout>
///     to memref<1x1xf32, strided<[?, ?], offset: ?>>
///
/// Examples that return no info:
///
///   reinterpret_cast memref<1xMxNxf32, identity-layout>
///     to memref<1xMx1xf32, strided<[?, N, 1]>>
///
///   reinterpret_cast memref<1xMxNxf32, identity-layout>
///     to memref<1xKx1xf32, strided<[M*N, N, 1], offset: OFF>>
static std::optional<ResultNonUnitDimsAndOffsetsForRC>
getResultNonUnitDimsAndOffsetsForRC(memref::ReinterpretCastOp rc) {
  MemRefType srcType = dyn_cast<MemRefType>(rc.getSource().getType());
  MemRefType resType = dyn_cast<MemRefType>(rc.getType());

  // Ranked memref types are required to statically build load/store index
  // lists.
  if (!srcType || !resType)
    return std::nullopt;

  // TODO: Support rank-modifying reinterpret_casts.
  if (srcType.getRank() != resType.getRank())
    return std::nullopt;

  // TODO: Support dynamic shapes with mixed size operands as loop bounds.
  if (!(srcType.hasStaticShape() && resType.hasStaticShape()))
    return std::nullopt;

  // TODO: Support non-identity source layouts by computing source strides from
  // the layout map.
  if (!srcType.getLayout().isIdentity())
    return std::nullopt;

  ResultNonUnitDimsAndOffsetsForRC dimsAndOffs;

  // Track non-unit result dimensions; unit dimensions are
  // always indexed at 0.
  for (auto [dim, resultSize] : llvm::enumerate(resType.getShape())) {
    if (resultSize != 1)
      dimsAndOffs.nonUnitDimsPos.push_back(static_cast<unsigned>(dim));
  }

  ArrayRef<int64_t> rcOffsets = rc.getStaticOffsets();
  // FIXME: Despite what `getStaticOffsets` implies, `reinterpret_cast` takes
  // only a single offset. That should be fixed at the op definition level.
  assert(rcOffsets.size() == 1 && "Expecting single offset");

  bool isScalarRes =
      llvm::all_of(resType.getShape(), [](int64_t size) { return size == 1; });

  bool isOffsetDynamic = ShapedType::isDynamic(rcOffsets[0]);

  // Cases with at least one non-unit dimension in reinterpret_cast's result are
  // restricted to preserving all but one dimension from the source, which
  // collapsed to `1` in the result, and fully static metadata.
  if (!isScalarRes) {
    if (isOffsetDynamic)
      return std::nullopt;

    SmallVector<int64_t> srcIdentityStrides =
        computeStrides(srcType.getShape());
    ArrayRef<int64_t> rcResultStrides = rc.getStaticStrides();

    if (!llvm::all_of(llvm::zip_equal(srcIdentityStrides, rcResultStrides),
                      [](auto pair) {
                        auto [srcStride, resultStride] = pair;
                        return !ShapedType::isDynamic(resultStride) &&
                               srcStride == resultStride;
                      }))
      return std::nullopt;

    if (!hasExactlyOneCollapsedNonUnitDim(rc))
      return std::nullopt;
  }

  // CASE 1: Dynamic ReinterpretCast offset.
  //
  // Dynamic offsets are supported only for effectively-1D to scalar
  // reinterpret_casts.
  if (isOffsetDynamic) {
    // With an effectively-1D source, a dynamic offset can be mapped to its
    // unique non-unit dim. For other cases, bail out.
    if (llvm::count_if(srcType.getShape(),
                       [](int64_t size) { return size != 1; }) != 1)
      return std::nullopt;

    return dimsAndOffs;
  }

  // CASE 2: Static ReinterpretCast offset
  // Delinearize static ReinterpretCast offset as in-bounds indices (one for
  // every source dimension).
  dimsAndOffs.delinearizedOffsets = delinearizeStaticRCOffset(rc);

  return dimsAndOffs;
}

/// Rewrites supported copy operations through `memref.reinterpret_cast` to
/// scalar load/store operations.
///
/// Supported cases:
///   1. Scalar-shaped reinterpret_cast results. Result strides are ignored;
///      the store index is derived from the reinterpret_cast offset.
///
///   2. Non-scalar reinterpret_cast results that preserve all non-unit source
///      dimensions except one collapsed-to-unit dimension. Result strides must
///      be static and identical to the identity strides of the source, and the
///      static offset selects the collapsed dimension.
///
///   // BEFORE (scalar-shaped result)
///   %strided = memref.reinterpret_cast %dst
///     to offset: [OFF], sizes: [1, ..., 1], strides: [...]
///   memref.copy %src, %strided
///
///   // AFTER
///   %v = memref.load %src[0, ..., 0]
///   memref.store %v, %dst[delinearized(OFF)]
///
///   // BEFORE (one collapsed non-unit dimension)
///   %strided = memref.reinterpret_cast %dst
///     to offset: [OFF], sizes: [1, M, 1], strides: [M*N, N, 1]
///     : memref<1xMxNxf32>
///       to memref<1xMx1xf32, strided<[M*N, N, 1], offset: OFF>>
///   memref.copy %src, %strided
///
///   // AFTER
///   // Assuming OFF delinearizes to [0, 0, OFF]:
///   scf.for %i = 0 to M step 1 {
///     %v = memref.load %src[0, %i, 0]
///     memref.store %v, %dst[0, %i, OFF]
///   }
struct CopyToLoadAndStore : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const final {
    Value src = op.getSource();
    MemRefType cpSrcType = cast<MemRefType>(src.getType());
    if (!cpSrcType || !cpSrcType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "only ranked, static copy sources are supported.");

    Value rcOutput = op.getTarget();
    auto rc = rcOutput.getDefiningOp<memref::ReinterpretCastOp>();
    if (!rc)
      return rewriter.notifyMatchFailure(
          op, "target is not a memref.reinterpret_cast");

    std::optional<ResultNonUnitDimsAndOffsetsForRC> dimsAndOffs =
        getResultNonUnitDimsAndOffsetsForRC(rc);
    if (!dimsAndOffs)
      return rewriter.notifyMatchFailure(
          op,
          "unsupported reinterpret_cast result dimensions, strides, or offset");

    Location loc = op.getLoc();
    Value dst = rc.getSource();
    MemRefType dstType = cast<MemRefType>(dst.getType());
    MemRefType rcResType = cast<MemRefType>(rc.getType());

    // Sanity check that the copy doesn't access strided MemRef out-of-bounds.
    // Such cases should probably be rejected by Op verifier.
    // FIXME: Add run-time verification for cases like this.
    if (ShapedType::isStatic(rc.getStaticOffsets()[0]) &&
        llvm::any_of(llvm::enumerate(rcResType.getShape()), [&](auto it) {
          unsigned dim = it.index();
          int64_t rcResultSize = it.value();
          return (*dimsAndOffs->delinearizedOffsets)[dim] + rcResultSize >
                 dstType.getDimSize(dim);
        }))
      return rewriter.notifyMatchFailure(op, "copy accesses are OOB");

    // Constant Op cache to reuse common index constants across bounds, steps,
    // and static offsets: 0 is stored at index 0 and 1 is stored at index 1.
    std::array<Value, 2> cachedIndexConstants;
    auto getOrCreateIndexConstant = [&](int64_t value) -> Value {
      if (value == 0 || value == 1) {
        Value &cached = cachedIndexConstants[value];
        if (!cached)
          cached = arith::ConstantIndexOp::create(rewriter, loc, value);
        return cached;
      }
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    };

    auto getZeroIdxs = [&](int64_t rank) {
      SmallVector<Value> idxs;
      idxs.reserve(rank);
      if (rank != 0)
        idxs.append(rank, getOrCreateIndexConstant(0));
      return idxs;
    };

    // Create loop bounds before moving the insertion point into the loop nest,
    // so loop-invariant constants are emitted outside the generated loops.
    SmallVector<Value> upperBounds;
    upperBounds.reserve(dimsAndOffs->nonUnitDimsPos.size());
    for (unsigned dim : dimsAndOffs->nonUnitDimsPos) {
      upperBounds.push_back(
          getOrCreateIndexConstant(rcResType.getDimSize(dim)));
    }

    // All indices are initialised to zero.
    SmallVector<Value> rcSrcStoreIdxs = getZeroIdxs(dstType.getRank());
    std::optional<unsigned> srcNonUnitDimPos;
    if (dimsAndOffs->delinearizedOffsets) {
      // Initialize store indices from the static reinterpret_cast offset,
      // delinearized in function gating rewrite.
      for (auto [idx, offset] :
           llvm::enumerate(*dimsAndOffs->delinearizedOffsets)) {
        if (offset == 0)
          continue;
        rcSrcStoreIdxs[idx] = getOrCreateIndexConstant(offset);
      }
    } else {
      // Dynamic offset is used directly only for effectively-1D sources.
      assert(dimsAndOffs->nonUnitDimsPos.size() <= 1 &&
             "Expecting at most one non-unit result dimension.");

      srcNonUnitDimPos = getSingleNonUnitDim(dstType);
      assert(srcNonUnitDimPos &&
             "Expecting single non-unit dimension source to receive the "
             "dynamic offset.");

      SmallVector<OpFoldResult> rcOffsets = rc.getMixedOffsets();
      // FIXME: Despite what `getMixedOffsets` implies, `reinterpret_cast` takes
      // only a single offset. That should be fixed at the op definition level.
      assert(rcOffsets.size() == 1 && "Expecting single offset");
      // Only the index corresponding to the single non-unit dim is updated.
      rcSrcStoreIdxs[*srcNonUnitDimPos] =
          getValueOrCreateConstantIndexOp(rewriter, loc, rcOffsets[0]);
    }

    // Create the loop nest and emit the load/store at the innermost insertion
    // point.
    {
      OpBuilder::InsertionGuard guard(rewriter);

      SmallVector<Value> loadIdxs = getZeroIdxs(cpSrcType.getRank());
      SmallVector<Value> storeIdxs(rcSrcStoreIdxs);

      if (!dimsAndOffs->nonUnitDimsPos.empty()) {
        Value lowerBound = getOrCreateIndexConstant(0);
        Value step = getOrCreateIndexConstant(1);

        // Build one nested loop per non-unit reinterpret_cast result dimension.
        for (auto [loopIndex, dim] :
             llvm::enumerate(dimsAndOffs->nonUnitDimsPos)) {
          scf::ForOp loop = scf::ForOp::create(rewriter, loc, lowerBound,
                                               upperBounds[loopIndex], step);

          rewriter.setInsertionPointToStart(loop.getBody());

          Value iv = loop.getInductionVar();
          // Since result strides match source identity strides dimension-wise,
          // each IV indexes the same dimension in both the copy source and rc
          // source.
          loadIdxs[dim] = iv;

          if (storeIdxs[dim] == getOrCreateIndexConstant(0)) {
            storeIdxs[dim] = iv;
          } else {
            storeIdxs[dim] =
                arith::AddIOp::create(rewriter, loc, storeIdxs[dim], iv);
          }
        }
      }

      // Emit the scalar load/store at the innermost loop body, or directly at
      // the original copy location for scalar copies.
      Value val = memref::LoadOp::create(rewriter, loc, src, loadIdxs);
      memref::StoreOp::create(rewriter, loc, val, dst, storeIdxs);
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
      // Pattern applies only when the copy source shape is static and the
      // reinterpret_cast result can be mapped back to base memref indices.
      MemRefType cpSrcType = dyn_cast<MemRefType>(op.getSource().getType());
      return !(cpSrcType && cpSrcType.hasStaticShape() &&
               getResultNonUnitDimsAndOffsetsForRC(rc));
    });
    target.addDynamicallyLegalOp<memref::LoadOp>([](memref::LoadOp op) {
      auto rc = op.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
      if (!rc)
        return true;
      return !getNonUnitDimMapping(rc);
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
