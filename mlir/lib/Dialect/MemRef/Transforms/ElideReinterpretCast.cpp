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

/// Returns row-major strides for static identity-layout memref type.
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

/// Finds the source dimension for a static reinterpret_cast result dimension.
/// Dimensions marked in `usedSourceDims` are skipped. Returns the smallest
/// source dimension whose size is at least the result dimension size, with the
/// same stride.
static std::optional<unsigned> findSourceDimForResultDim(
    memref::ReinterpretCastOp rc, unsigned resultDim, MemRefType sourceType,
    ArrayRef<int64_t> sourceStrides, ArrayRef<bool> usedSourceDims) {
  MemRefType resultType = cast<MemRefType>(rc.getType());
  assert(resultDim < resultType.getRank() && "result dimension out of range");
  assert(sourceType.getRank() == static_cast<int64_t>(sourceStrides.size()) &&
         sourceStrides.size() == usedSourceDims.size() &&
         "expected same-rank source type, strides, and used-dimension mask");
  assert(!ShapedType::isDynamic(rc.getStaticStrides()[resultDim]) &&
         "expected static result stride");

  int64_t resultStride = rc.getStaticStrides()[resultDim];
  int64_t resultSize = resultType.getDimSize(resultDim);
  std::optional<unsigned> sourceDim;
  for (auto [idx, stride] : llvm::enumerate(sourceStrides)) {
    if (usedSourceDims[idx] || stride != resultStride ||
        sourceType.getDimSize(idx) < resultSize)
      continue;

    if (!sourceDim ||
        sourceType.getDimSize(idx) < sourceType.getDimSize(*sourceDim))
      sourceDim = idx;
  }
  return sourceDim;
}

/// Returns source indices for a static reinterpret_cast offset.
static std::optional<SmallVector<int64_t>>
delinearizeStaticOffset(memref::ReinterpretCastOp rc, MemRefType sourceType,
                        ArrayRef<int64_t> sourceStrides) {
  ArrayRef<int64_t> offsets = rc.getStaticOffsets();
  // FIXME: Despite what `getStaticOffsets` implies, `reinterpret_cast` takes
  // only a single offset. That should be fixed at the op definition level.
  assert(offsets.size() == 1 && "Expecting single offset");
  assert(!ShapedType::isDynamic(offsets[0]) && "expected static offset");

  if (offsets[0] < 0)
    return std::nullopt;

  SmallVector<int64_t> indices(sourceType.getRank(), 0);
  int64_t remainder = offsets[0];
  for (auto [idx, stride] : llvm::enumerate(sourceStrides)) {
    indices[idx] = remainder / stride;
    if (indices[idx] >= sourceType.getDimSize(idx))
      return std::nullopt;
    remainder %= stride;
  }

  if (remainder != 0)
    return std::nullopt;
  return indices;
}

/// Returns the dimension whose static size is not one if it is unique.
static std::optional<unsigned> getSingleNonUnitDim(MemRefType type) {
  assert(type.hasStaticShape() && "expected static shape");
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.empty())
    return std::nullopt;

  std::optional<unsigned> nonUnitDim;
  for (auto [idx, dim] : llvm::enumerate(shape)) {
    if (dim == 1)
      continue;
    if (nonUnitDim)
      return std::nullopt;
    nonUnitDim = idx;
  }
  return nonUnitDim;
}

/// Per-dimension loop nest info.
struct CopyLoopDimInfo {
  unsigned copyDim;
  unsigned baseDim;
  int64_t size;
};

/// Rewrite info from reinterpret_cast layout, captured after passing legality
/// checks.
struct CopyFromReinterCastInfo {
  SmallVector<CopyLoopDimInfo> loopDims;
  std::optional<SmallVector<int64_t>> staticOffsetIndices;
  std::optional<unsigned> dynamicOffsetDim;
};

/// Builds the index mapping needed to replace a copy into a reinterpret_cast
/// strided memref with scalar stores into the reinterpret_cast base.
///
/// Examples that return rewrite info:
///
///   // Scalar-shaped copy. There are no copied non-unit dimensions, so dynamic
///   // strides in the strided memref do not affect index mapping.
///   copy memref<1 x ... x 1 x f32>
///     to reinterpret_cast memref<base-shape>
///       to memref<1 x ... x 1 x f32, strided<[?, ..., ?], offset: ?>>
///
///   // Effectively-1D copy. The single non-unit strided memref dimension is
///   // mapped to an identity-layout base dimension by its static stride.
///   copy memref<1 x ... x N x ... x 1 x f32>
///     to reinterpret_cast memref<base-shape>
///       to memref<1 x ... x N x ... x 1 x f32, strided<[..., S, ...]>>
///
///   // Multidimensional copy with static offset. Each non-unit strided memref
///   // dimension is mapped independently by its static stride.
///   copy memref<1 x ... x N_0 x ... x N_K x ... x 1 x f32>
///     to reinterpret_cast memref<base-shape>
///       to memref<1 x ... x N_0 x ... x N_K x ... x 1 x f32,
///                 strided<[..., S_0, ..., S_1, ...], offset: O>>
///
/// Examples that return no info:
///
///   // Dynamic stride on a copied strided memref dimension.
///   copy memref<1xNxf32>
///     to reinterpret_cast memref<1xNxMxf32>
///       to memref<1xNxf32, strided<[?, ?]>>
///
///   // Multidimensional copy with dynamic linear offset.
///   copy memref<1xNxKxf32>
///     to reinterpret_cast memref<1xNxMxf32>
///       to memref<1xNxKxf32, strided<[N*M, M, 1], offset: ?>>
static std::optional<CopyFromReinterCastInfo>
getCopyFromReinterCastInfo(memref::CopyOp op, memref::ReinterpretCastOp rc) {
  MemRefType srcType = dyn_cast<MemRefType>(op.getSource().getType());
  MemRefType baseType = dyn_cast<MemRefType>(rc.getSource().getType());
  MemRefType resultType = dyn_cast<MemRefType>(rc.getType());

  // Ranked memref types are required to statically build load/store index
  // lists.
  if (!srcType || !baseType || !resultType)
    return std::nullopt;

  if (srcType.getShape() != resultType.getShape())
    return std::nullopt;

  // TODO: Support rank-changing reinterpret_casts by converting the
  // strided memref indices to base indices. For example, a copy to
  // a strided memref<2x3xf32> of base memref<6xf32> needs to linearize the
  // strided memref indices as `i * 3 + j`, then combine that with the
  // reinterpret_cast offset before indexing the rank-1 base memref.
  if (baseType.getRank() != resultType.getRank())
    return std::nullopt;

  // TODO: Support dynamic shapes with mixed size operands as loop bounds.
  if (!(srcType.hasStaticShape() && baseType.hasStaticShape() &&
        resultType.hasStaticShape()))
    return std::nullopt;

  std::optional<SmallVector<int64_t>> baseStrides =
      getIdentityStrides(baseType);
  // TODO: Support non-identity source layouts by computing source strides from
  // the layout map.
  if (!baseStrides)
    return std::nullopt;

  CopyFromReinterCastInfo info;
  SmallVector<bool> usedBaseDims(baseType.getRank(), false);

  for (auto [resultDim, resultSize] : llvm::enumerate(resultType.getShape())) {
    if (resultSize == 1)
      continue;

    // TODO: Support dynamic strides on copied dimensions.
    if (ShapedType::isDynamic(rc.getStaticStrides()[resultDim]))
      return std::nullopt;

    // Each copied result dimension must map by stride to a source dimension
    // whose static size >= result dimension size.
    std::optional<unsigned> baseDim =
        findSourceDimForResultDim(rc, static_cast<unsigned>(resultDim),
                                  baseType, *baseStrides, usedBaseDims);
    assert(baseDim &&
           "static reinterpret_cast stride must map to an identity base "
           "dimension");

    usedBaseDims[*baseDim] = true;
    info.loopDims.push_back(CopyLoopDimInfo{static_cast<unsigned>(resultDim),
                                            *baseDim, resultSize});
  }

  ArrayRef<int64_t> offsets = rc.getStaticOffsets();
  // FIXME: Despite what `getStaticOffsets` implies, `reinterpret_cast` takes
  // only a single offset. That should be fixed at the op definition level.
  assert(offsets.size() == 1 && "Expecting single offset");
  if (!ShapedType::isDynamic(offsets[0])) {
    // Static offset must delinearize to in-bounds source indices.
    std::optional<SmallVector<int64_t>> offsetIndices =
        delinearizeStaticOffset(rc, baseType, *baseStrides);
    assert(offsetIndices &&
           "static reinterpret_cast offset must delinearize to in-bounds base "
           "indices");

    for (const CopyLoopDimInfo &loopDim : info.loopDims) {
      assert((*offsetIndices)[loopDim.baseDim] + loopDim.size <=
                 baseType.getDimSize(loopDim.baseDim) &&
             "reinterpret_cast metadata describes an invalid accessible "
             "region");
    }
    info.staticOffsetIndices = std::move(offsetIndices);
    return info;
  }

  // Dynamic offsets are kept only when they can be used as a single base index.
  // TODO: Support dynamic offsets for copies with multiple loop dimensions by
  // delinearizing the offset into base start indices at runtime before adding
  // loop IVs.
  if (info.loopDims.size() > 1)
    return std::nullopt;

  if (info.loopDims.empty()) {
    // Dynamic scalar offsets cannot be delinearized statically. They can be
    // used directly only when the base has a single non-unit dimension to
    // receive them.
    std::optional<unsigned> nonUnitDim = getSingleNonUnitDim(baseType);
    // TODO: Support scalar dynamic offsets into bases with multiple non-unit
    // dimensions by delinearizing the single accessed element offset at
    // runtime.
    if (!nonUnitDim)
      return std::nullopt;

    info.dynamicOffsetDim = *nonUnitDim;
    return info;
  }

  unsigned baseDim = info.loopDims.front().baseDim;
  info.dynamicOffsetDim =
      (*baseStrides)[baseDim] == 1 ? baseDim : baseStrides->size() - 1;
  return info;
}

/// Rewrites supported copy operations through `memref.reinterpret_cast` to
/// scalar load/store operations.
///
///   // BEFORE (scalar copy)
///   %strided = memref.reinterpret_cast %dst
///     to offset: [O], sizes: [1, ..., 1], strides: [...]
///   memref.copy %src, %strided
///
///   // AFTER
///   %v = memref.load %src[0, ..., 0]
///   memref.store %v, %dst[delinearized(O)]
///
///   // BEFORE (effectively-1D copy)
///   %strided = memref.reinterpret_cast %dst
///     to offset: [O], sizes: [1, N, 1], strides: [...]
///   memref.copy %src, %strided
///
///   // AFTER
///   scf.for %i = 0 to N step 1 {
///     %v = memref.load %src[0, %i, 0]
///     memref.store %v, %dst[delinearized(O) + mapped(%i)]
///   }
///
///   // BEFORE (multidimensional copy with static offset)
///   %strided = memref.reinterpret_cast %dst
///     to offset: [O], sizes: [1, N, K], strides: [...]
///   memref.copy %src, %strided
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

    std::optional<CopyFromReinterCastInfo> copyInfo =
        getCopyFromReinterCastInfo(op, rc);
    if (!copyInfo)
      return rewriter.notifyMatchFailure(
          op, "reinterpret_cast does not match scalar or loop copy region");

    Location loc = op.getLoc();
    Value src = op.getSource();
    Value dst = rc.getSource();

    MemRefType srcType = cast<MemRefType>(src.getType());
    MemRefType dstType = cast<MemRefType>(dst.getType());

    // Reuse common index constants across bounds, steps, and static offsets,
    // but avoid creating them for rank-0 copies.
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
    auto getZeroIndices = [&](int64_t rank) {
      SmallVector<Value> indices;
      indices.reserve(rank);
      if (rank != 0)
        indices.append(rank, getOrCreateIndexConstant(0));
      return indices;
    };

    // Create all loop bounds before building the loop nest. Otherwise an
    // inner-loop bound can be inserted inside an outer loop body.
    SmallVector<Value> upperBounds;
    upperBounds.reserve(copyInfo->loopDims.size());
    for (const CopyLoopDimInfo &loopDim : copyInfo->loopDims)
      upperBounds.push_back(getOrCreateIndexConstant(loopDim.size));

    SmallVector<Value> baseStoreIndices = getZeroIndices(dstType.getRank());
    // Static offsets were already delinearized into base indices. Fill the
    // non-zero starting indices before creating loop bodies.
    if (copyInfo->staticOffsetIndices) {
      for (auto [idx, offset] :
           llvm::enumerate(*copyInfo->staticOffsetIndices)) {
        if (offset == 0)
          continue;
        baseStoreIndices[idx] = getOrCreateIndexConstant(offset);
      }
    } else {
      // Supported dynamic offsets are used directly in exactly one base
      // dimension selected by getCopyFromReinterCastInfo.
      assert(copyInfo->dynamicOffsetDim &&
             "expected dynamic offset dimension for dynamic offset");
      SmallVector<OpFoldResult> offsets = rc.getMixedOffsets();
      // FIXME: Despite what `getMixedOffsets` implies, `reinterpret_cast` takes
      // only a single offset. That should be fixed at the op definition level.
      assert(offsets.size() == 1 && "Expecting single offset");
      baseStoreIndices[*copyInfo->dynamicOffsetDim] =
          getValueOrCreateConstantIndexOp(rewriter, loc, offsets[0]);
    }

    // Scope for OpBuilder::InsertionGuard.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Value lowerBound;
      Value step;
      if (!upperBounds.empty()) {
        lowerBound = getOrCreateIndexConstant(0);
        step = getOrCreateIndexConstant(1);
      }

      SmallVector<Value> loopIvs;
      loopIvs.reserve(copyInfo->loopDims.size());

      // Build one nested loop per non-unit copied strided memref dimension.
      for (Value upperBound : upperBounds) {
        scf::ForOp loop =
            scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
        loopIvs.push_back(loop.getInductionVar());
        rewriter.setInsertionPointToStart(loop.getBody());
      }

      // Load indices are zero except for copied strided memref dimensions,
      // which use the corresponding loop induction variables.
      SmallVector<Value> loadIndices = getZeroIndices(srcType.getRank());
      unsigned loopIndex = 0;
      for (const CopyLoopDimInfo &loopDim : copyInfo->loopDims)
        loadIndices[loopDim.copyDim] = loopIvs[loopIndex++];

      // Store indices start from the offset-derived base indices. Add each loop
      // IV to the mapped base dimension.
      SmallVector<Value> storeIndices(baseStoreIndices);
      loopIndex = 0;
      for (const CopyLoopDimInfo &loopDim : copyInfo->loopDims) {
        Value iv = loopIvs[loopIndex++];
        if (storeIndices[loopDim.baseDim] == getOrCreateIndexConstant(0)) {
          storeIndices[loopDim.baseDim] = iv;
        } else {
          storeIndices[loopDim.baseDim] = arith::AddIOp::create(
              rewriter, loc, storeIndices[loopDim.baseDim], iv);
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
      return !getCopyFromReinterCastInfo(op, rc);
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
