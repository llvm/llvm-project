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

/// Non-unit reinterpret_cast result dimension and the source dimension it
/// advances through.
struct NonUnitDimAssocMapForRC {
  unsigned resultDimPos;
  unsigned sourceDimPos;
};

/// Copy-relevant information derived from a reinterpret_cast.
struct AssocMapAndOffsetsForRC {
  // Non-unit dimensions of the reinterpret_cast result.
  SmallVector<NonUnitDimAssocMapForRC> assocMap;
  // Delinearized offsets to in-bounds reinterpret_cast source indices.
  // Optional since it is only supported for static offsets.
  std::optional<SmallVector<int64_t>> delinearizedOffsets;
};

/// Maps each non-unit result dimension to a source dimension by stride. Returns
/// false if a stride of a non-unit dimension in the rc result is dynamic or rc
/// result strides are not equivalent to rc source strides.
static bool findSourceDimForResultDim(memref::ReinterpretCastOp rc,
                                      AssocMapAndOffsetsForRC &mapAndOffs) {
  MemRefType resType = dyn_cast<MemRefType>(rc.getType());
  MemRefType srcType = dyn_cast<MemRefType>(rc.getSource().getType());
  assert(srcType.getLayout().isIdentity() &&
         "Expecting identity source layout.");

  SmallVector<int64_t> srcIdentityStrides = computeStrides(srcType.getShape());
  ArrayRef<int64_t> rcResultStrides = rc.getStaticStrides();

  assert(srcType.getRank() == resType.getRank() &&
         "Expecting rank-preserving reinterpret_casts");

  for (auto [resultDim, resultSize] : llvm::enumerate(resType.getShape())) {
    if (resultSize == 1)
      continue;

    int64_t resultStride = rcResultStrides[resultDim];

    if (ShapedType::isDynamic(resultStride))
      return false;

    // For non-scalar strided memrefs, only support result strides identical to
    // the identity strides of the source. This enables direct indexing into the
    // same source dimensions, without linearization.
    if (resultStride != srcIdentityStrides[resultDim])
      return false;

    assert(srcType.getDimSize(resultDim) >= resultSize &&
           "reinterpret_cast result dimension does not fit in the matching "
           "source dimension");

    mapAndOffs.assocMap.push_back(NonUnitDimAssocMapForRC{
        static_cast<unsigned>(resultDim), static_cast<unsigned>(resultDim)});
  }

  return true;
}

/// Returns source indices for a static reinterpret_cast offset of an
/// identity-layout source.
static std::optional<SmallVector<int64_t>>
delinearizeStaticRCOffset(memref::ReinterpretCastOp rc) {
  ArrayRef<int64_t> rcOffsets = rc.getStaticOffsets();
  // FIXME: Despite what `getStaticOffsets` implies, `reinterpret_cast` takes
  // only a single offset. That should be fixed at the op definition level.
  assert(rcOffsets.size() == 1 && "Expecting single offset");
  assert(ShapedType::isStatic(rcOffsets[0]) && "expected static offset");

  assert(rcOffsets[0] >= 0 &&
         "static reinterpret_cast offset must be non-negative");

  MemRefType srcType = dyn_cast<MemRefType>(rc.getSource().getType());
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

/// Returns the unique non-unit dim or nullopt of # non-unit-dims != 1.
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

/// Returns the copy-relevant reinterpret_cast information: non-unit result
/// dimensions, their source-dimension mapping, and optional source starting
/// indices for a static offset.
///
/// Examples that return rewrite info:
///
///   // Scalar-shaped copy into a source with at most one non-unit dimension.
///   // There are no non-unit result dimensions, so strides in the strided
///   // memref do not affect index mapping.
///   copy memref<1 x ... x 1 x f32>
///     to reinterpret_cast memref<source-shape>
///       to memref<1 x ... x 1 x f32, strided<[?, ..., ?], offset: ?>>
///
///   // Effectively-1D copy. The single non-unit strided memref dimension is
///   // mapped to its matching identity-layout stride source dimension.
///   copy memref<1 x ... x N x ... x 1 x f32>
///     to reinterpret_cast memref<source-shape>
///       to memref<1 x ... x N x ... x 1 x f32, strided<[..., S, ...]>>
///
///   // Multidimensional copy with static offset. Each non-unit strided memref
///   // dimension is mapped in order, checking for matching source static
///   stride. copy memref<1 x ... x N_0 x ... x N_K x ... x 1 x f32>
///     to reinterpret_cast memref<source-shape>
///       to memref<1 x ... x N_0 x ... x N_K x ... x 1 x f32,
///                 strided<[..., S_0, ..., S_1, ...], offset: O>>
///
/// Examples that return no info:
///
///   // Non-scalar copies: dynamic strides in strided memref.
///   copy memref<1xNxf32>
///     to reinterpret_cast memref<1xNxMxf32>
///       to memref<1xNxf32, strided<[?, ?]>>
///
///   // Multidimensional copy with dynamic linear offset.
///   copy memref<1xNxKxf32>
///     to reinterpret_cast memref<1xNxMxf32>
///       to memref<1xNxKxf32, strided<[N*M, M, 1], offset: ?>>
static std::optional<AssocMapAndOffsetsForRC>
getAssocMapAndOffsetsForRC(memref::ReinterpretCastOp rc) {
  MemRefType srcType = dyn_cast<MemRefType>(rc.getSource().getType());
  MemRefType resType = dyn_cast<MemRefType>(rc.getType());

  // Ranked memref types are required to statically build load/store index
  // lists.
  if (!srcType || !resType)
    return std::nullopt;

  // TODO: Support rank-modifying reinterpret_casts
  if (srcType.getRank() != resType.getRank())
    return std::nullopt;

  // TODO: Support dynamic shapes with mixed size operands as loop bounds.
  if (!(srcType.hasStaticShape() && resType.hasStaticShape()))
    return std::nullopt;

  // TODO: Support non-identity source layouts by computing source strides from
  // the layout map.
  if (!srcType.getLayout().isIdentity())
    return std::nullopt;

  AssocMapAndOffsetsForRC mapAndOffs;

  assert(resType.hasStaticShape() && "expected static shape");
  // For scalar copies, result strides are irrelevant, including dynamic ones.
  // For non-scalar copies, require static result strides identical to the
  // identity strides of the reinterpret_cast source.
  if (!llvm::all_of(resType.getShape(),
                    [](int64_t size) { return size == 1; }) &&
      !findSourceDimForResultDim(rc, mapAndOffs))
    return std::nullopt;

  ArrayRef<int64_t> rcOffsets = rc.getStaticOffsets();
  // FIXME: Despite what `getStaticOffsets` implies, `reinterpret_cast` takes
  // only a single offset. That should be fixed at the op definition level.
  assert(rcOffsets.size() == 1 && "Expecting single offset");

  // CASE 1: Static ReinterpretCast offset
  if (ShapedType::isStatic(rcOffsets[0])) {
    // Delinearize static ReinterpretCast offset as in-bounds indices (one for
    // every source dimension).
    mapAndOffs.delinearizedOffsets = delinearizeStaticRCOffset(rc);
    assert(mapAndOffs.delinearizedOffsets &&
           "static reinterpret_cast offset must delinearize to in-bounds "
           "reinterpret_cast source indices");

    // Relevant for non-scalar copies: assert that the rectangular
    // copied slice is in bounds.
    assert(
        llvm::all_of(
            mapAndOffs.assocMap,
            [&](const NonUnitDimAssocMapForRC &assocMap) {
              return (*mapAndOffs.delinearizedOffsets)[assocMap.sourceDimPos] +
                         resType.getDimSize(assocMap.resultDimPos) <=
                     srcType.getDimSize(assocMap.sourceDimPos);
            }) &&
        "reinterpret_cast metadata describes an invalid accessible region");
    return mapAndOffs;
  }

  // CASE 2: Dynamic ReinterpretCast offset.
  // TODO: Support dynamic offsets into sources with multiple non-unit
  // dimensions by delinearizing the offset into source start indices at runtime
  // before adding loop IVs.

  // With an effectively-1D source, a dynamic linear offset can be used directly
  // as the index of the unique non-unit source dimension.
  if (!getSingleNonUnitDim(srcType))
    return std::nullopt;

  // Non-scalar copies require identical strides and no rank-changing,
  // so there can be at most one non-unit result dimension in this case.
  assert(mapAndOffs.assocMap.size() <= 1 &&
         "effectively-1D source cannot have multiple mapped non-unit dims");

  return mapAndOffs;
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

    std::optional<AssocMapAndOffsetsForRC> mapAndOffs =
        getAssocMapAndOffsetsForRC(rc);
    if (!mapAndOffs)
      return rewriter.notifyMatchFailure(
          op, "reinterpret_cast does not match scalar or loop copy region");

    Location loc = op.getLoc();
    Value dst = rc.getSource();
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
    upperBounds.reserve(mapAndOffs->assocMap.size());
    MemRefType rcResType = dyn_cast<MemRefType>(rc.getType());
    for (const NonUnitDimAssocMapForRC &assocMap : mapAndOffs->assocMap)
      upperBounds.push_back(getOrCreateIndexConstant(
          rcResType.getDimSize(assocMap.resultDimPos)));

    SmallVector<Value> rcSrcStoreIdxs = getZeroIdxs(dstType.getRank());
    std::optional<unsigned> srcNonUnitDimPos;
    // Static offset has been delinearized in function gating rewrite.
    if (mapAndOffs->delinearizedOffsets) {
      for (auto [idx, offset] :
           llvm::enumerate(*mapAndOffs->delinearizedOffsets)) {
        if (offset == 0)
          continue;
        rcSrcStoreIdxs[idx] = getOrCreateIndexConstant(offset);
      }
    } else {
      // Without runtime delinearization, use the dynamic offset directly only
      // when the source has a single non-unit dimension.
      assert(mapAndOffs->assocMap.size() <= 1 &&
             "Expecting single non-unit dimension mapping.");
      srcNonUnitDimPos = getSingleNonUnitDim(dstType);
      assert(srcNonUnitDimPos &&
             "Expecting single non-unit dimension source to receive the "
             "dynamic offset.");

      SmallVector<OpFoldResult> rcOffsets = rc.getMixedOffsets();
      // FIXME: Despite what `getMixedOffsets` implies, `reinterpret_cast` takes
      // only a single offset. That should be fixed at the op definition level.
      assert(rcOffsets.size() == 1 && "Expecting single offset");
      rcSrcStoreIdxs[*srcNonUnitDimPos] =
          getValueOrCreateConstantIndexOp(rewriter, loc, rcOffsets[0]);
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
      loopIvs.reserve(mapAndOffs->assocMap.size());

      // Build one nested loop per non-unit strided memref dimension.
      for (Value upperBound : upperBounds) {
        scf::ForOp loop =
            scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
        loopIvs.push_back(loop.getInductionVar());
        rewriter.setInsertionPointToStart(loop.getBody());
      }

      // Load indices are zero except for non-unit strided memref dimensions,
      // which use the corresponding loop induction variables.
      SmallVector<Value> loadIdxs = getZeroIdxs(cpSrcType.getRank());
      unsigned loopIndex = 0;
      for (const NonUnitDimAssocMapForRC &assocMap : mapAndOffs->assocMap)
        loadIdxs[assocMap.resultDimPos] = loopIvs[loopIndex++];

      // Store indices start from the offset-derived source indices. Add each
      // loop IV to the mapped source dimension.
      SmallVector<Value> storeIdxs(rcSrcStoreIdxs);
      loopIndex = 0;
      for (const NonUnitDimAssocMapForRC &assocMap : mapAndOffs->assocMap) {
        Value iv = loopIvs[loopIndex++];
        // Add each IV to one source index.
        if (storeIdxs[assocMap.sourceDimPos] == getOrCreateIndexConstant(0)) {
          storeIdxs[assocMap.sourceDimPos] = iv;
        } else {
          storeIdxs[assocMap.sourceDimPos] = arith::AddIOp::create(
              rewriter, loc, storeIdxs[assocMap.sourceDimPos], iv);
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

  // The base and result must either both have a single non-unit dimension
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
/// mapping the load indices directly onto the base MemRef.

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
      // Pattern applies only when the copy source shape is static and the
      // reinterpret_cast result can be mapped back to base memref indices.
      MemRefType cpSrcType = dyn_cast<MemRefType>(op.getSource().getType());
      return !(cpSrcType && cpSrcType.hasStaticShape() &&
               getAssocMapAndOffsetsForRC(rc));
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
