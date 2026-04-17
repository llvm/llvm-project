//===- Vectorization.cpp - Implementation of linalg Vectorization ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect Vectorization transformations.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/Utils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Interfaces/MaskableOpInterface.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-vectorization"

/// Try to vectorize `convOp` as a convolution.
static FailureOr<Operation *>
vectorizeConvolution(RewriterBase &rewriter, LinalgOp convOp,
                     ArrayRef<int64_t> inputVecSizes = {},
                     ArrayRef<bool> inputVecScalableFlags = {},
                     bool flatten1DDepthwiseConv = false);

/// Vectorize tensor::InsertSliceOp with:
///   * vector::TransferReadOp + vector::TransferWriteOp
/// The vector sizes are either:
///   * user-provided in `inputVectorSizes`, or
///   * inferred from the static dims in the input and output tensors.
/// Bails out if:
///   * vector sizes are not user-provided, and
///   * at least one dim is dynamic (in both the input and output tensors).
///
/// Before:
///     !t_in_type = tensor<1x2x3xf32>
///     !t_out_type = tensor<9x8x7x1x2x3xf32>
///     !v_type = vector<1x2x3xf32>
///     %inserted_slice = tensor.insert_slice %src into %dest ... : !t_in_type
///     into !t_out_type
/// After:
///     %read = vector.transfer_read %src[...], %pad ... : !t_in_type, !v_type
///     %write = vector.transfer_write %read, %dest ... : !v_type, !t_out_type
static LogicalResult
vectorizeAsInsertSliceOp(RewriterBase &rewriter, tensor::InsertSliceOp sliceOp,
                         ArrayRef<int64_t> inputVectorSizes,
                         SmallVectorImpl<Value> &newResults);

/// Returns the effective Pad value for the input op, provided it's a scalar.
///
/// Many Ops exhibit pad-like behaviour, but this isn't always explicit. If
/// this Op performs padding, retrieve the padding value provided that it's
/// a scalar and static/fixed for all the padded values. Returns an empty value
/// otherwise.
static Value getStaticPadVal(Operation *op);

/// Return the unique instance of OpType in `block` if it is indeed unique.
/// Return null if none or more than 1 instances exist.
template <typename OpType>
static OpType getSingleOpOfType(Block &block) {
  OpType res;
  block.walk([&](OpType op) {
    if (res) {
      res = nullptr;
      return WalkResult::interrupt();
    }
    res = op;
    return WalkResult::advance();
  });
  return res;
}

/// Layout of the canonical 1D conv/pool form:
///   - Scalar:    no batch, no channels (e.g. 1D non-channeled conv).
///   - Batched:   [n, w, c] / [n, w, f] (e.g. NWC conv).
///   - Batchless: [w, c]    / [w, f]    (no batch dim).
enum class ConvLayoutKind { Scalar, Batched, Batchless };

/// Extract input slices after unrolling the filter along kw.
///
/// Per-tile shape and offset depend on `layout`:
///   Scalar:    {wSizeStep}            @ [w + kw]
///   Batched:   {n, wSizeStep, c}      @ [0, sw*w + dw*kw, 0]
///   Batchless: {wSizeStep, c}         @ [sw*w + dw*kw, 0]
static SmallVector<Value>
extractConvInputSlices(RewriterBase &rewriter, Location loc, Value input,
                       int64_t nSize, int64_t wSize, int64_t cSize,
                       int64_t kwSize, int strideW, int dilationW,
                       int64_t wSizeStep, ConvLayoutKind layout) {
  SmallVector<Value> result;
  for (int64_t kw = 0; kw < kwSize; ++kw) {
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      int64_t offset = w * strideW + kw * dilationW;
      SmallVector<int64_t> sizes, offsets;
      switch (layout) {
      case ConvLayoutKind::Scalar:
        sizes = {wSizeStep};
        offsets = {w + kw};
        break;
      case ConvLayoutKind::Batched:
        sizes = {nSize, wSizeStep, cSize};
        offsets = {0, offset, 0};
        break;
      case ConvLayoutKind::Batchless:
        sizes = {wSizeStep, cSize};
        offsets = {offset, 0};
        break;
      }
      SmallVector<int64_t> strides(sizes.size(), 1);
      result.push_back(vector::ExtractStridedSliceOp::create(
          rewriter, loc, input, offsets, sizes, strides));
    }
  }
  return result;
}

/// Helper function to extract the filter slices after filter is unrolled along
/// kw.
static SmallVector<Value> extractConvFilterSlices(RewriterBase &rewriter,
                                                  Location loc, Value filter,
                                                  int64_t kwSize) {
  SmallVector<Value> result;
  // Extract rhs slice of size [{c, f} for channeled convolutions and {1} for
  // non-chanelled convolution] @ [kw].
  for (int64_t kw = 0; kw < kwSize; ++kw) {
    result.push_back(vector::ExtractOp::create(
        rewriter, loc, filter, /*offsets=*/ArrayRef<int64_t>{kw}));
  }
  return result;
}

/// Extract result slices, one per output spatial tile.
///
/// Per-tile shape and offset depend on `layout`:
///   Scalar:    {wSizeStep}            @ [w]
///   Batched:   {n, wSizeStep, f}      @ [0, w, 0]
///   Batchless: {wSizeStep, f}         @ [w, 0]
static SmallVector<Value>
extractConvResultSlices(RewriterBase &rewriter, Location loc, Value res,
                        int64_t nSize, int64_t wSize, int64_t fSize,
                        int64_t wSizeStep, ConvLayoutKind layout) {
  SmallVector<Value> result;
  for (int64_t w = 0; w < wSize; w += wSizeStep) {
    SmallVector<int64_t> sizes, offsets;
    switch (layout) {
    case ConvLayoutKind::Scalar:
      sizes = {wSizeStep};
      offsets = {w};
      break;
    case ConvLayoutKind::Batched:
      sizes = {nSize, wSizeStep, fSize};
      offsets = {0, w, 0};
      break;
    case ConvLayoutKind::Batchless:
      sizes = {wSizeStep, fSize};
      offsets = {w, 0};
      break;
    }
    SmallVector<int64_t> strides(sizes.size(), 1);
    result.push_back(vector::ExtractStridedSliceOp::create(
        rewriter, loc, res, offsets, sizes, strides));
  }
  return result;
}

/// Insert computed result slices back into the result vector.
///
/// Per-tile offset depends on `layout`:
///   Scalar:    @ [w]
///   Batched:   @ [0, w, 0]
///   Batchless: @ [w, 0]
static Value insertConvResultSlices(RewriterBase &rewriter, Location loc,
                                    Value res, int64_t wSize, int64_t wSizeStep,
                                    SmallVectorImpl<Value> &resVals,
                                    ConvLayoutKind layout) {
  for (int64_t w = 0; w < wSize; w += wSizeStep) {
    SmallVector<int64_t> offsets;
    switch (layout) {
    case ConvLayoutKind::Scalar:
      offsets = {w};
      break;
    case ConvLayoutKind::Batched:
      offsets = {0, w, 0};
      break;
    case ConvLayoutKind::Batchless:
      offsets = {w, 0};
      break;
    }
    SmallVector<int64_t> strides(offsets.size(), 1);
    res = vector::InsertStridedSliceOp::create(
        rewriter, loc, resVals[w / wSizeStep], res, offsets, strides);
  }
  return res;
}

/// Contains the vectorization state and related methods used across the
/// vectorization process of a given operation.
struct VectorizationState {
  VectorizationState(RewriterBase &rewriter) : rewriterGuard(rewriter) {}

  /// Initializes the vectorization state, including the computation of the
  /// canonical vector shape for vectorization.
  LogicalResult initState(RewriterBase &rewriter, LinalgOp linalgOp,
                          ArrayRef<int64_t> inputVectorSizes,
                          ArrayRef<bool> inputScalableVecDims,
                          bool assumeDynamicDimsMatchVecSizes = false);

  /// Returns the canonical vector shape used to vectorize the iteration space.
  ArrayRef<int64_t> getCanonicalVecShape() const { return canonicalVecShape; }

  /// Returns the vector dimensions that are scalable in the canonical vector
  /// shape.
  ArrayRef<bool> getScalableVecDims() const { return scalableVecDims; }

  /// Returns a vector type of the provided `elementType` with the canonical
  /// vector shape and the corresponding fixed/scalable dimensions bit. If
  /// `dimPermutation` is provided, the canonical vector dimensions are permuted
  /// accordingly.
  VectorType getCanonicalVecType(
      Type elementType,
      std::optional<AffineMap> dimPermutation = std::nullopt) const {
    SmallVector<int64_t> vectorShape;
    SmallVector<bool> scalableDims;
    if (dimPermutation.has_value()) {
      vectorShape =
          applyPermutationMap<int64_t>(*dimPermutation, canonicalVecShape);
      scalableDims =
          applyPermutationMap<bool>(*dimPermutation, scalableVecDims);
    } else {
      vectorShape.append(canonicalVecShape.begin(), canonicalVecShape.end());
      scalableDims.append(scalableVecDims.begin(), scalableVecDims.end());
    }

    return VectorType::get(vectorShape, elementType, scalableDims);
  }

  /// Masks an operation with the canonical vector mask if the operation needs
  /// masking. Returns the masked operation or the original operation if masking
  /// is not needed. If provided, the canonical mask for this operation is
  /// permuted using `maybeIndexingMap`.
  Operation *
  maskOperation(RewriterBase &rewriter, Operation *opToMask, LinalgOp linalgOp,
                std::optional<AffineMap> maybeIndexingMap = std::nullopt);

private:
  /// Initializes the iteration space static sizes using the Linalg op
  /// information. This may become more complicated in the future.
  void initIterSpaceStaticSizes(LinalgOp linalgOp) {
    iterSpaceStaticSizes.append(linalgOp.getStaticLoopRanges());
  }

  /// Generates 'arith.constant' and 'tensor/memref.dim' operations for
  /// all the static and dynamic dimensions of the iteration space to be
  /// vectorized and store them in `iterSpaceValueSizes`.
  LogicalResult precomputeIterSpaceValueSizes(RewriterBase &rewriter,
                                              LinalgOp linalgOp);

  /// Create or retrieve an existing mask value to mask `opToMask` in the
  /// canonical vector iteration space. If `maybeMaskingMap` the mask is
  /// permuted using that permutation map. If a new mask is created, it will be
  /// cached for future users.
  Value getOrCreateMaskFor(RewriterBase &rewriter, Operation *opToMask,
                           LinalgOp linalgOp,
                           std::optional<AffineMap> maybeMaskingMap);

  /// Check whether this permutation map can be used for masking. At the
  /// moment we only make sure that there are no broadcast dimensions, but this
  /// might change if indexing maps evolve.
  bool isValidMaskingMap(AffineMap maskingMap) {
    return maskingMap.getBroadcastDims().empty();
  }

  /// Turn the input indexing map into a valid masking map.
  ///
  /// The input indexing map may contain "zero" results, e.g.:
  ///    (d0, d1, d2, d3) -> (d2, d1, d0, 0)
  /// Applying such maps to canonical vector shapes like this one:
  ///    (1, 16, 16, 4)
  /// would yield an invalid vector shape like this:
  ///    (16, 16, 1, 0)
  /// Instead, drop the broadcasting dims that make no sense for masking perm.
  /// maps:
  ///    (d0, d1, d2, d3) -> (d2, d1, d0)
  /// This way, the corresponding vector/mask type will be:
  ///    vector<16x16x1xty>
  /// rather than this invalid Vector type:
  ///    vector<16x16x1x0xty>
  AffineMap getMaskingMapFromIndexingMap(AffineMap &indexingMap) {
    return indexingMap.dropZeroResults();
  }

  // Holds the compile-time static sizes of the iteration space to vectorize.
  // Dynamic dimensions are represented using ShapedType::kDynamic.
  SmallVector<int64_t> iterSpaceStaticSizes;

  /// Holds the value sizes of the iteration space to vectorize. Static
  /// dimensions are represented by 'arith.constant' and dynamic
  /// dimensions by 'tensor/memref.dim'.
  SmallVector<Value> iterSpaceValueSizes;

  /// Holds the canonical vector shape used to vectorize the iteration space.
  SmallVector<int64_t> canonicalVecShape;

  /// Holds the vector dimensions that are scalable in the canonical vector
  /// shape.
  SmallVector<bool> scalableVecDims;

  /// Holds the active masks for permutations of the canonical vector iteration
  /// space.
  DenseMap<AffineMap, Value> activeMaskCache;

  /// Global vectorization guard for the incoming rewriter. It's initialized
  /// when the vectorization state is initialized.
  OpBuilder::InsertionGuard rewriterGuard;

  /// Do all dynamic dims match the corresponding vector sizes?
  ///
  /// When a dynamic tensor/memref dimension matches the corresponding vector
  /// dimension, masking can be safely skipped, despite the presence of dynamic
  /// shapes. Use this flag with care and only for cases where you are
  /// confident the assumption holds.
  bool assumeDynamicDimsMatchVecSizes = false;
};

LogicalResult
VectorizationState::precomputeIterSpaceValueSizes(RewriterBase &rewriter,
                                                  LinalgOp linalgOp) {
  // TODO: Support 0-d vectors.
  for (int vecDim = 0, end = canonicalVecShape.size(); vecDim < end; ++vecDim) {
    if (ShapedType::isStatic(iterSpaceStaticSizes[vecDim])) {
      // Create constant index op for static dimensions.
      iterSpaceValueSizes.push_back(arith::ConstantIndexOp::create(
          rewriter, linalgOp.getLoc(), iterSpaceStaticSizes[vecDim]));
      continue;
    }

    // Find an operand defined on this dimension of the iteration space to
    // extract the runtime dimension size.
    Value operand;
    unsigned operandDimPos;
    if (failed(linalgOp.mapIterationSpaceDimToOperandDim(vecDim, operand,
                                                         operandDimPos)))
      return failure();

    Value dynamicDim =
        linalgOp.hasPureTensorSemantics()
            ? (Value)tensor::DimOp::create(rewriter, linalgOp.getLoc(), operand,
                                           operandDimPos)
            : (Value)memref::DimOp::create(rewriter, linalgOp.getLoc(), operand,
                                           operandDimPos);
    iterSpaceValueSizes.push_back(dynamicDim);
  }

  return success();
}

/// Initializes the vectorization state, including the computation of the
/// canonical vector shape for vectorization.
// TODO: Move this to the constructor when we can remove the failure cases.
LogicalResult VectorizationState::initState(RewriterBase &rewriter,
                                            LinalgOp linalgOp,
                                            ArrayRef<int64_t> inputVectorSizes,
                                            ArrayRef<bool> inputScalableVecDims,
                                            bool assumeDimsMatchVec) {
  assumeDynamicDimsMatchVecSizes = assumeDimsMatchVec;
  // Initialize the insertion point.
  rewriter.setInsertionPoint(linalgOp);

  if (!inputVectorSizes.empty()) {
    // Get the canonical vector shape from the input vector sizes provided. This
    // path should be taken to vectorize code with dynamic shapes and when using
    // vector sizes greater than the iteration space sizes.
    canonicalVecShape.append(inputVectorSizes.begin(), inputVectorSizes.end());
    scalableVecDims.append(inputScalableVecDims.begin(),
                           inputScalableVecDims.end());
  } else {
    // Compute the canonical vector shape from the operation shape. If there are
    // dynamic shapes, the operation won't be vectorized. We assume all the
    // vector dimensions are fixed.
    canonicalVecShape = linalgOp.getStaticLoopRanges();
    scalableVecDims.append(linalgOp.getNumLoops(), false);
  }

  LDBG() << "Canonical vector shape: " << llvm::interleaved(canonicalVecShape);
  LDBG() << "Scalable vector dims: " << llvm::interleaved(scalableVecDims);

  if (ShapedType::isDynamicShape(canonicalVecShape))
    return failure();

  // Initialize iteration space static sizes.
  initIterSpaceStaticSizes(linalgOp);

  // Generate 'arith.constant' and 'tensor/memref.dim' operations for
  // all the static and dynamic dimensions of the iteration space, needed to
  // compute a mask during vectorization.
  if (failed(precomputeIterSpaceValueSizes(rewriter, linalgOp)))
    return failure();

  return success();
}

/// Create or retrieve an existing mask value to mask `opToMask` in the
/// canonical vector iteration space. If `maybeMaskingMap` the mask is permuted
/// using that permutation map. If a new mask is created, it will be cached for
/// future users.
Value VectorizationState::getOrCreateMaskFor(
    RewriterBase &rewriter, Operation *opToMask, LinalgOp linalgOp,
    std::optional<AffineMap> maybeMaskingMap) {

  assert((!maybeMaskingMap || isValidMaskingMap(*maybeMaskingMap)) &&
         "Ill-formed masking map.");

  // No mask is needed if the operation is not maskable.
  auto maskableOp = dyn_cast<vector::MaskableOpInterface>(opToMask);
  if (!maskableOp)
    return Value();

  assert(!maskableOp.isMasked() &&
         "Masking an operation that is already masked");

  // If no masking map was provided, use an identity map with the loop dims.
  assert((!maybeMaskingMap || *maybeMaskingMap) &&
         "Unexpected null mask permutation map");
  AffineMap maskingMap =
      maybeMaskingMap ? *maybeMaskingMap
                      : AffineMap::getMultiDimIdentityMap(
                            linalgOp.getNumLoops(), rewriter.getContext());

  LDBG() << "Masking map: " << maskingMap;

  // Return the active mask for the masking map of this operation if it was
  // already created.
  auto activeMaskIt = activeMaskCache.find(maskingMap);
  if (activeMaskIt != activeMaskCache.end()) {
    Value mask = activeMaskIt->second;
    LDBG() << "Reusing mask: " << mask;
    return mask;
  }

  // Compute permuted projection of the iteration space to be masked and the
  // corresponding mask shape. If the resulting iteration space dimensions are
  // static and identical to the mask shape, masking is not needed for this
  // operation.
  // TODO: Improve this check. Only projected permutation indexing maps are
  // supported.
  SmallVector<int64_t> permutedStaticSizes =
      applyPermutationMap<int64_t>(maskingMap, iterSpaceStaticSizes);
  auto maskType = getCanonicalVecType(rewriter.getI1Type(), maskingMap);
  auto maskShape = maskType.getShape();

  LDBG() << "Mask shape: " << llvm::interleaved(maskShape);

  if (permutedStaticSizes == maskShape) {
    LDBG() << "Masking is not needed for masking map: " << maskingMap;
    activeMaskCache[maskingMap] = Value();
    return Value();
  }

  if (assumeDynamicDimsMatchVecSizes) {
    // While for _dynamic_ dim sizes we can _assume_ that the corresponding
    // vector sizes match, we still need to check the _static_ dim sizes. Only
    // then we can be 100% sure that masking is not required.
    if (llvm::all_of(llvm::zip(permutedStaticSizes, maskType.getShape()),
                     [](auto it) {
                       return std::get<0>(it) == ShapedType::kDynamic
                                  ? true
                                  : std::get<0>(it) == std::get<1>(it);
                     })) {
      LDBG()
          << "Dynamic + static dimensions match vector sizes, masking is not "
             "required.";
      activeMaskCache[maskingMap] = Value();
      return Value();
    }
  }

  // Permute the iteration space value sizes to compute the mask upper bounds.
  SmallVector<Value> upperBounds =
      applyPermutationMap(maskingMap, ArrayRef<Value>(iterSpaceValueSizes));
  assert(!maskShape.empty() && !upperBounds.empty() &&
         "Masked 0-d vectors are not supported yet");

  // Create the mask based on the dimension values.
  Value mask = vector::CreateMaskOp::create(rewriter, linalgOp.getLoc(),
                                            maskType, upperBounds);
  LDBG() << "Creating new mask: " << mask;
  activeMaskCache[maskingMap] = mask;
  return mask;
}

Operation *
VectorizationState::maskOperation(RewriterBase &rewriter, Operation *opToMask,
                                  LinalgOp linalgOp,
                                  std::optional<AffineMap> maybeIndexingMap) {
  LDBG() << "Trying to mask: " << *opToMask;

  std::optional<AffineMap> maybeMaskingMap = std::nullopt;
  if (maybeIndexingMap)
    maybeMaskingMap = getMaskingMapFromIndexingMap(*maybeIndexingMap);

  // Create or retrieve mask for this operation.
  Value mask =
      getOrCreateMaskFor(rewriter, opToMask, linalgOp, maybeMaskingMap);

  if (!mask) {
    LDBG() << "No mask required";
    if (assumeDynamicDimsMatchVecSizes) {
      llvm::TypeSwitch<Operation *>(opToMask)
          .Case<vector::TransferReadOp, vector::TransferWriteOp>(
              [&](auto xferOp) {
                // For vector.transfer_read and vector.transfer_write, there is
                // also the `in-bounds` attribute that has to be set explicitly
                // to true. Otherwise, "out-of-bounds" access will be assumed
                // and masks will be generated while lowering these.
                LDBG() << "Assuming dynamic dimensions match vector sizes and "
                          "setting their in-bounds to true!";
                SmallVector<bool> inBoundsMap = xferOp.getInBoundsValues();
                ShapedType xferType = xferOp.getShapedType();
                AffineMap permMap = xferOp.getPermutationMap();
                // Only set the in-bounds values to true for dynamic dims.
                // Different mechanisms will set these accordingly for the
                // static dims.
                for (unsigned i = 0; i < xferOp.getTransferRank(); i++) {
                  auto dimExpr = dyn_cast<AffineDimExpr>(permMap.getResult(i));
                  // Skip broadcast dimensions.
                  if (!dimExpr)
                    continue;
                  unsigned pos = dimExpr.getPosition();
                  if (xferType.isDynamicDim(pos))
                    inBoundsMap[i] = true;
                }
                rewriter.modifyOpInPlace(xferOp, [&]() {
                  xferOp.setInBoundsAttr(
                      rewriter.getBoolArrayAttr(inBoundsMap));
                });
              })
          .Default([](Operation *op) {
            // No-op if the operation is not an xfer read or write.
          });
    }
    return opToMask;
  }

  // Wrap the operation with a new `vector.mask` and update D-U chain.
  assert(opToMask && "Expected a valid operation to mask");
  auto maskOp = cast<vector::MaskOp>(
      mlir::vector::maskOperation(rewriter, opToMask, mask));
  Operation *maskOpTerminator = &maskOp.getMaskRegion().front().back();

  for (auto [resIdx, resVal] : llvm::enumerate(opToMask->getResults()))
    rewriter.replaceAllUsesExcept(resVal, maskOp.getResult(resIdx),
                                  maskOpTerminator);

  LDBG() << "Masked operation: " << *maskOp;
  return maskOp;
}

/// Given an indexing `map` coming from a LinalgOp indexing, restricted to a
/// projectedPermutation, compress the unused dimensions to serve as a
/// permutation_map for a vector transfer operation.
/// For example, given a linalg op such as:
///
/// ```
///   %0 = linalg.generic {
///        indexing_maps = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d2)>,
///        indexing_maps = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3)>
///      }
///     ins(%0 : tensor<2x3x4xf32>)
///    outs(%1 : tensor<5x6xf32>)
/// ```
///
/// the iteration domain size of the linalg op is 3x5x4x6x2. The first affine
/// map is reindexed to `affine_map<(d0, d1, d2) -> (d2, d0, d1)>`, the second
/// affine map is reindexed to `affine_map<(d0, d1) -> (d0, d1)>`.
static AffineMap reindexIndexingMap(AffineMap map) {
  assert(map.isProjectedPermutation(/*allowZeroInResults=*/true) &&
         "expected projected permutation");
  auto res = compressUnusedDims(map);
  assert(res.getNumDims() ==
             (res.getNumResults() - res.getNumOfZeroResults()) &&
         "expected reindexed map with same number of dims and results");
  return res;
}

/// Helper data structure to represent the result of vectorization for a single
/// operation. In certain specific cases, like terminators, we do not want to
/// propagate.
enum VectorizationHookStatus {
  /// Op failed to vectorize.
  Failure = 0,
  /// Op vectorized and custom function took care of replacement logic
  NoReplace,
  /// Op vectorized into a new Op whose results will replace original Op's
  /// results.
  NewOp
  // TODO: support values if Op vectorized to Many-Ops whose results we need to
  // aggregate for replacement.
};
/// VectorizationHookResult contains the vectorized op returned from a
/// CustomVectorizationHook. This is an internal implementation detail of
/// linalg vectorization, not to be confused with VectorizationResult.
struct VectorizationHookResult {
  /// Return status from vectorizing the current op.
  enum VectorizationHookStatus status = VectorizationHookStatus::Failure;
  /// New vectorized operation to replace the current op.
  /// Replacement behavior is specified by `status`.
  Operation *newOp;
};

std::optional<vector::CombiningKind>
mlir::linalg::getCombinerOpKind(Operation *combinerOp) {
  using ::mlir::vector::CombiningKind;

  if (!combinerOp)
    return std::nullopt;
  return llvm::TypeSwitch<Operation *, std::optional<CombiningKind>>(combinerOp)
      .Case<arith::AddIOp, arith::AddFOp>(
          [&](auto op) { return CombiningKind::ADD; })
      .Case([&](arith::AndIOp op) { return CombiningKind::AND; })
      .Case([&](arith::MaxSIOp op) { return CombiningKind::MAXSI; })
      .Case([&](arith::MaxUIOp op) { return CombiningKind::MAXUI; })
      .Case([&](arith::MaximumFOp op) { return CombiningKind::MAXIMUMF; })
      .Case([&](arith::MaxNumFOp op) { return CombiningKind::MAXNUMF; })
      .Case([&](arith::MinSIOp op) { return CombiningKind::MINSI; })
      .Case([&](arith::MinUIOp op) { return CombiningKind::MINUI; })
      .Case([&](arith::MinimumFOp op) { return CombiningKind::MINIMUMF; })
      .Case([&](arith::MinNumFOp op) { return CombiningKind::MINNUMF; })
      .Case<arith::MulIOp, arith::MulFOp>(
          [&](auto op) { return CombiningKind::MUL; })
      .Case([&](arith::OrIOp op) { return CombiningKind::OR; })
      .Case([&](arith::XOrIOp op) { return CombiningKind::XOR; })
      .Default(std::nullopt);
}

/// Check whether `outputOperand` is a reduction with a single combiner
/// operation. Return the combiner operation of the reduction. Return
/// nullptr otherwise. Multiple reduction operations would impose an
/// ordering between reduction dimensions and is currently unsupported in
/// Linalg. This limitation is motivated by the fact that e.g. min(max(X)) !=
/// max(min(X))
// TODO: use in LinalgOp verification, there is a circular dependency atm.
static Operation *matchLinalgReduction(OpOperand *outputOperand) {
  auto linalgOp = cast<LinalgOp>(outputOperand->getOwner());
  unsigned outputPos =
      outputOperand->getOperandNumber() - linalgOp.getNumDpsInputs();
  // Only single combiner operations are supported for now.
  SmallVector<Operation *, 4> combinerOps;
  if (!matchReduction(linalgOp.getRegionOutputArgs(), outputPos, combinerOps) ||
      combinerOps.size() != 1)
    return nullptr;

  // Return the combiner operation.
  return combinerOps[0];
}

/// Broadcast `value` to a vector of `shape` if possible. Return value
/// otherwise.
static Value broadcastIfNeeded(OpBuilder &b, Value value, Type dstType) {
  auto dstVecType = dyn_cast<VectorType>(dstType);
  // If no shape to broadcast to, just return `value`.
  if (dstVecType.getRank() == 0)
    return value;
  if (vector::isBroadcastableTo(value.getType(), dstVecType) !=
      vector::BroadcastableToResult::Success)
    return value;
  Location loc = b.getInsertionPoint()->getLoc();
  return b.createOrFold<vector::BroadcastOp>(loc, dstVecType, value);
}

/// Create MultiDimReductionOp to compute the reduction for `reductionOp`. This
/// assumes that `reductionOp` has two operands and one of them is the reduction
/// initial value.buildMultiDimReduce
// Note: this is a true builder that notifies the OpBuilder listener.
// TODO: Consider moving as a static helper on the ReduceOp.
static Operation *buildMultiDimReduce(OpBuilder &b, Operation *reduceOp,
                                      Value valueToReduce, Value acc,
                                      ArrayRef<bool> dimsToMask) {
  auto maybeKind = getCombinerOpKind(reduceOp);
  assert(maybeKind && "Failed precondition: could not get reduction kind");
  return vector::MultiDimReductionOp::create(
      b, reduceOp->getLoc(), valueToReduce, acc, dimsToMask, *maybeKind);
}

static SmallVector<bool> getDimsToReduce(LinalgOp linalgOp) {
  return llvm::map_to_vector(linalgOp.getIteratorTypesArray(),
                             isReductionIterator);
}

/// Check if `op` is a linalg.reduce or a linalg.generic that has at least one
/// reduction iterator.
static bool hasReductionIterator(LinalgOp &op) {
  return isa<linalg::ReduceOp>(op) ||
         (isa<linalg::GenericOp>(op) &&
          llvm::any_of(op.getIteratorTypesArray(), isReductionIterator));
}

/// Build a vector.transfer_write of `value` into `outputOperand` at indices set
/// to all `0`; where `outputOperand` is an output operand of the LinalgOp
/// currently being vectorized. If `dest` has null rank, build an memref.store.
/// Return the produced value or null if no value is produced.
// Note: this is a true builder that notifies the OpBuilder listener.
// TODO: Consider moving as a static helper on the ReduceOp.
static Value buildVectorWrite(RewriterBase &rewriter, Value value,
                              OpOperand *outputOperand,
                              VectorizationState &state) {
  Location loc = value.getLoc();
  auto linalgOp = cast<LinalgOp>(outputOperand->getOwner());
  AffineMap opOperandMap = linalgOp.getMatchingIndexingMap(outputOperand);

  // Compute the vector type of the value to store. This type should be an
  // identity or projection of the canonical vector type without any permutation
  // applied, given that any permutation in a transfer write happens as part of
  // the write itself.
  AffineMap vectorTypeMap = AffineMap::getFilteredIdentityMap(
      opOperandMap.getContext(), opOperandMap.getNumInputs(),
      [&](AffineDimExpr dimExpr) -> bool {
        return llvm::is_contained(opOperandMap.getResults(), dimExpr);
      });
  auto vectorType = state.getCanonicalVecType(
      getElementTypeOrSelf(outputOperand->get().getType()), vectorTypeMap);

  SmallVector<Value> indices(linalgOp.getRank(outputOperand),
                             arith::ConstantIndexOp::create(rewriter, loc, 0));

  Operation *write;
  if (vectorType.getRank() > 0) {
    AffineMap writeMap = inversePermutation(reindexIndexingMap(opOperandMap));
    value = broadcastIfNeeded(rewriter, value, vectorType);
    assert(value.getType() == vectorType && "Incorrect type");
    write = vector::TransferWriteOp::create(
        rewriter, loc, value, outputOperand->get(), indices, writeMap);
  } else {
    // 0-d case is still special: do not invert the reindexing writeMap.
    if (!isa<VectorType>(value.getType()))
      value = vector::BroadcastOp::create(rewriter, loc, vectorType, value);
    assert(value.getType() == vectorType && "Incorrect type");
    write = vector::TransferWriteOp::create(rewriter, loc, value,
                                            outputOperand->get(), indices);
  }

  write = state.maskOperation(rewriter, write, linalgOp, opOperandMap);

  // If masked, set in-bounds to true. Masking guarantees that the access will
  // be in-bounds.
  if (auto maskOp = dyn_cast<vector::MaskingOpInterface>(write)) {
    auto maskedWriteOp = cast<vector::TransferWriteOp>(maskOp.getMaskableOp());
    SmallVector<bool> inBounds(maskedWriteOp.getVectorType().getRank(), true);
    maskedWriteOp.setInBoundsAttr(rewriter.getBoolArrayAttr(inBounds));
  }

  LDBG() << "vectorized op: " << *write;
  if (!write->getResults().empty())
    return write->getResult(0);
  return Value();
}

// Custom vectorization precondition function type. This is intented to be used
// with CustomVectorizationHook. Returns success if the corresponding custom
// hook can vectorize the op.
using CustomVectorizationPrecondition =
    std::function<LogicalResult(Operation *, bool)>;

// Custom vectorization function type. Produce a vector form of Operation*
// assuming all its vectorized operands are already in the IRMapping.
// Return nullptr if the Operation cannot be vectorized.
using CustomVectorizationHook =
    std::function<VectorizationHookResult(Operation *, const IRMapping &)>;

/// Helper function to vectorize the terminator of a `linalgOp`. New result
/// vector values are appended to `newResults`. Return
/// VectorizationHookStatus::NoReplace to signal the vectorization algorithm
/// that it should not try to map produced operations and instead return the
/// results using the `newResults` vector making them available to the
/// vectorization algorithm for RAUW. This function is meant to be used as a
/// CustomVectorizationHook.
static VectorizationHookResult
vectorizeLinalgYield(RewriterBase &rewriter, Operation *op,
                     const IRMapping &bvm, VectorizationState &state,
                     LinalgOp linalgOp, SmallVectorImpl<Value> &newResults) {
  auto yieldOp = dyn_cast<linalg::YieldOp>(op);
  if (!yieldOp)
    return VectorizationHookResult{VectorizationHookStatus::Failure, nullptr};
  for (const auto &output : llvm::enumerate(yieldOp.getValues())) {
    // TODO: Scan for an opportunity for reuse.
    // TODO: use a map.
    Value vectorValue = bvm.lookup(output.value());
    Value newResult =
        buildVectorWrite(rewriter, vectorValue,
                         linalgOp.getDpsInitOperand(output.index()), state);
    if (newResult)
      newResults.push_back(newResult);
  }

  return VectorizationHookResult{VectorizationHookStatus::NoReplace, nullptr};
}

/// Helper function to vectorize the index operations of a `linalgOp`. Return
/// VectorizationHookStatus::NewOp to signal the vectorization algorithm that it
/// should map the produced operations. This function is meant to be used as a
/// CustomVectorizationHook.
static VectorizationHookResult vectorizeLinalgIndex(RewriterBase &rewriter,
                                                    VectorizationState &state,
                                                    Operation *op,
                                                    LinalgOp linalgOp) {
  IndexOp indexOp = dyn_cast<linalg::IndexOp>(op);
  if (!indexOp)
    return VectorizationHookResult{VectorizationHookStatus::Failure, nullptr};
  auto loc = indexOp.getLoc();
  // Compute the static loop sizes of the index op.
  ArrayRef<int64_t> targetShape = state.getCanonicalVecShape();
  auto dim = indexOp.getDim();
  // Compute a one-dimensional index vector for the index op dimension.
  auto indexVectorType =
      VectorType::get({targetShape[dim]}, rewriter.getIndexType(),
                      state.getScalableVecDims()[dim]);
  auto indexSteps = vector::StepOp::create(rewriter, loc, indexVectorType);
  // Return the one-dimensional index vector if it lives in the trailing
  // dimension of the iteration space since the vectorization algorithm in this
  // case can handle the broadcast.
  if (dim == targetShape.size() - 1)
    return VectorizationHookResult{VectorizationHookStatus::NewOp, indexSteps};
  // Otherwise permute the targetShape to move the index dimension last,
  // broadcast the one-dimensional index vector to the permuted shape, and
  // finally transpose the broadcasted index vector to undo the permutation.
  auto permPattern =
      llvm::to_vector(llvm::seq<unsigned>(0, targetShape.size()));
  std::swap(permPattern[dim], permPattern.back());
  auto permMap =
      AffineMap::getPermutationMap(permPattern, linalgOp.getContext());

  auto broadCastOp = vector::BroadcastOp::create(
      rewriter, loc,
      state.getCanonicalVecType(rewriter.getIndexType(), permMap), indexSteps);
  SmallVector<int64_t> transposition =
      llvm::to_vector<16>(llvm::seq<int64_t>(0, linalgOp.getNumLoops()));
  std::swap(transposition.back(), transposition[dim]);
  auto transposeOp =
      vector::TransposeOp::create(rewriter, loc, broadCastOp, transposition);
  return VectorizationHookResult{VectorizationHookStatus::NewOp, transposeOp};
}

/// Helper function to check if the tensor.extract can be vectorized by the
/// custom hook vectorizeTensorExtract.
static LogicalResult
tensorExtractVectorizationPrecondition(Operation *op, bool vectorizeNDExtract) {
  tensor::ExtractOp extractOp = dyn_cast<tensor::ExtractOp>(op);
  if (!extractOp)
    return failure();

  if (extractOp.getIndices().size() != 1 && !vectorizeNDExtract)
    return failure();

  // Check the index type, but only for non 0-d tensors (for which we do need
  // access indices).
  if (not extractOp.getIndices().empty()) {
    if (!VectorType::isValidElementType(extractOp.getIndices()[0].getType()))
      return failure();
  }

  if (!llvm::all_of(extractOp->getResultTypes(),
                    VectorType::isValidElementType)) {
    return failure();
  }

  return success();
}

/// Calculates the offsets (`$index_vec`) for `vector.gather` operations
/// generated from `tensor.extract`. The offset is calculated as follows
/// (example using scalar values):
///
///    offset = extractOp.indices[0]
///    for (i = 1; i < numIndices; i++)
///      offset = extractOp.dimSize[i] * offset + extractOp.indices[i];
///
/// For tensor<45 x 80 x 15 x f32> and index [1, 2, 3], this leads to:
///  offset = ( ( 1 ) * 80 +  2 ) * 15  + 3
static Value calculateGatherOffset(RewriterBase &rewriter,
                                   VectorizationState &state,
                                   tensor::ExtractOp extractOp,
                                   const IRMapping &bvm) {
  // The vector of indices for GatherOp should be shaped as the output vector.
  auto indexVecType = state.getCanonicalVecType(rewriter.getIndexType());
  auto loc = extractOp.getLoc();

  Value offset = broadcastIfNeeded(
      rewriter, bvm.lookup(extractOp.getIndices()[0]), indexVecType);

  const size_t numIndices = extractOp.getIndices().size();
  for (size_t i = 1; i < numIndices; i++) {
    Value dimIdx = arith::ConstantIndexOp::create(rewriter, loc, i);

    auto dimSize = broadcastIfNeeded(
        rewriter,
        tensor::DimOp::create(rewriter, loc, extractOp.getTensor(), dimIdx),
        indexVecType);

    offset = arith::MulIOp::create(rewriter, loc, offset, dimSize);

    auto extractOpIndex = broadcastIfNeeded(
        rewriter, bvm.lookup(extractOp.getIndices()[i]), indexVecType);

    offset = arith::AddIOp::create(rewriter, loc, extractOpIndex, offset);
  }

  return offset;
}

enum VectorMemoryAccessKind { ScalarBroadcast, Contiguous, Gather };

/// Find the index of the trailing non-unit dim in linalgOp. This hook is used
/// when checking whether `tensor.extract` Op (within a `linalg.generic` Op)
/// represents a contiguous load operation.
///
/// Note that when calling this hook, it is assumed that the output vector is
/// effectively 1D. Other cases (i.e. reading n-D vectors) should've been
/// labelled as a gather load before entering this method.
///
/// Following on from the above, it is assumed that:
///   * for statically shaped loops, when no masks are used, only one dim is !=
///   1 (that's what the shape of the output vector is based on).
///   * for dynamically shaped loops, there might be more non-unit dims
///   as the output vector type is user-specified.
///
/// TODO: Statically shaped loops + vector masking
static uint64_t getTrailingNonUnitLoopDimIdx(LinalgOp linalgOp) {
  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
  assert(
      (linalgOp.hasDynamicShape() ||
       llvm::count_if(loopRanges, [](int64_t dim) { return dim != 1; }) == 1) &&
      "For statically shaped Linalg Ops, only one "
      "non-unit loop dim is expected");
  assert(!loopRanges.empty() && "Empty loops, nothing to analyse.");

  size_t idx = loopRanges.size() - 1;
  for (; idx != 0; idx--)
    if (loopRanges[idx] != 1)
      break;

  return idx;
}

/// Checks whether `val` can be used for calculating a loop invariant index.
static bool isLoopInvariantIdx(LinalgOp &linalgOp, Value &val,
                               VectorType resType) {

  assert(((llvm::count_if(resType.getShape(),
                          [](int64_t dimSize) { return dimSize > 1; }) == 1)) &&
         "n-D vectors are not yet supported");

  // Blocks outside _this_ linalg.generic are effectively loop invariant.
  // However, analysing block arguments for _this_ linalg.generic Op is a bit
  // tricky. Just bail out in the latter case.
  // TODO: We could try analysing the corresponding affine map here.
  auto *block = linalgOp.getBlock();
  if (isa<BlockArgument>(val))
    return !llvm::is_contained(block->getArguments(), val);

  Operation *defOp = val.getDefiningOp();
  assert(defOp && "This is neither a block argument nor an operation result");

  // IndexOp is loop invariant as long as its result remains constant across
  // iterations. Note that for dynamic shapes, the corresponding dim will also
  // be conservatively treated as != 1.
  if (auto indexOp = dyn_cast<linalg::IndexOp>(defOp)) {
    return linalgOp.getStaticLoopRanges()[indexOp.getDim()] == 1;
  }

  auto *ancestor = block->findAncestorOpInBlock(*defOp);

  // Values define outside `linalgOp` are loop invariant.
  if (!ancestor)
    return true;

  // Values defined inside `linalgOp`, which are constant, are loop invariant.
  if (isa<arith::ConstantOp>(ancestor))
    return true;

  bool result = true;
  for (auto op : ancestor->getOperands())
    result &= isLoopInvariantIdx(linalgOp, op, resType);

  return result;
}

/// Check whether `val` could be used for calculating the trailing index for a
/// contiguous load operation.
///
/// There are currently 3 types of values that are allowed here:
///   1. loop-invariant values,
///   2. values that increment by 1 with every loop iteration,
///   3. results of basic arithmetic operations (linear and continuous)
///      involving 1., 2. and 3.
/// This method returns True if indeed only such values are used in calculating
/// `val.`
///
/// Additionally, the trailing index for a contiguous load operation should
/// increment by 1 with every loop iteration, i.e. be based on:
///   * `linalg.index <dim>` ,
/// where <dim> is the trailing non-unit dim of the iteration space (this way,
/// `linalg.index <dim>` increments by 1 with every loop iteration).
/// `foundIndexOp` is updated to `true` when such Op is found.
static bool isContiguousLoadIdx(LinalgOp &linalgOp, Value &val,
                                bool &foundIndexOp, VectorType resType) {

  assert(((llvm::count_if(resType.getShape(),
                          [](int64_t dimSize) { return dimSize > 1; }) == 1)) &&
         "n-D vectors are not yet supported");

  // Blocks outside _this_ linalg.generic are effectively loop invariant.
  // However, analysing block arguments for _this_ linalg.generic Op is a bit
  // tricky. Just bail out in the latter case.
  // TODO: We could try analysing the corresponding affine map here.
  auto *block = linalgOp.getBlock();
  if (isa<BlockArgument>(val))
    return !llvm::is_contained(block->getArguments(), val);

  Operation *defOp = val.getDefiningOp();
  assert(defOp && "This is neither a block argument nor an operation result");

  if (auto indexOp = dyn_cast<linalg::IndexOp>(defOp)) {
    auto loopDimThatIncrementsByOne = getTrailingNonUnitLoopDimIdx(linalgOp);

    foundIndexOp = (indexOp.getDim() == loopDimThatIncrementsByOne);
    return true;
  }

  auto *ancestor = block->findAncestorOpInBlock(*defOp);

  if (!ancestor)
    return false;

  // Conservatively reject Ops that could lead to indices with stride other
  // than 1.
  if (!isa<arith::AddIOp, arith::ConstantOp, linalg::IndexOp>(ancestor))
    return false;

  bool result = false;
  for (auto op : ancestor->getOperands())
    result |= isContiguousLoadIdx(linalgOp, op, foundIndexOp, resType);

  return result;
}

/// Infer the memory access pattern for the input ExtractOp
///
/// Based on the ExtratOp result shape and the access indices, decides whether
/// this Op corresponds to a contiguous load (including a broadcast of a scalar)
/// or a gather load. When analysing the ExtractOp indices (to identify
/// contiguous laods), this method looks for "loop" invariant indices (e.g.
/// block arguments) and indices that change linearly (e.g. via `linalg.index`
/// Op).
///
/// Note that it is always safe to use gather load operations for contiguous
/// loads (albeit slow), but not vice-versa. When in doubt, bail out and assume
/// that `extractOp` is a gather load.
static VectorMemoryAccessKind
getTensorExtractMemoryAccessPattern(tensor::ExtractOp extractOp,
                                    LinalgOp &linalgOp, VectorType resType) {

  auto inputShape = cast<ShapedType>(extractOp.getTensor().getType());

  // 0. Is this a 0-D vector? If yes then this is a scalar broadcast.
  if (inputShape.getShape().empty())
    return VectorMemoryAccessKind::ScalarBroadcast;

  // 0a. Is the result a 0-D vector? If yes, there are no iteration dimensions
  // so the tensor.extract is a single scalar load regardless of the index.
  if (resType.getRank() == 0)
    return VectorMemoryAccessKind::ScalarBroadcast;

  // True for vectors that are effectively 1D, e.g. `vector<1x4x1xi32>`, false
  // otherwise.
  bool isOutput1DVector =
      (llvm::count_if(resType.getShape(),
                      [](int64_t dimSize) { return dimSize > 1; }) == 1);
  // 1. Assume that it's a gather load when reading non-1D vector.
  if (!isOutput1DVector)
    return VectorMemoryAccessKind::Gather;

  bool leadingIdxsLoopInvariant = true;

  // 2. Analyze the leading indices of `extractOp`.
  // Look at the way each index is calculated and decide whether it is suitable
  // for a contiguous load, i.e. whether it's loop invariant. If not, it's a
  // gather load.
  auto indices = extractOp.getIndices();
  auto leadIndices = indices.drop_back(1);

  for (auto [i, indexVal] : llvm::enumerate(leadIndices)) {
    if (inputShape.getShape()[i] == 1)
      continue;

    leadingIdxsLoopInvariant &= isLoopInvariantIdx(linalgOp, indexVal, resType);
  }

  if (!leadingIdxsLoopInvariant) {
    LDBG() << "Found gather load: " << extractOp;
    return VectorMemoryAccessKind::Gather;
  }

  // 3. Analyze the trailing index for `extractOp`.
  // At this point we know that the leading indices are loop invariant. This
  // means that is potentially a scalar or a contiguous load. We can decide
  // based on the trailing idx.
  auto extractOpTrailingIdx = indices.back();

  // 3a. Scalar broadcast load
  // If the trailing index is loop invariant then this is a scalar load.
  if (leadingIdxsLoopInvariant &&
      isLoopInvariantIdx(linalgOp, extractOpTrailingIdx, resType)) {
    LDBG() << "Found scalar broadcast load: " << extractOp;

    return VectorMemoryAccessKind::ScalarBroadcast;
  }

  // 3b. Contiguous loads
  // The trailing `extractOp` index should increment with every loop iteration.
  // This effectively means that it must be based on the trailing loop index.
  // This is what the following bool captures.
  bool foundIndexOp = false;
  bool isContiguousLoad = isContiguousLoadIdx(linalgOp, extractOpTrailingIdx,
                                              foundIndexOp, resType);
  // TODO: Support generating contiguous loads for column vectors - that will
  // require adding a permutation map to tranfer_read Ops.
  bool isRowVector = resType.getShape().back() != 1;
  isContiguousLoad &= (foundIndexOp && isRowVector);

  if (isContiguousLoad) {
    LDBG() << "Found contigous load: " << extractOp;
    return VectorMemoryAccessKind::Contiguous;
  }

  // 4. Fallback case - gather load.
  LDBG() << "Found gather load: " << extractOp;
  return VectorMemoryAccessKind::Gather;
}

/// Helper function to vectorize the tensor.extract operations. Returns
/// VectorizationHookStatus::NewOp to signal the vectorization algorithm that it
/// should map the produced operations. This function is meant to be used as a
/// CustomVectorizationHook.
static VectorizationHookResult
vectorizeTensorExtract(RewriterBase &rewriter, VectorizationState &state,
                       Operation *op, LinalgOp linalgOp, const IRMapping &bvm) {
  tensor::ExtractOp extractOp = dyn_cast<tensor::ExtractOp>(op);
  if (!extractOp)
    return VectorizationHookResult{VectorizationHookStatus::Failure, nullptr};
  auto loc = extractOp.getLoc();

  // Compute the static loop sizes of the extract op.
  auto resultType = state.getCanonicalVecType(extractOp.getResult().getType());
  auto maskConstantOp = arith::ConstantOp::create(
      rewriter, loc,
      DenseIntElementsAttr::get(state.getCanonicalVecType(rewriter.getI1Type()),
                                /*value=*/true));
  auto passThruConstantOp = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(resultType));

  // Base indices are currently set to 0. We will need to re-visit if more
  // generic scenarios are to be supported.
  SmallVector<Value> baseIndices(
      extractOp.getIndices().size(),
      arith::ConstantIndexOp::create(rewriter, loc, 0));

  VectorMemoryAccessKind memAccessKind =
      getTensorExtractMemoryAccessPattern(extractOp, linalgOp, resultType);

  // 1. Handle gather access
  if (memAccessKind == VectorMemoryAccessKind::Gather) {
    Value offset = calculateGatherOffset(rewriter, state, extractOp, bvm);

    // Generate the gather load
    Operation *gatherOp = vector::GatherOp::create(
        rewriter, loc, resultType, extractOp.getTensor(), baseIndices, offset,
        maskConstantOp, passThruConstantOp);
    gatherOp = state.maskOperation(rewriter, gatherOp, linalgOp);

    LDBG() << "Vectorised as gather load: " << extractOp;
    return VectorizationHookResult{VectorizationHookStatus::NewOp, gatherOp};
  }

  // 2. Handle:
  //  a. scalar loads + broadcast,
  //  b. contiguous loads.
  // Both cases use vector.transfer_read.

  // Collect indices for `vector.transfer_read`. At this point, the indices will
  // either be scalars or would have been broadcast to vectors matching the
  // result type. For indices that are vectors, there are two options:
  //    * for non-trailing indices, all elements are identical (contiguous
  //      loads are identified by looking for non-trailing indices that are
  //      invariant with respect to the corresponding linalg.generic), or
  //    * for trailing indices, the index vector will contain values with stride
  //      one, but for `vector.transfer_read` only the first (i.e. 0th) index is
  //      needed.
  // This means that
  //   * for scalar indices - just re-use it,
  //   * for vector indices (e.g. `vector<1x1x4xindex>`) - extract the bottom
  //    (0th) element and use that.
  SmallVector<Value> transferReadIdxs;
  for (size_t i = 0; i < extractOp.getIndices().size(); i++) {
    Value idx = bvm.lookup(extractOp.getIndices()[i]);
    if (idx.getType().isIndex()) {
      transferReadIdxs.push_back(idx);
      continue;
    }

    auto indexAs1dVector = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get(resultType.getShape().back(), rewriter.getIndexType(),
                        resultType.getScalableDims().back()),
        idx);
    transferReadIdxs.push_back(
        vector::ExtractOp::create(rewriter, loc, indexAs1dVector, 0));
  }

  // `tensor.extract_element` is always in-bounds, hence the following holds.
  auto dstRank = resultType.getRank();
  auto srcRank = extractOp.getTensor().getType().getRank();
  SmallVector<bool> inBounds(dstRank, true);

  // 2a. Handle scalar broadcast access.
  if (memAccessKind == VectorMemoryAccessKind::ScalarBroadcast) {
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineExpr> exprs(dstRank, getAffineConstantExpr(0, ctx));
    auto permutationMap = AffineMap::get(srcRank, 0, exprs, ctx);

    auto transferReadOp = vector::TransferReadOp::create(
        rewriter, loc, resultType, extractOp.getTensor(), transferReadIdxs,
        /*padding=*/std::nullopt, permutationMap, inBounds);

    Operation *readOrMaskedReadOp = transferReadOp;
    if (dstRank > 0) {
      // Mask this broadcasting xfer_read here rather than relying on the
      // generic path (the generic path assumes identity masking map, which
      // wouldn't be valid here).
      SmallVector<int64_t> readMaskShape = {1};
      auto readMaskType = VectorType::get(readMaskShape, rewriter.getI1Type());
      auto allTrue = vector::ConstantMaskOp::create(
          rewriter, loc, readMaskType, vector::ConstantMaskKind::AllTrue);
      readOrMaskedReadOp =
          mlir::vector::maskOperation(rewriter, transferReadOp, allTrue);
    }

    LDBG() << "Vectorised as scalar broadcast load: " << extractOp;
    return VectorizationHookResult{VectorizationHookStatus::NewOp,
                                   readOrMaskedReadOp};
  }

  // 2b. Handle contiguous access.
  auto permutationMap = AffineMap::getMinorIdentityMap(
      srcRank, std::min(dstRank, srcRank), rewriter.getContext());

  int32_t rankDiff = dstRank - srcRank;
  // When dstRank > srcRank, broadcast the source tensor to the unitary leading
  // dims so that the ranks match. This is done by extending the map with 0s.
  // For example, for dstRank = 3, srcRank = 2, the following map created
  // above:
  //    (d0, d1) --> (d0, d1)
  // is extended as:
  //    (d0, d1) --> (0, d0, d1)
  while (rankDiff > 0) {
    permutationMap = permutationMap.insertResult(
        mlir::getAffineConstantExpr(0, rewriter.getContext()), 0);
    rankDiff--;
  }

  auto transferReadOp = vector::TransferReadOp::create(
      rewriter, loc, resultType, extractOp.getTensor(), transferReadIdxs,
      /*padding=*/std::nullopt, permutationMap, inBounds);

  LDBG() << "Vectorised as contiguous load: " << extractOp;
  return VectorizationHookResult{VectorizationHookStatus::NewOp,
                                 transferReadOp};
}

/// Emit reduction operations if the shapes of the value to reduce is different
/// that the result shape.
// Note: this is a true builder that notifies the OpBuilder listener.
// TODO: Consider moving as a static helper on the ReduceOp.
static Operation *reduceIfNeeded(OpBuilder &b, LinalgOp linalgOp, Operation *op,
                                 Value reduceValue, Value initialValue,
                                 const IRMapping &bvm) {
  Value reduceVec = bvm.lookup(reduceValue);
  Value outputVec = bvm.lookup(initialValue);
  auto reduceType = dyn_cast<VectorType>(reduceVec.getType());
  auto outputType = dyn_cast<VectorType>(outputVec.getType());
  // Reduce only if needed as the value may already have been reduce for
  // contraction vectorization.
  if (!reduceType ||
      (outputType && reduceType.getShape() == outputType.getShape()))
    return nullptr;
  SmallVector<bool> dimsToMask = getDimsToReduce(linalgOp);
  return buildMultiDimReduce(b, op, reduceVec, outputVec, dimsToMask);
}

/// Generic vectorization for a single operation `op`, given already vectorized
/// operands carried by `bvm`. Vectorization occurs as follows:
///   1. Try to apply any of the `customVectorizationHooks` and return its
///   result on success.
///   2. Clone any constant in the current scope without vectorization: each
///   consumer of the constant will later determine the shape to which the
///   constant needs to be broadcast to.
///   3. Fail on any remaining non `ElementwiseMappable` op. It is the purpose
///   of the `customVectorizationHooks` to cover such cases.
///   4. Clone `op` in vector form to a vector of shape prescribed by the first
///   operand of maximal rank. Other operands have smaller rank and are
///   broadcast accordingly. It is assumed this broadcast is always legal,
///   otherwise, it means one of the `customVectorizationHooks` is incorrect.
///
/// This function assumes all operands of `op` have been vectorized and are in
/// the `bvm` mapping. As a consequence, this function is meant to be called  on
/// a topologically-sorted list of ops.
/// This function does not update `bvm` but returns a VectorizationHookStatus
/// that instructs the caller what `bvm` update needs to occur.
static VectorizationHookResult
vectorizeOneOp(RewriterBase &rewriter, VectorizationState &state,
               LinalgOp linalgOp, Operation *op, const IRMapping &bvm,
               ArrayRef<CustomVectorizationHook> customVectorizationHooks) {
  LDBG() << "vectorize op " << *op;

  // 1. Try to apply any CustomVectorizationHook.
  if (!customVectorizationHooks.empty()) {
    for (auto &customFunc : customVectorizationHooks) {
      VectorizationHookResult result = customFunc(op, bvm);
      if (result.status == VectorizationHookStatus::Failure)
        continue;
      return result;
    }
  }

  // 2. Constant ops don't get vectorized but rather broadcasted at their users.
  // Clone so that the constant is not confined to the linalgOp block .
  if (isa<arith::ConstantOp, func::ConstantOp>(op))
    return VectorizationHookResult{VectorizationHookStatus::NewOp,
                                   rewriter.clone(*op)};

  // 3. Only ElementwiseMappable are allowed in the generic vectorization.
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return VectorizationHookResult{VectorizationHookStatus::Failure, nullptr};

  // 4 . Check if the operation is a reduction.
  SmallVector<std::pair<Value, Value>> reductionOperands;
  for (Value operand : op->getOperands()) {
    auto blockArg = dyn_cast<BlockArgument>(operand);
    if (!blockArg || blockArg.getOwner() != linalgOp.getBlock() ||
        blockArg.getArgNumber() < linalgOp.getNumDpsInputs())
      continue;
    SmallVector<Operation *> reductionOps;
    Value reduceValue = matchReduction(
        linalgOp.getRegionOutputArgs(),
        blockArg.getArgNumber() - linalgOp.getNumDpsInputs(), reductionOps);
    if (!reduceValue)
      continue;
    reductionOperands.push_back(std::make_pair(reduceValue, operand));
  }
  if (!reductionOperands.empty()) {
    assert(reductionOperands.size() == 1);
    Operation *reduceOp =
        reduceIfNeeded(rewriter, linalgOp, op, reductionOperands[0].first,
                       reductionOperands[0].second, bvm);
    if (reduceOp)
      return VectorizationHookResult{VectorizationHookStatus::NewOp, reduceOp};
  }

  // 5. Generic vectorization path for ElementwiseMappable ops.
  //   a. Get the first max ranked shape.
  VectorType firstMaxRankedType;
  for (Value operand : op->getOperands()) {
    auto vecOperand = bvm.lookup(operand);
    assert(vecOperand && "Vector operand couldn't be found");

    auto vecType = dyn_cast<VectorType>(vecOperand.getType());
    if (vecType && (!firstMaxRankedType ||
                    firstMaxRankedType.getRank() < vecType.getRank()))
      firstMaxRankedType = vecType;
  }
  //   b. Broadcast each op if needed.
  SmallVector<Value> vecOperands;
  for (Value scalarOperand : op->getOperands()) {
    Value vecOperand = bvm.lookup(scalarOperand);
    assert(vecOperand && "Vector operand couldn't be found");

    if (firstMaxRankedType) {
      auto vecType = VectorType::get(firstMaxRankedType.getShape(),
                                     getElementTypeOrSelf(vecOperand.getType()),
                                     firstMaxRankedType.getScalableDims());
      vecOperands.push_back(broadcastIfNeeded(rewriter, vecOperand, vecType));
    } else {
      vecOperands.push_back(vecOperand);
    }
  }
  //   c. for elementwise, the result is the vector with the firstMaxRankedShape
  SmallVector<Type> resultTypes;
  for (Type resultType : op->getResultTypes()) {
    resultTypes.push_back(
        firstMaxRankedType
            ? VectorType::get(firstMaxRankedType.getShape(), resultType,
                              firstMaxRankedType.getScalableDims())
            : resultType);
  }
  //   d. Build and return the new op.
  return VectorizationHookResult{
      VectorizationHookStatus::NewOp,
      rewriter.create(op->getLoc(), op->getName().getIdentifier(), vecOperands,
                      resultTypes, op->getAttrs())};
}

/// Generic vectorization function that rewrites the body of a `linalgOp` into
/// vector form. Generic vectorization proceeds as follows:
///   1. Verify the `linalgOp` has one non-empty region.
///   2. Values defined above the region are mapped to themselves and will be
///   broadcasted on a per-need basis by their consumers.
///   3. Each region argument is vectorized into a vector.transfer_read (or 0-d
///   load).
///   TODO: Reuse opportunities for RAR dependencies.
///   4a. Register CustomVectorizationHook for YieldOp to capture the results.
///   4rewriter. Register CustomVectorizationHook for IndexOp to access the
///   iteration indices.
///   5. Iteratively call vectorizeOneOp on the region operations.
///
/// When `broadcastToMaximalCommonShape` is set to true, eager broadcasting is
/// performed to the maximal common vector size implied by the `linalgOp`
/// iteration space. This eager broadcasting is introduced in the
/// permutation_map of the vector.transfer_read operations. The eager
/// broadcasting makes it trivial to determine where broadcast, transposes and
/// reductions should occur, without any bookkeeping. The tradeoff is that, in
/// the absence of good canonicalizations, the amount of work increases.
/// This is not deemed a problem as we expect canonicalizations and foldings to
/// aggressively clean up the useless work.
static LogicalResult
vectorizeAsLinalgGeneric(RewriterBase &rewriter, VectorizationState &state,
                         LinalgOp linalgOp,
                         SmallVectorImpl<Value> &newResults) {
  LDBG() << "Vectorizing operation as linalg generic/n";
  Block *block = linalgOp.getBlock();

  // 2. Values defined above the region can only be broadcast for now. Make them
  // map to themselves.
  IRMapping bvm;
  SetVector<Value> valuesSet;
  mlir::getUsedValuesDefinedAbove(linalgOp->getRegion(0), valuesSet);
  bvm.map(valuesSet.getArrayRef(), valuesSet.getArrayRef());

  if (linalgOp.getNumDpsInits() == 0)
    return failure();

  // 3. Turn all BBArgs into vector.transfer_read / load.
  Location loc = linalgOp.getLoc();
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  for (OpOperand *opOperand : linalgOp.getOpOperandsMatchingBBargs()) {
    BlockArgument bbarg = linalgOp.getMatchingBlockArgument(opOperand);
    if (linalgOp.isScalar(opOperand)) {
      bvm.map(bbarg, opOperand->get());
      continue;
    }

    // 3.a. Convert the indexing map for this input/output to a transfer read
    // permutation map and masking map.
    AffineMap indexingMap = linalgOp.getMatchingIndexingMap(opOperand);

    AffineMap readMap;
    VectorType readType;
    Type elemType = getElementTypeOrSelf(opOperand->get());
    if (linalgOp.isDpsInput(opOperand)) {
      // 3.a.i. For input reads we use the canonical vector shape.
      readMap = inverseAndBroadcastProjectedPermutation(indexingMap);
      readType = state.getCanonicalVecType(elemType);
    } else {
      // 3.a.ii. For output reads (iteration-carried dependence, e.g.,
      // reductions), the vector shape is computed by mapping the canonical
      // vector shape to the output domain and back to the canonical domain.
      readMap = inversePermutation(reindexIndexingMap(indexingMap));
      readType =
          state.getCanonicalVecType(elemType, readMap.compose(indexingMap));
    }

    SmallVector<Value> indices(linalgOp.getShape(opOperand).size(), zero);

    Operation *read = vector::TransferReadOp::create(
        rewriter, loc, readType, opOperand->get(), indices,
        /*padding=*/std::nullopt, readMap);
    read = state.maskOperation(rewriter, read, linalgOp, indexingMap);
    Value readValue = read->getResult(0);

    // 3.b. If masked, set in-bounds to true. Masking guarantees that the access
    // will be in-bounds.
    if (auto maskOp = dyn_cast<vector::MaskingOpInterface>(read)) {
      SmallVector<bool> inBounds(readType.getRank(), true);
      cast<vector::TransferReadOp>(maskOp.getMaskableOp())
          .setInBoundsAttr(rewriter.getBoolArrayAttr(inBounds));
    }

    // 3.c. Not all ops support 0-d vectors, extract the scalar for now.
    // TODO: remove this.
    if (readType.getRank() == 0)
      readValue = vector::ExtractOp::create(rewriter, loc, readValue,
                                            ArrayRef<int64_t>());

    LDBG() << "New vectorized bbarg(" << bbarg.getArgNumber()
           << "): " << readValue;
    bvm.map(bbarg, readValue);
    bvm.map(opOperand->get(), readValue);
  }

  SmallVector<CustomVectorizationHook> hooks;
  // 4a. Register CustomVectorizationHook for yieldOp.
  CustomVectorizationHook vectorizeYield =
      [&](Operation *op, const IRMapping &bvm) -> VectorizationHookResult {
    return vectorizeLinalgYield(rewriter, op, bvm, state, linalgOp, newResults);
  };
  hooks.push_back(vectorizeYield);

  // 4b. Register CustomVectorizationHook for indexOp.
  CustomVectorizationHook vectorizeIndex =
      [&](Operation *op, const IRMapping &bvm) -> VectorizationHookResult {
    return vectorizeLinalgIndex(rewriter, state, op, linalgOp);
  };
  hooks.push_back(vectorizeIndex);

  // 4c. Register CustomVectorizationHook for extractOp.
  CustomVectorizationHook vectorizeExtract =
      [&](Operation *op, const IRMapping &bvm) -> VectorizationHookResult {
    return vectorizeTensorExtract(rewriter, state, op, linalgOp, bvm);
  };
  hooks.push_back(vectorizeExtract);

  // 5. Iteratively call `vectorizeOneOp` to each op in the slice.
  for (Operation &op : block->getOperations()) {
    VectorizationHookResult result =
        vectorizeOneOp(rewriter, state, linalgOp, &op, bvm, hooks);
    if (result.status == VectorizationHookStatus::Failure) {
      LDBG() << "failed to vectorize: " << op;
      return failure();
    }
    if (result.status == VectorizationHookStatus::NewOp) {
      Operation *maybeMaskedOp =
          state.maskOperation(rewriter, result.newOp, linalgOp);
      LDBG() << "New vector op: " << *maybeMaskedOp;
      bvm.map(op.getResults(), maybeMaskedOp->getResults());
    }
  }

  return success();
}

/// Given the re-associations, "collapses" the input Vector type
///
/// This is similar to CollapseShapeOp::inferCollapsedType with two notable
/// differences:
///   * We can safely assume that there are no dynamic sizes.
///   * Scalable flags are updated alongside regular dims.
///
/// When collapsing scalable flags, conservatively avoids cases with two
/// scalable dims. We could re-visit this in the future.
///
/// EXAMPLE:
///  type = vector<4x16x[8]x16xf32>
///  reassociation =  [(d0, d1, d2, d3) -> (d0, d1),
///                    (d0, d1, d2, d3) -> (d2, d3)]
///  Result:
///   vector<64x[128]xf32>
static VectorType getCollapsedVecType(VectorType type,
                                      ArrayRef<AffineMap> reassociation) {
  assert(type.getNumScalableDims() < 2 &&
         "Collapsing more than 1 scalable dim is not supported ATM");

  // Use the fact that reassociation is valid to simplify the logic: only use
  // each map's rank.
  assert(isReassociationValid(reassociation) && "invalid reassociation");

  auto shape = type.getShape();
  auto scalableFlags = type.getScalableDims();
  SmallVector<int64_t> newShape;
  SmallVector<bool> newScalableFlags;

  unsigned currentDim = 0;
  for (AffineMap m : reassociation) {
    unsigned dim = m.getNumResults();
    int64_t size = 1;
    bool flag = false;
    for (unsigned d = 0; d < dim; ++d) {
      size *= shape[currentDim + d];
      flag |= scalableFlags[currentDim + d];
    }
    newShape.push_back(size);
    newScalableFlags.push_back(flag);
    currentDim += dim;
  }

  return VectorType::get(newShape, type.getElementType(), newScalableFlags);
}

/// Vectorize `linalg.pack` as:
///   * xfer_read -> shape_cast -> transpose -> xfer_write
///
/// The input-vector-sizes specify the _write_ vector sizes (i.e. the vector
/// sizes for the xfer_write operation). This is sufficient to infer the other
/// vector sizes required here.
///
/// If the vector sizes are not provided:
///  * the vector sizes are determined from the destination tensor static shape.
///  * the inBounds attribute is used instead of masking.
///
/// EXAMPLE (no vector sizes):
/// ```
///   %pack = tensor.pack %src
///     inner_dims_pos = [2, 1]
///     inner_tiles = [16, 2]
///     into %dst : tensor<32x8x16xf32> -> tensor<32x4x1x16x2xf32>
/// ``
/// is vectorizes as:
/// ```
///   %read = vector.transfer_read %src
///     : tensor<32x7x16xf32>, vector<32x8x16xf32>
///   %sc = vector.shape_cast %read
///     : vector<32x8x16xf32> to vector<32x4x2x1x16xf32>
///   %tr = vector.transpose %sc, [0, 1, 3, 4, 2]
///     : vector<32x4x2x1x16xf32> to vector<32x4x1x16x2xf32>
///   %write = vector.transfer_write %tr into %dest
///     : vector<32x4x1x16x2xf32>, tensor<32x4x1x16x2xf32>
/// ```
static LogicalResult
vectorizeAsTensorPackOp(RewriterBase &rewriter, linalg::PackOp packOp,
                        ArrayRef<int64_t> inputVectorSizes,
                        SmallVectorImpl<Value> &newResults) {
  if (!inputVectorSizes.empty()) {
    assert(inputVectorSizes.size() == packOp.getDestRank() &&
           "Invalid number of input vector sizes!");
  }

  // TODO: Introduce a parent class that will handle the insertion point update.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(packOp);

  Location loc = packOp.getLoc();
  std::optional<Value> padValue = packOp.getPaddingValue()
                                      ? std::optional(packOp.getPaddingValue())
                                      : std::nullopt;

  SmallVector<int64_t> destShape =
      SmallVector<int64_t>(packOp.getDestType().getShape());

  // This is just a convenience alias to clearly communicate that the input
  // vector sizes determine the _write_ sizes.
  ArrayRef<int64_t> &writeVectorSizes = inputVectorSizes;

  // In the absence of input-vector-sizes, use the _static_ input tensor shape.
  // In addition, use the inBounds attribute instead of masking.
  bool useInBoundsInsteadOfMasking = false;
  if (writeVectorSizes.empty()) {
    if (ShapedType::isDynamicShape(destShape))
      return rewriter.notifyMatchFailure(packOp,
                                         "unable to infer vector sizes");

    writeVectorSizes = destShape;
    useInBoundsInsteadOfMasking = true;
  }

  // Compute pre-transpose-write-vector-type, i.e. the write vector type
  // _before_ the transposition (i.e. before dimension permutation). This is
  // done by inverting the permutation/transposition that's part of the Pack
  // operation. This type is required to:
  //   1) compute the read vector type for masked-read below, and
  //   2) generate shape-cast Op below that expands the read vector type.
  PackingMetadata packMetadata;
  SmallVector<int64_t> preTransposeWriteVecSizses(writeVectorSizes);
  auto destInvPermutation = getPackInverseDestPerm(packOp, packMetadata);
  applyPermutationToVector(preTransposeWriteVecSizses, destInvPermutation);
  auto preTransposeWriteVecType =
      VectorType::get(preTransposeWriteVecSizses,
                      packOp.getResult().getType().getElementType());

  // Compute vector type for the _read_ opeartion. This is simply
  // pre-transpose-write-vector-type with the dimensions collapsed
  // as per the Pack operation.
  VectorType readVecType = getCollapsedVecType(
      preTransposeWriteVecType,
      getSymbolLessAffineMaps(convertReassociationIndicesToExprs(
          rewriter.getContext(), packMetadata.reassociations)));

  // Create masked TransferReadOp.
  auto maskedRead = vector::createReadOrMaskedRead(
      rewriter, loc, packOp.getSource(), readVecType, padValue,
      useInBoundsInsteadOfMasking);

  // Create ShapeCastOp.
  auto shapeCastOp = vector::ShapeCastOp::create(
      rewriter, loc, preTransposeWriteVecType, maskedRead);

  // Create TransposeOp.
  auto destPermutation = invertPermutationVector(destInvPermutation);
  auto transposeOp = vector::TransposeOp::create(
      rewriter, loc, shapeCastOp.getResult(), destPermutation);

  // Create TransferWriteOp.
  Operation *write = vector::createWriteOrMaskedWrite(
      rewriter, loc, transposeOp.getResult(), packOp.getDest());
  newResults.push_back(write->getResult(0));
  return success();
}

/// Vectorize `linalg.unpack` as:
///   * xfer_read -> vector.transpose -> vector.shape_cast -> xfer_write
///
/// The input-vector-sizes specify the _read_ vector sizes (i.e. the vector
/// sizes for the xfer_read operation). This is sufficient to infer the other
/// vector sizes required here.
///
/// If the vector sizes are not provided:
///  * the vector sizes are determined from the input tensor static shape.
///  * the inBounds attribute is used instead of masking.
///
/// EXAMPLE (no vector sizes):
/// ```
///   %unpack = linalg.unpack  %src
///    inner_dims_pos = [0, 1]
///    inner_tiles = [8, 8]
///    into %dest : tensor<1x1x8x8xf32> -> tensor<8x8xf32>
/// ```
/// is vectorized as:
/// ```
///   %read = vector.transfer_read %src
///     : tensor<1x1x8x8xf32>, vector<1x1x8x8xf32>
///   %tr = vector.transpose %read, [0, 2, 1, 3]
///     : vector<1x1x8x8xf32> to vector<1x8x1x8xf32>
///   %sc = vector.shape_cast %tr
///     : vector<1x8x1x8xf32> to vector<8x8xf32>
///   %vector = vector.transfer_write %sc into %dest
///     : vector<8x8xf32>, tensor<8x8xf32>
/// ```
static LogicalResult
vectorizeAsTensorUnpackOp(RewriterBase &rewriter, linalg::UnPackOp unpackOp,
                          ArrayRef<int64_t> inputVectorSizes,
                          ArrayRef<bool> inputScalableVecDims,
                          SmallVectorImpl<Value> &newResults) {
  if (!inputVectorSizes.empty()) {
    assert(inputVectorSizes.size() == unpackOp.getSourceRank() &&
           "Invalid number of input vector sizes!");
    assert(inputVectorSizes.size() == inputScalableVecDims.size() &&
           "Incompatible number of vector sizes and vector scalable flags!");
  }

  // TODO: Introduce a parent class that will handle the insertion point update.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(unpackOp);

  ShapedType unpackTensorType = unpackOp.getSourceType();

  ArrayRef<int64_t> sourceShape = unpackTensorType.getShape();
  bool useInBoundsInsteadOfMasking = false;

  Location loc = unpackOp->getLoc();

  // Obtain vector sizes for the read operation.
  SmallVector<int64_t> readVectorSizes(inputVectorSizes);
  SmallVector<bool> readScalableVectorFlags(inputScalableVecDims);

  // In the absence of input-vector-sizes, use the _static_ input tensor shape.
  if (inputVectorSizes.empty()) {
    if (ShapedType::isDynamicShape(sourceShape))
      return rewriter.notifyMatchFailure(unpackOp,
                                         "Unable to infer vector sizes!");

    readVectorSizes.assign(sourceShape.begin(), sourceShape.end());
    useInBoundsInsteadOfMasking = true;
  }

  // -- Generate the read operation --
  VectorType readVecType =
      VectorType::get(readVectorSizes, unpackTensorType.getElementType(),
                      readScalableVectorFlags);
  Value readResult = vector::createReadOrMaskedRead(
      rewriter, loc, unpackOp.getSource(), readVecType, std::nullopt,
      useInBoundsInsteadOfMasking);

  // -- Generate the transpose operation --
  PackingMetadata packMetadata;
  SmallVector<int64_t> lastDimToInsertPosPerm =
      getUnPackInverseSrcPerm(unpackOp, packMetadata);
  vector::TransposeOp transposeOp = vector::TransposeOp::create(
      rewriter, loc, readResult, lastDimToInsertPosPerm);

  // -- Generate the shape_cast operation --
  VectorType collapsedVecType = getCollapsedVecType(
      transposeOp.getType(),
      getSymbolLessAffineMaps(convertReassociationIndicesToExprs(
          rewriter.getContext(), packMetadata.reassociations)));
  vector::ShapeCastOp shapeCastOp = vector::ShapeCastOp::create(
      rewriter, loc, collapsedVecType, transposeOp->getResult(0));

  // -- Generate the write operation --
  Operation *write = vector::createWriteOrMaskedWrite(
      rewriter, loc, shapeCastOp.getResult(), unpackOp.getDest(),
      /*writeIndices=*/{}, useInBoundsInsteadOfMasking);

  newResults.push_back(write->getResult(0));
  return success();
}

/// Vectorize a `padOp` with (1) static result type, (2) constant padding value
/// and (3) all-zero lowPad to
///   `transfer_write_in_bounds(transfer_read_masked(pad_source, pad_value))`.
static LogicalResult
vectorizeAsTensorPadOp(RewriterBase &rewriter, tensor::PadOp padOp,
                       ArrayRef<int64_t> inputVectorSizes,
                       SmallVectorImpl<Value> &newResults) {
  auto padValue = padOp.getConstantPaddingValue();
  Location loc = padOp.getLoc();

  // TODO: Introduce a parent class that will handle the insertion point update.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(padOp);

  ReifiedRankedShapedTypeDims reifiedReturnShapes;
  LogicalResult status =
      cast<ReifyRankedShapedTypeOpInterface>(padOp.getOperation())
          .reifyResultShapes(rewriter, reifiedReturnShapes);
  (void)status; // prevent unused variable warning on non-assert builds
  assert(succeeded(status) && "failed to reify result shapes");
  auto readType = VectorType::get(inputVectorSizes, padValue.getType());
  auto maskedRead = vector::createReadOrMaskedRead(
      rewriter, loc, padOp.getSource(), readType, padValue,
      /*useInBoundsInsteadOfMasking=*/false);

  // Create Xfer write Op
  Value dest = tensor::EmptyOp::create(rewriter, loc, reifiedReturnShapes[0],
                                       padOp.getResultType().getElementType());
  Operation *write =
      vector::createWriteOrMaskedWrite(rewriter, loc, maskedRead, dest);
  newResults.push_back(write->getResult(0));
  return success();
}

// TODO: probably need some extra checks for reduction followed by consumer
// ops that may not commute (e.g. linear reduction + non-linear instructions).
static LogicalResult reductionPreconditions(LinalgOp op) {
  if (llvm::none_of(op.getIteratorTypesArray(), isReductionIterator)) {
    LDBG() << "reduction precondition failed: no reduction iterator";
    return failure();
  }
  for (OpOperand &opOperand : op.getDpsInitsMutable()) {
    AffineMap indexingMap = op.getMatchingIndexingMap(&opOperand);
    if (indexingMap.isPermutation())
      continue;

    Operation *reduceOp = matchLinalgReduction(&opOperand);
    if (!reduceOp || !getCombinerOpKind(reduceOp)) {
      LDBG() << "reduction precondition failed: reduction detection failed";
      return failure();
    }
  }
  return success();
}

static LogicalResult
vectorizeDynamicConvOpPrecondition(linalg::LinalgOp conv,
                                   bool flatten1DDepthwiseConv) {
  if (flatten1DDepthwiseConv) {
    LDBG() << "Vectorization of flattened convs with dynamic shapes is not "
              "supported";
    return failure();
  }

  if (!isaConvolutionOpOfType<linalg::DepthwiseConv1DNwcWcOp>(conv)) {
    LDBG() << "Not a 1D depth-wise WC conv, dynamic shapes are not supported";
    return failure();
  }

  // Support dynamic shapes in 1D depthwise convolution, but only in the
  // _channel_ dimension.
  Value lhs = conv.getDpsInputOperand(0)->get();
  ArrayRef<int64_t> lhsShape = cast<ShapedType>(lhs.getType()).getShape();
  auto shapeWithoutCh = lhsShape.drop_back(1);
  if (ShapedType::isDynamicShape(shapeWithoutCh)) {
    LDBG() << "Dynamically-shaped op vectorization precondition failed: only "
              "channel dim can be dynamic";
    return failure();
  }

  return success();
}

static LogicalResult
vectorizeDynamicLinalgOpPrecondition(linalg::LinalgOp op,
                                     bool flatten1DDepthwiseConv) {
  if (isaConvolutionOpInterface(op))
    return vectorizeDynamicConvOpPrecondition(op, flatten1DDepthwiseConv);

  if (hasReductionIterator(op))
    return reductionPreconditions(op);

  // TODO: Masking only supports dynamic element-wise ops, linalg.generic ops,
  // linalg.copy ops and ops that implement ContractionOpInterface for now.
  if (!isElementwise(op) &&
      !isa<linalg::GenericOp, linalg::CopyOp, linalg::ContractionOpInterface>(
          op.getOperation()))
    return failure();

  LDBG() << "Dynamically-shaped op meets vectorization pre-conditions";
  return success();
}

//// This hook considers two cases:
///   (1) If the input-vector-sizes are empty, then the vector sizes will be
///       infered. This is only possible when all shapes are static.
///   (2) If the input-vector-sizes are non-empty (i.e. user provided), then
///       carry out basic sanity-checking.
static LogicalResult
vectorizeUnPackOpPrecondition(linalg::UnPackOp unpackOp,
                              ArrayRef<int64_t> inputVectorSizes) {
  // TODO: Support Memref UnPackOp. Temporarily return failure.
  if (!unpackOp.hasPureTensorSemantics())
    return failure();

  // If there are no input vector sizes and all shapes are static, there is
  // nothing left to check.
  if (inputVectorSizes.empty() && unpackOp.getDestType().hasStaticShape() &&
      unpackOp.getSourceType().hasStaticShape())
    return success();

  // The number of input vector sizes must be equal to:
  //  * read-vector-rank
  if (!inputVectorSizes.empty() &&
      (inputVectorSizes.size() != unpackOp.getSourceRank())) {
    LDBG() << "Incorrect number of input vector sizes";
    return failure();
  }

  // Check the vector sizes for the read operation.
  if (failed(vector::isValidMaskedInputVector(
          unpackOp.getSourceType().getShape(), inputVectorSizes))) {
    LDBG() << "Invalid vector sizes for the read operation";
    return failure();
  }

  return success();
}

static LogicalResult
vectorizeInsertSliceOpPrecondition(tensor::InsertSliceOp sliceOp,
                                   ArrayRef<int64_t> inputVectorSizes) {

  TypedValue<RankedTensorType> source = sliceOp.getSource();
  auto sourceType = source.getType();
  if (!VectorType::isValidElementType(sourceType.getElementType()))
    return failure();

  // Get the pad value.
  // TransferReadOp (which is used to vectorize InsertSliceOp), requires a
  // scalar padding value. Note that:
  //    * for in-bounds accesses,
  // the value is actually irrelevant. There are 2 cases in which xfer.read
  // accesses are known to be in-bounds:
  //  1. The source shape is static (output vector sizes would be based on
  //     the source shape and hence all memory accesses would be in-bounds),
  //  2. Masking is used, i.e. the output vector sizes are user-provided. In
  //     this case it is safe to assume that all memory accesses are in-bounds.
  //
  // When the value is not known and not needed, use 0. Otherwise, bail out.
  Value padValue = getStaticPadVal(sliceOp);
  bool isOutOfBoundsRead =
      !sourceType.hasStaticShape() && inputVectorSizes.empty();

  if (!padValue && isOutOfBoundsRead) {
    LDBG() << "Failed to get a pad value for out-of-bounds read access";
    return failure();
  }
  return success();
}

/// Vectorize a named linalg contraction op into:
///   vector::TransferReadOp - Reads vectors from the operands
///   vector::ContractionOp - Performs contraction
///   vector::TransferWriteOp - Write the result vector back to the
///   destination
/// The operands shapes are preserved and loaded directly into vectors.
/// Any further permutations or numerical casting remain within contraction op.
static LogicalResult
vectorizeAsLinalgContraction(RewriterBase &rewriter, VectorizationState &state,
                             LinalgOp linalgOp,
                             SmallVectorImpl<Value> &newResults) {
  Location loc = linalgOp.getLoc();
  MLIRContext *ctx = linalgOp.getContext();

  // For simplicity, contraction vectorization is limited to linalg named ops.
  // Generic op is ignored as not every arbitrary contraction body can be
  // expressed by a vector.contract.
  if (!isa<ContractionOpInterface>(linalgOp.getOperation()))
    return failure();

  OpOperand *outOperand = linalgOp.getDpsInitOperand(0);
  Operation *reduceOp = matchLinalgReduction(outOperand);
  auto maybeKind = getCombinerOpKind(reduceOp);
  if (!maybeKind) {
    LDBG() << "Failed to determine contraction combining kind.";
    return failure();
  }

  // Check that all dimensions are present in the input operands.
  // Arbitrary broadcasts are not supported by the vector contraction.
  // Broadcasts are expected to be decomposed before vectorization.
  AffineMap lhsMap = linalgOp.getIndexingMapsArray()[0];
  AffineMap rhsMap = linalgOp.getIndexingMapsArray()[1];
  if (getUnusedDimsBitVector({lhsMap, rhsMap}).any()) {
    LDBG() << "Contractions with broadcasts are not supported.";
    return failure();
  }

  // Load operands.
  SmallVector<Value> vecOperands;
  for (OpOperand &opOperand : linalgOp->getOpOperands()) {
    // The operand vector shape is computed by mapping the canonical vector
    // shape to the operand's domain. Further permutations are left as a part of
    // the contraction.
    AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&opOperand);
    AffineMap readMap = AffineMap::getMultiDimIdentityMap(
        indexingMap.getNumResults(), rewriter.getContext());
    Type elemType = getElementTypeOrSelf(opOperand.get());
    VectorType readType =
        state.getCanonicalVecType(elemType, readMap.compose(indexingMap));

    Value read = mlir::vector::createReadOrMaskedRead(
        rewriter, loc, opOperand.get(), readType,
        /*padding=*/arith::getZeroConstant(rewriter, loc, elemType),
        /*useInBoundsInsteadOfMasking=*/false);
    vecOperands.push_back(read);
  }

  // Remap iterators from linalg to vector.
  SmallVector<Attribute> iterAttrs;
  auto iterators = linalgOp.getIteratorTypesArray();
  for (utils::IteratorType iter : iterators) {
    auto vecIter = iter == utils::IteratorType::parallel
                       ? vector::IteratorType::parallel
                       : vector::IteratorType::reduction;
    iterAttrs.push_back(vector::IteratorTypeAttr::get(ctx, vecIter));
  }

  // Create contraction.
  Operation *contractOp = vector::ContractionOp::create(
      rewriter, loc, /*lhs=*/vecOperands[0],
      /*rhs=*/vecOperands[1], /*acc=*/vecOperands[2],
      linalgOp.getIndexingMaps(), rewriter.getArrayAttr(iterAttrs), *maybeKind);
  contractOp = state.maskOperation(rewriter, contractOp, linalgOp);

  // Store result.
  Operation *write = vector::createWriteOrMaskedWrite(
      rewriter, loc, contractOp->getResult(0), outOperand->get());

  // Finalize.
  if (!write->getResults().empty())
    newResults.push_back(write->getResult(0));

  return success();
}

namespace {
enum class ConvOperationKind { Conv, Pool };
} // namespace

static bool isCastOfBlockArgument(Operation *op) {
  return isa<CastOpInterface>(op) && op->getNumOperands() == 1 &&
         isa<BlockArgument>(op->getOperand(0));
}

// Returns the ConvOperationKind of the op using reduceOp of the generic
// payload. If it is neither a convolution nor a pooling, it returns
// std::nullopt.
//
// If (region has 2 ops (reduction + yield) or 3 ops (extension + reduction
// + yield) and rhs is not used) then it is the body of a pooling
// If conv, check for single `mul` predecessor. The `mul` operands must be
// block arguments or extension of block arguments.
// Otherwise, check for one or zero `ext` predecessor. The `ext` operands
// must be block arguments or extension of block arguments.
static std::optional<ConvOperationKind>
getConvOperationKind(Operation *reduceOp) {
  int numBlockArguments =
      llvm::count_if(reduceOp->getOperands(), llvm::IsaPred<BlockArgument>);

  switch (numBlockArguments) {
  case 1: {
    // Will be convolution if feeder is a MulOp.
    // A strength reduced version of MulOp for i1 type is AndOp which is also
    // supported. Otherwise, it can be pooling. This strength reduction logic
    // is in `buildBinaryFn` helper in the Linalg dialect.
    auto feedValIt = llvm::find_if_not(reduceOp->getOperands(),
                                       llvm::IsaPred<BlockArgument>);
    assert(feedValIt != reduceOp->operand_end() &&
           "Expected a non-block argument operand");
    Operation *feedOp = (*feedValIt).getDefiningOp();
    if (isCastOfBlockArgument(feedOp)) {
      return ConvOperationKind::Pool;
    }

    if (!((isa<arith::MulIOp, arith::MulFOp>(feedOp) ||
           (isa<arith::AndIOp>(feedOp) &&
            feedOp->getResultTypes()[0].isInteger(1))) &&
          llvm::all_of(feedOp->getOperands(), [](Value v) {
            if (isa<BlockArgument>(v))
              return true;
            if (Operation *op = v.getDefiningOp())
              return isCastOfBlockArgument(op);
            return false;
          }))) {
      return std::nullopt;
    }

    return ConvOperationKind::Conv;
  }
  case 2:
    // Must be pooling
    return ConvOperationKind::Pool;
  default:
    return std::nullopt;
  }
}

static bool isSupportedPoolKind(vector::CombiningKind kind) {
  switch (kind) {
  case vector::CombiningKind::ADD:
  case vector::CombiningKind::MAXNUMF:
  case vector::CombiningKind::MAXIMUMF:
  case vector::CombiningKind::MAXSI:
  case vector::CombiningKind::MAXUI:
  case vector::CombiningKind::MINNUMF:
  case vector::CombiningKind::MINIMUMF:
  case vector::CombiningKind::MINSI:
  case vector::CombiningKind::MINUI:
    return true;
  default:
    return false;
  }
}

namespace {
/// Classification of a 1D convolution/pooling derived from
/// ConvolutionDimensions. Cached state used by every step of the rewrite
/// pipeline.
struct Conv1DConfig {
  ConvolutionDimensions dims;
  ConvOperationKind convKind = ConvOperationKind::Conv;
  ConvLayoutKind layout = ConvLayoutKind::Scalar;
  bool isDepthwise = false;
  bool isPooling = false;
  /// Subset of `dims.batch` treated as outer (N-like) batch dims.
  SmallVector<unsigned> nLikeBatch;
  /// Subset of `dims.batch` treated as the channel-like dim for pooling.
  /// Derived from the LHS indexing map (innermost LHS position) by
  /// `splitPoolBatchByLhsInnermost`, not from the order of `dims.batch`.
  SmallVector<unsigned> cLikeBatch;
};

/// Sizes extracted from the canonical 1D conv/pool form.
struct Conv1DSizes {
  int64_t kw = 0;
  int64_t w = 0;
  int64_t n = 0;
  int64_t c = 0;
  int64_t f = 0;
};

/// Permutations from the original operand layouts to the canonical NWC form.
/// An empty vector means the permutation is identity (no transpose needed).
struct Conv1DPerms {
  SmallVector<int64_t> lhs;
  SmallVector<int64_t> rhs;
  SmallVector<int64_t> res;

  bool allIdentity() const { return lhs.empty() && rhs.empty() && res.empty(); }
};

/// Pre-permutation (i.e. as-stored-in-tensor) shapes used for transfer_read.
struct Conv1DShapes {
  SmallVector<int64_t> lhs;
  SmallVector<int64_t> rhs;
  SmallVector<int64_t> res;
};

/// Triple of operand vector values (in canonical NWC form). `rhs` is null
/// for pooling.
struct Conv1DOperandVecs {
  Value lhs;
  Value rhs;
  Value res;
};
} // namespace

/// Build a map from loop-dim position to tensor-dim position for the dims of
/// `map` that are plain `AffineDimExpr`s. Non-plain results (e.g. strided
/// access expressions) are skipped.
static DenseMap<unsigned, unsigned> buildLoopToTensorDimMap(AffineMap map) {
  DenseMap<unsigned, unsigned> result;
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i)))
      result[dimExpr.getPosition()] = i;
  }
  return result;
}

/// For a pool-like op (no inputChannel/outputChannel dims), split `batch`
/// into (outer, channel-like) based on the LHS indexing map: the batch loop
/// that appears at the innermost LHS tensor position plays the channel-like
/// role in the canonical NWC vector form; all other batch loops are outer
/// (N-like) batch dims. This derivation is independent of the order in
/// which `inferConvolutionDims` returns `batch`, so equivalent pools written
/// with different iterator orderings classify the same way.
///
/// Returns failure when:
///   - any batch loop is missing from the LHS indexing map (malformed pool);
///   - two batch loops tie at the innermost LHS position (ambiguous).
static FailureOr<std::pair<SmallVector<unsigned>, SmallVector<unsigned>>>
splitPoolBatchByLhsInnermost(LinalgOp linalgOp, ArrayRef<unsigned> batch) {
  DenseMap<unsigned, unsigned> loopToLhs =
      buildLoopToTensorDimMap(linalgOp.getIndexingMapsArray()[0]);

  std::optional<unsigned> cLike;
  unsigned maxPos = 0;
  for (unsigned d : batch) {
    auto it = loopToLhs.find(d);
    if (it == loopToLhs.end())
      return failure();
    unsigned p = it->second;
    if (!cLike || p > maxPos) {
      cLike = d;
      maxPos = p;
    } else if (p == maxPos) {
      return failure();
    }
  }

  SmallVector<unsigned> nLike, cLikeOut;
  for (unsigned d : batch) {
    if (d == *cLike)
      cLikeOut.push_back(d);
    else
      nLike.push_back(d);
  }
  return std::make_pair(std::move(nLike), std::move(cLikeOut));
}

/// Classify `op` as a vectorizable 1D convolution/pooling. On failure,
/// optionally writes a human-readable reason into `*reason`.
static FailureOr<Conv1DConfig> classifyAs1DConv(LinalgOp op,
                                                std::string *reason = nullptr) {
  auto fail = [&](StringRef msg) -> FailureOr<Conv1DConfig> {
    if (reason)
      *reason = msg.str();
    return failure();
  };

  FailureOr<ConvolutionDimensions> maybeDims = inferConvolutionDims(op);
  if (failed(maybeDims))
    return fail("op is not a convolution (inferConvolutionDims failed)");

  // Restrict to 1D: exactly one filter-loop dim and one output-image dim.
  if (maybeDims->filterLoop.size() != 1 || maybeDims->outputImage.size() != 1)
    return fail("only 1D convolutions are supported");
  if (maybeDims->strides.empty() || maybeDims->dilations.empty())
    return fail("strides and dilations must be statically known");

  // Channel symmetry: either both channel kinds present or both folded away.
  if (maybeDims->inputChannel.empty() != maybeDims->outputChannel.empty())
    return fail("asymmetric inputChannel/outputChannel dims");

  // Multiplicity restrictions: every category we index later assumes <= 1.
  if (maybeDims->inputChannel.size() > 1 || maybeDims->outputChannel.size() > 1)
    return fail("multiple inputChannel/outputChannel dims are not supported");
  if (maybeDims->depth.size() > 1)
    return fail("multiple depth dims are not supported");

  // Inspect the reduce body to know whether this is a conv or a pool.
  Operation *reduceOp = matchLinalgReduction(op.getDpsInitOperand(0));
  if (!reduceOp)
    return fail("could not match the reduction in the op body");
  std::optional<ConvOperationKind> maybeKind = getConvOperationKind(reduceOp);
  if (!maybeKind)
    return fail("unsupported reduction body (not a conv or pool kind)");

  Conv1DConfig config;
  config.dims = std::move(*maybeDims);
  config.convKind = *maybeKind;
  config.isDepthwise = !config.dims.depth.empty();

  bool bothChannelsEmpty =
      config.dims.inputChannel.empty() && config.dims.outputChannel.empty();
  bool noBatchOrChannels =
      bothChannelsEmpty && config.dims.batch.empty() && !config.isDepthwise;

  // Pure non-channeled 1D convolution: lhs=[iw], rhs=[kw], res=[w].
  if (noBatchOrChannels) {
    config.layout = ConvLayoutKind::Scalar;
    return config;
  }

  // Both channel dims folded away with a batch dim => structurally pooling.
  // Reject if the body is actually a multiply-add convolution (it would need
  // channel dims that do not exist).
  if (bothChannelsEmpty && !config.isDepthwise &&
      config.convKind != ConvOperationKind::Pool)
    return fail("convolution body but no channel dims (use a pooling body or "
                "add channel dims)");
  config.isPooling = bothChannelsEmpty && !config.isDepthwise;

  // Batch-dim splitting.
  //
  // For pooling, the op body alone cannot distinguish "N" from "C": both are
  // parallel loops on LHS and result that do not appear on RHS, so
  // `inferConvolutionDims` groups them together in `dims.batch`. We derive
  // which one plays the channel-like role from the LHS tensor layout
  // (innermost LHS position) via `splitPoolBatchByLhsInnermost`, so that the
  // classification does not depend on the order of `dims.batch`.
  if (config.isPooling && !config.dims.batch.empty()) {
    auto split = splitPoolBatchByLhsInnermost(op, config.dims.batch);
    if (failed(split))
      return fail("cannot determine channel-like batch dim for pooling op "
                  "(ambiguous LHS layout or batch dim missing from LHS)");
    std::tie(config.nLikeBatch, config.cLikeBatch) = std::move(*split);
  } else if (!config.isPooling) {
    config.nLikeBatch.assign(config.dims.batch.begin(),
                             config.dims.batch.end());
  }

  // Multi-batch is not supported by the current canonical shapes.
  if (config.nLikeBatch.size() > 1)
    return fail("more than one N-like batch dim is not supported");

  config.layout = config.nLikeBatch.empty() ? ConvLayoutKind::Batchless
                                            : ConvLayoutKind::Batched;
  return config;
}

static LogicalResult vectorizeConvOpPrecondition(linalg::LinalgOp convOp) {
  // Single shared classifier ensures the precondition and the matcher in
  // Conv1DGenerator::create cannot drift apart.
  FailureOr<Conv1DConfig> maybeConfig = classifyAs1DConv(convOp);
  if (failed(maybeConfig))
    return failure();

  Operation *reduceOp = matchLinalgReduction(convOp.getDpsInitOperand(0));
  std::optional<vector::CombiningKind> maybeKind = getCombinerOpKind(reduceOp);
  // Typically convolution will have a `Add` CombiningKind but for i1 type it
  // can get strength reduced to `OR` which is also supported. This strength
  // reduction logic is in `buildBinaryFn` helper in the Linalg dialect.
  if (!maybeKind || ((*maybeKind != vector::CombiningKind::ADD &&
                      *maybeKind != vector::CombiningKind::OR) &&
                     (maybeConfig->convKind != ConvOperationKind::Pool ||
                      !isSupportedPoolKind(*maybeKind))))
    return failure();

  ShapedType rhsShapedType =
      cast<ShapedType>(convOp.getDpsInputOperand(1)->get().getType());
  int64_t rhsRank = rhsShapedType.getRank();
  if (maybeConfig->convKind == ConvOperationKind::Pool) {
    if (rhsRank != 1)
      return failure();
  } else {
    if (rhsRank != 1 && rhsRank != 2 && rhsRank != 3)
      return failure();
  }

  return success();
}

static LogicalResult vectorizeLinalgOpPrecondition(
    LinalgOp linalgOp, ArrayRef<int64_t> inputVectorSizes,
    bool vectorizeNDExtract, bool flatten1DDepthwiseConv) {
  // tensor with dimension of 0 cannot be vectorized.
  if (llvm::any_of(linalgOp->getOpOperands(), [&](OpOperand &operand) {
        return llvm::is_contained(linalgOp.getShape(&operand), 0);
      }))
    return failure();
  // Check API contract for input vector sizes.
  if (!inputVectorSizes.empty() &&
      failed(vector::isValidMaskedInputVector(linalgOp.getStaticLoopRanges(),
                                              inputVectorSizes)))
    return failure();

  if (linalgOp.hasDynamicShape() && failed(vectorizeDynamicLinalgOpPrecondition(
                                        linalgOp, flatten1DDepthwiseConv))) {
    LDBG() << "Dynamically-shaped op failed vectorization pre-conditions";
    return failure();
  }

  SmallVector<CustomVectorizationPrecondition> customPreconditions;

  // Register CustomVectorizationPrecondition for extractOp.
  customPreconditions.push_back(tensorExtractVectorizationPrecondition);

  // All types in the body should be a supported element type for VectorType.
  for (Operation &innerOp : linalgOp->getRegion(0).front()) {
    // Check if any custom hook can vectorize the inner op.
    if (llvm::any_of(
            customPreconditions,
            [&](const CustomVectorizationPrecondition &customPrecondition) {
              return succeeded(
                  customPrecondition(&innerOp, vectorizeNDExtract));
            })) {
      continue;
    }
    if (!llvm::all_of(innerOp.getOperandTypes(),
                      VectorType::isValidElementType)) {
      return failure();
    }
    if (!llvm::all_of(innerOp.getResultTypes(),
                      VectorType::isValidElementType)) {
      return failure();
    }
  }
  if (isElementwise(linalgOp))
    return success();

  // Check for both named as well as generic convolution ops.
  if (isaConvolutionOpInterface(linalgOp))
    return vectorizeConvOpPrecondition(linalgOp);

  // TODO: the common vector shape is equal to the static loop sizes only when
  // all indexing maps are projected permutations. For convs and stencils the
  // logic will need to evolve.
  if (!allIndexingsAreProjectedPermutation(linalgOp)) {
    LDBG() << "precondition failed: not projected permutations";
    return failure();
  }
  if (failed(reductionPreconditions(linalgOp))) {
    LDBG() << "precondition failed: reduction preconditions";
    return failure();
  }
  return success();
}

static LogicalResult
vectorizePackOpPrecondition(linalg::PackOp packOp,
                            ArrayRef<int64_t> inputVectorSizes) {
  // TODO: Support Memref PackOp. Temporarily return failure.
  if (!packOp.hasPureTensorSemantics())
    return failure();

  auto padValue = packOp.getPaddingValue();
  Attribute cstAttr;
  // TODO: Relax this condiiton
  if (padValue && !matchPattern(padValue, m_Constant(&cstAttr))) {
    LDBG() << "pad value is not constant: " << packOp;
    return failure();
  }

  ArrayRef<int64_t> resultTensorShape = packOp.getDestType().getShape();
  bool satisfyEmptyCond = true;
  if (inputVectorSizes.empty()) {
    if (!packOp.getDestType().hasStaticShape() ||
        !packOp.getSourceType().hasStaticShape())
      satisfyEmptyCond = false;
  }

  if (!satisfyEmptyCond &&
      failed(vector::isValidMaskedInputVector(
          resultTensorShape.take_front(packOp.getSourceRank()),
          inputVectorSizes)))
    return failure();

  if (llvm::any_of(packOp.getInnerTiles(), [](OpFoldResult v) {
        return !getConstantIntValue(v).has_value();
      })) {
    LDBG() << "inner_tiles must be constant: " << packOp;
    return failure();
  }

  return success();
}

static LogicalResult
vectorizePadOpPrecondition(tensor::PadOp padOp,
                           ArrayRef<int64_t> inputVectorSizes) {
  auto padValue = padOp.getConstantPaddingValue();
  if (!padValue) {
    LDBG() << "pad value is not constant: " << padOp;
    return failure();
  }

  ArrayRef<int64_t> resultTensorShape = padOp.getResultType().getShape();
  if (failed(vector::isValidMaskedInputVector(resultTensorShape,
                                              inputVectorSizes)))
    return failure();

  // Padding with non-zero low pad values is not supported, unless the
  // corresponding result dim is 1 as this would require shifting the results to
  // the right for the low padded dims by the required amount of low padding.
  // However, we do support low padding if the dims being low padded have result
  // sizes of 1. The reason is when we have a low pad on a unit result dim, the
  // input size of that dimension will be dynamically zero (as the sum of the
  // low pad and input dim size has to be one) and hence we will create a zero
  // mask as the lowering logic just makes the mask one for the input dim size -
  // which is zero here. Hence we will load the pad value which is what we want
  // in this case. If the low pad is dynamically zero then the lowering is
  // correct as well as no shifts are necessary.
  if (llvm::any_of(llvm::enumerate(padOp.getMixedLowPad()),
                   [&](const auto &en) {
                     OpFoldResult padValue = en.value();
                     unsigned pos = en.index();
                     std::optional<int64_t> pad = getConstantIntValue(padValue);
                     return (!pad.has_value() || pad.value() != 0) &&
                            resultTensorShape[pos] != 1;
                   })) {
    LDBG() << "low pad must all be zero for all non unit dims: " << padOp;
    return failure();
  }

  return success();
}

/// Preconditions for scalable vectors.
///
/// For Ops implementing the LinalgOp interface, this is quite restrictive - it
/// models the fact that in practice we would only make selected dimensions
/// scalable. For other Ops (e.g. `linalg.unpack`), this will succeed
/// unconditionally - we are yet to identify meaningful conditions.
static LogicalResult
vectorizeScalableVectorPrecondition(Operation *op,
                                    ArrayRef<int64_t> inputVectorSizes,
                                    ArrayRef<bool> inputScalableVecDims) {
  assert(inputVectorSizes.size() == inputScalableVecDims.size() &&
         "Number of input vector sizes and scalable dims doesn't match");

  size_t numOfScalableDims =
      llvm::count_if(inputScalableVecDims, [](bool flag) { return flag; });

  if (numOfScalableDims == 0)
    return success();

  auto linalgOp = dyn_cast<LinalgOp>(op);

  // Cond 1: Reject Ops that don't implement the LinalgOp interface, with the
  // exception of UnpackOp for which there is a dedicated hook.
  if (!linalgOp) {
    return success(isa<linalg::UnPackOp>(op));
  }

  // Cond 2: There's been no need for more than 2 scalable dims so far
  if (numOfScalableDims > 2)
    return failure();

  // Cond 3: Look at the configuration in `inputScalableVecDims` and verify that
  // it matches one of the supported cases:
  //  1. Exactly 1 dim is scalable and that's the _last_ non-unit parallel dim
  //    (*).
  //  2. Exactly 2 dims are scalable and those are the _last two adjacent_
  //     parallel dims.
  //  3. Exactly 1 reduction dim is scalable and that's the last (innermost)
  //  dim.
  // The 2nd restriction above means that only Matmul-like Ops are supported
  // when 2 dims are scalable, e.g. :
  //    * iterators = [parallel, parallel, reduction]
  //    * scalable flags = [true, true, false]
  //
  // (*) Non-unit dims get folded away in practice.
  // TODO: Relax these conditions as good motivating examples are identified.

  // Find the first scalable flag.
  bool seenNonUnitParallel = false;
  auto iterators = linalgOp.getIteratorTypesArray();
  SmallVector<bool> scalableFlags(inputScalableVecDims);
  int64_t idx = scalableFlags.size() - 1;
  while (!scalableFlags[idx]) {
    bool isNonUnitDim = (inputVectorSizes[idx] != 1);
    seenNonUnitParallel |=
        (iterators[idx] == utils::IteratorType::parallel && isNonUnitDim);

    iterators.pop_back();
    scalableFlags.pop_back();
    --idx;
  }

  // Analyze the iterator corresponding to the first scalable dim.
  switch (iterators.back()) {
  case utils::IteratorType::reduction: {
    // Check 3. above is met.
    if (iterators.size() != inputVectorSizes.size()) {
      LDBG() << "Non-trailing reduction dim requested for scalable "
                "vectorization";
      return failure();
    }
    if (isa<linalg::MatmulOp>(op)) {
      LDBG()
          << "Scalable vectorization of the reduction dim in Matmul-like ops "
             "is not supported";
      return failure();
    }
    break;
  }
  case utils::IteratorType::parallel: {
    // Check 1. and 2. above are met.
    if (seenNonUnitParallel) {
      LDBG() << "Inner parallel dim not requested for scalable "
                "vectorization";
      return failure();
    }
    break;
  }
  }

  // If present, check the 2nd scalable dim. ATM, only Matmul-like Ops are
  // supported for which expect the folowing config:
  //    * iterators = [parallel, parallel, reduction]
  //    * scalable flags = [true, true, false]
  if (numOfScalableDims == 2) {
    // Disallow below case which breaks 3. above:
    //    * iterators = [..., parallel, reduction]
    //    * scalable flags = [..., true, true]
    if (iterators.back() == utils::IteratorType::reduction) {
      LDBG() << "Higher dim than the trailing reduction dim requested for "
                "scalable "
                "vectorizatio";
      return failure();
    }
    scalableFlags.pop_back();
    iterators.pop_back();

    if (!scalableFlags.back() ||
        (iterators.back() != utils::IteratorType::parallel))
      return failure();
  }

  // Cond 4: Only the following ops are supported in the
  // presence of scalable vectors
  return success(
      isElementwise(linalgOp) || isa<linalg::MatmulOp>(op) ||
      isa<linalg::BatchMatmulOp>(op) ||
      isaConvolutionOpOfType<linalg::DepthwiseConv1DNwcWcOp>(linalgOp) ||
      isa<linalg::MatvecOp>(op) || isa<linalg::Mmt4DOp>(op) ||
      isa<linalg::BatchMmt4DOp>(op) || hasReductionIterator(linalgOp));
}

LogicalResult mlir::linalg::vectorizeOpPrecondition(
    Operation *op, ArrayRef<int64_t> inputVectorSizes,
    ArrayRef<bool> inputScalableVecDims, bool vectorizeNDExtract,
    bool flatten1DDepthwiseConv) {

  if (!hasVectorizationImpl(op))
    return failure();

  if (failed(vectorizeScalableVectorPrecondition(op, inputVectorSizes,
                                                 inputScalableVecDims)))
    return failure();

  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](linalg::LinalgOp linalgOp) {
        return vectorizeLinalgOpPrecondition(linalgOp, inputVectorSizes,
                                             vectorizeNDExtract,
                                             flatten1DDepthwiseConv);
      })
      .Case([&](tensor::PadOp padOp) {
        return vectorizePadOpPrecondition(padOp, inputVectorSizes);
      })
      .Case([&](linalg::PackOp packOp) {
        return vectorizePackOpPrecondition(packOp, inputVectorSizes);
      })
      .Case([&](linalg::UnPackOp unpackOp) {
        return vectorizeUnPackOpPrecondition(unpackOp, inputVectorSizes);
      })
      .Case([&](tensor::InsertSliceOp sliceOp) {
        return vectorizeInsertSliceOpPrecondition(sliceOp, inputVectorSizes);
      })
      .Default(failure());
}

/// Converts affine.apply Ops to arithmetic operations.
static void convertAffineApply(RewriterBase &rewriter, LinalgOp linalgOp) {
  OpBuilder::InsertionGuard g(rewriter);
  auto toReplace = linalgOp.getBlock()->getOps<affine::AffineApplyOp>();

  for (auto op : make_early_inc_range(toReplace)) {
    rewriter.setInsertionPoint(op);
    auto expanded = affine::expandAffineExpr(
        rewriter, op->getLoc(), op.getAffineMap().getResult(0),
        op.getOperands().take_front(op.getAffineMap().getNumDims()),
        op.getOperands().take_back(op.getAffineMap().getNumSymbols()));
    rewriter.replaceOp(op, expanded);
  }
}

bool mlir::linalg::hasVectorizationImpl(Operation *op) {
  return isa<linalg::LinalgOp, tensor::PadOp, linalg::PackOp, linalg::UnPackOp,
             tensor::InsertSliceOp>(op);
}

FailureOr<VectorizationResult> mlir::linalg::vectorize(
    RewriterBase &rewriter, Operation *op, ArrayRef<int64_t> inputVectorSizes,
    ArrayRef<bool> inputScalableVecDims, bool vectorizeNDExtract,
    bool flatten1DDepthwiseConv, bool assumeDynamicDimsMatchVecSizes,
    bool createNamedContraction) {
  LDBG() << "Attempting to vectorize: " << *op;
  LDBG() << "Input vector sizes: " << llvm::interleaved(inputVectorSizes);
  LDBG() << "Input scalable vector dims: "
         << llvm::interleaved(inputScalableVecDims);

  if (failed(vectorizeOpPrecondition(op, inputVectorSizes, inputScalableVecDims,
                                     vectorizeNDExtract,
                                     flatten1DDepthwiseConv))) {
    LDBG() << "Vectorization pre-conditions failed";
    return failure();
  }

  // Initialize vectorization state.
  VectorizationState state(rewriter);
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (failed(state.initState(rewriter, linalgOp, inputVectorSizes,
                               inputScalableVecDims,
                               assumeDynamicDimsMatchVecSizes))) {
      LDBG() << "Vectorization state couldn't be initialized";
      return failure();
    }
  }

  SmallVector<Value> results;
  auto vectorizeResult =
      TypeSwitch<Operation *, LogicalResult>(op)
          .Case([&](linalg::LinalgOp linalgOp) {
            // Check for both named as well as generic convolution ops.
            if (isaConvolutionOpInterface(linalgOp)) {
              FailureOr<Operation *> convOr = vectorizeConvolution(
                  rewriter, linalgOp, inputVectorSizes, inputScalableVecDims,
                  flatten1DDepthwiseConv);
              if (succeeded(convOr)) {
                llvm::append_range(results, (*convOr)->getResults());
                return success();
              }

              LDBG() << "Unsupported convolution can't be vectorized.";
              return failure();
            }

            if (createNamedContraction &&
                isa<ContractionOpInterface>(linalgOp.getOperation()))
              return vectorizeAsLinalgContraction(rewriter, state, linalgOp,
                                                  results);

            LDBG()
                << "Vectorize generic by broadcasting to the canonical vector "
                   "shape";

            // Pre-process before proceeding.
            convertAffineApply(rewriter, linalgOp);

            // TODO: 'vectorize' takes in a 'RewriterBase' which is up-casted
            // to 'OpBuilder' when it is passed over to some methods like
            // 'vectorizeAsLinalgGeneric'. This is highly problematic: if we
            // erase an op within these methods, the actual rewriter won't be
            // notified and we will end up with read-after-free issues!
            return vectorizeAsLinalgGeneric(rewriter, state, linalgOp, results);
          })
          .Case([&](tensor::PadOp padOp) {
            return vectorizeAsTensorPadOp(rewriter, padOp, inputVectorSizes,
                                          results);
          })
          .Case([&](linalg::PackOp packOp) {
            return vectorizeAsTensorPackOp(rewriter, packOp, inputVectorSizes,
                                           results);
          })
          .Case([&](linalg::UnPackOp unpackOp) {
            return vectorizeAsTensorUnpackOp(rewriter, unpackOp,
                                             inputVectorSizes,
                                             inputScalableVecDims, results);
          })
          .Case([&](tensor::InsertSliceOp sliceOp) {
            return vectorizeAsInsertSliceOp(rewriter, sliceOp, inputVectorSizes,
                                            results);
          })
          .Default(failure());

  if (failed(vectorizeResult)) {
    LDBG() << "Vectorization failed";
    return failure();
  }

  return VectorizationResult{results};
}

LogicalResult mlir::linalg::vectorizeCopy(RewriterBase &rewriter,
                                          memref::CopyOp copyOp) {
  auto srcType = cast<MemRefType>(copyOp.getSource().getType());
  auto dstType = cast<MemRefType>(copyOp.getTarget().getType());
  if (!srcType.hasStaticShape() || !dstType.hasStaticShape())
    return failure();

  auto srcElementType = getElementTypeOrSelf(srcType);
  auto dstElementType = getElementTypeOrSelf(dstType);
  if (!VectorType::isValidElementType(srcElementType) ||
      !VectorType::isValidElementType(dstElementType))
    return failure();

  auto readType = VectorType::get(srcType.getShape(), srcElementType);
  auto writeType = VectorType::get(dstType.getShape(), dstElementType);

  Location loc = copyOp->getLoc();
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> indices(srcType.getRank(), zero);

  Value readValue = vector::TransferReadOp::create(
      rewriter, loc, readType, copyOp.getSource(), indices,
      /*padding=*/std::nullopt,
      rewriter.getMultiDimIdentityMap(srcType.getRank()));
  if (cast<VectorType>(readValue.getType()).getRank() == 0) {
    readValue = vector::ExtractOp::create(rewriter, loc, readValue,
                                          ArrayRef<int64_t>());
    readValue =
        vector::BroadcastOp::create(rewriter, loc, writeType, readValue);
  }
  Operation *writeValue = vector::TransferWriteOp::create(
      rewriter, loc, readValue, copyOp.getTarget(), indices,
      rewriter.getMultiDimIdentityMap(srcType.getRank()));
  rewriter.replaceOp(copyOp, writeValue->getResults());
  return success();
}

//----------------------------------------------------------------------------//
// Misc. vectorization patterns.
//----------------------------------------------------------------------------//
/// Base pattern for rewriting tensor::PadOps whose result is consumed by a
/// given operation type OpTy.
template <typename OpTy>
struct VectorizePadOpUserPattern : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const final {
    bool changed = false;
    // Insert users in vector, because some users may be replaced/removed.
    for (auto *user : llvm::to_vector<4>(padOp->getUsers()))
      if (auto op = dyn_cast<OpTy>(user))
        changed |= rewriteUser(rewriter, padOp, op).succeeded();
    return success(changed);
  }

protected:
  virtual LogicalResult rewriteUser(PatternRewriter &rewriter,
                                    tensor::PadOp padOp, OpTy op) const = 0;
};

/// Rewrite use of tensor::PadOp result in TransferReadOp. E.g.:
/// ```
/// %0 = tensor.pad %src ... : tensor<?x?xf32> to tensor<17x5xf32>
/// %r = vector.transfer_read %0[%c0, %c0], %cst
///     {in_bounds = [true, true]} : tensor<17x5xf32>, vector<17x5xf32>
/// ```
/// is rewritten to:
/// ```
/// %r = vector.transfer_read %src[%c0, %c0], %padding
///     {in_bounds = [true, true]}
///     : tensor<?x?xf32>, vector<17x5xf32>
/// ```
/// Note: By restricting this pattern to in-bounds TransferReadOps, we can be
/// sure that the original padding value %cst was never used.
///
/// This rewrite is possible if:
/// - `xferOp` has no out-of-bounds dims or mask.
/// - Low padding is static 0.
/// - Single, scalar padding value.
struct PadOpVectorizationWithTransferReadPattern
    : public VectorizePadOpUserPattern<vector::TransferReadOp> {
  using VectorizePadOpUserPattern<
      vector::TransferReadOp>::VectorizePadOpUserPattern;

  LogicalResult rewriteUser(PatternRewriter &rewriter, tensor::PadOp padOp,
                            vector::TransferReadOp xferOp) const override {
    // Low padding must be static 0.
    if (!padOp.hasZeroLowPad())
      return failure();
    // Pad value must be a constant.
    auto padValue = padOp.getConstantPaddingValue();
    if (!padValue)
      return failure();
    // Padding value of existing `xferOp` is unused.
    if (xferOp.hasOutOfBoundsDim() || xferOp.getMask())
      return failure();

    rewriter.modifyOpInPlace(xferOp, [&]() {
      SmallVector<bool> inBounds(xferOp.getVectorType().getRank(), false);
      xferOp->setAttr(xferOp.getInBoundsAttrName(),
                      rewriter.getBoolArrayAttr(inBounds));
      xferOp.getBaseMutable().assign(padOp.getSource());
      xferOp.getPaddingMutable().assign(padValue);
    });

    return success();
  }
};

/// Rewrite use of tensor::PadOp result in TransferWriteOp.
/// This pattern rewrites TransferWriteOps that write to a padded tensor
/// value, where the same amount of padding is immediately removed again after
/// the write. In such cases, the TransferWriteOp can write to the non-padded
/// tensor value and apply out-of-bounds masking. E.g.:
/// ```
/// %0 = tensor.extract_slice ...[...] [%s0, %s1] [1, 1]
///     : tensor<...> to tensor<?x?xf32>
/// %1 = tensor.pad %0 ... : tensor<?x?xf32> to tensor<17x5xf32>
/// %2 = vector.transfer_write %vec, %1[...]
///     : vector<17x5xf32>, tensor<17x5xf32>
/// %r = tensor.extract_slice %2[0, 0] [%s0, %s1] [1, 1]
///     : tensor<17x5xf32> to tensor<?x?xf32>
/// ```
/// is rewritten to:
/// ```
/// %0 = tensor.extract_slice ...[...] [%s0, %s1] [1, 1]
///     : tensor<...> to tensor<?x?xf32>
/// %r = vector.transfer_write %vec, %0[...] : vector<17x5xf32>,
/// tensor<?x?xf32>
/// ```
/// Note: It is important that the ExtractSliceOp %r resizes the result of the
/// TransferWriteOp to the same size as the input of the TensorPadOp (or an
/// even smaller size). Otherwise, %r's new (dynamic) dimensions would differ
/// from %r's old dimensions.
///
/// This rewrite is possible if:
/// - Low padding is static 0.
/// - `xferOp` has exactly one use, which is an ExtractSliceOp. This
///   ExtractSliceOp trims the same amount of padding that was added
///   beforehand.
/// - Single, scalar padding value.
struct PadOpVectorizationWithTransferWritePattern
    : public VectorizePadOpUserPattern<vector::TransferWriteOp> {
  using VectorizePadOpUserPattern<
      vector::TransferWriteOp>::VectorizePadOpUserPattern;

  LogicalResult rewriteUser(PatternRewriter &rewriter, tensor::PadOp padOp,
                            vector::TransferWriteOp xferOp) const override {
    // TODO: support 0-d corner case.
    if (xferOp.getTransferRank() == 0)
      return failure();

    // Low padding must be static 0.
    if (!padOp.hasZeroLowPad())
      return failure();
    // Pad value must be a constant.
    auto padValue = padOp.getConstantPaddingValue();
    if (!padValue)
      return failure();
    // TransferWriteOp result must be directly consumed by an ExtractSliceOp.
    if (!xferOp->hasOneUse())
      return failure();
    auto trimPadding = dyn_cast<tensor::ExtractSliceOp>(*xferOp->user_begin());
    if (!trimPadding)
      return failure();
    // Only static zero offsets supported when trimming padding.
    if (!trimPadding.hasZeroOffset())
      return failure();
    // trimPadding must remove the amount of padding that was added earlier.
    if (!hasSameTensorSize(padOp.getSource(), trimPadding))
      return failure();

    // Insert the new TransferWriteOp at position of the old TransferWriteOp.
    rewriter.setInsertionPoint(xferOp);

    SmallVector<bool> inBounds(xferOp.getVectorType().getRank(), false);
    auto newXferOp = rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        xferOp, padOp.getSource().getType(), xferOp.getVector(),
        padOp.getSource(), xferOp.getIndices(), xferOp.getPermutationMapAttr(),
        xferOp.getMask(), rewriter.getBoolArrayAttr(inBounds));
    rewriter.replaceOp(trimPadding, newXferOp->getResult(0));

    return success();
  }

  /// Check if `beforePadding` and `afterTrimming` have the same tensor size,
  /// i.e., same dimensions.
  ///
  /// Dimensions may be static, dynamic or mix of both. In case of dynamic
  /// dimensions, this function tries to infer the (static) tensor size by
  /// looking at the defining op and utilizing op-specific knowledge.
  ///
  /// This is a conservative analysis. In case equal tensor sizes cannot be
  /// proven statically, this analysis returns `false` even though the tensor
  /// sizes may turn out to be equal at runtime.
  bool hasSameTensorSize(Value beforePadding,
                         tensor::ExtractSliceOp afterTrimming) const {
    // If the input to tensor::PadOp is a CastOp, try with both CastOp
    // result and CastOp operand.
    if (auto castOp = beforePadding.getDefiningOp<tensor::CastOp>())
      if (hasSameTensorSize(castOp.getSource(), afterTrimming))
        return true;

    auto t1 = dyn_cast<RankedTensorType>(beforePadding.getType());
    auto t2 = dyn_cast<RankedTensorType>(afterTrimming.getType());
    // Only RankedTensorType supported.
    if (!t1 || !t2)
      return false;
    // Rank of both values must be the same.
    if (t1.getRank() != t2.getRank())
      return false;

    // All static dimensions must be the same. Mixed cases (e.g., dimension
    // static in `t1` but dynamic in `t2`) are not supported.
    for (unsigned i = 0; i < t1.getRank(); ++i) {
      if (t1.isDynamicDim(i) != t2.isDynamicDim(i))
        return false;
      if (!t1.isDynamicDim(i) && t1.getDimSize(i) != t2.getDimSize(i))
        return false;
    }

    // Nothing more to check if all dimensions are static.
    if (t1.getNumDynamicDims() == 0)
      return true;

    // All dynamic sizes must be the same. The only supported case at the
    // moment is when `beforePadding` is an ExtractSliceOp (or a cast
    // thereof).

    // Apart from CastOp, only ExtractSliceOp is supported.
    auto beforeSlice = beforePadding.getDefiningOp<tensor::ExtractSliceOp>();
    if (!beforeSlice)
      return false;

    assert(static_cast<size_t>(t1.getRank()) ==
           beforeSlice.getMixedSizes().size());
    assert(static_cast<size_t>(t2.getRank()) ==
           afterTrimming.getMixedSizes().size());

    for (unsigned i = 0; i < t1.getRank(); ++i) {
      // Skip static dimensions.
      if (!t1.isDynamicDim(i))
        continue;
      auto size1 = beforeSlice.getMixedSizes()[i];
      auto size2 = afterTrimming.getMixedSizes()[i];

      // Case 1: Same value or same constant int.
      if (isEqualConstantIntOrValue(size1, size2))
        continue;

      // Other cases: Take a deeper look at defining ops of values.
      auto v1 = llvm::dyn_cast_if_present<Value>(size1);
      auto v2 = llvm::dyn_cast_if_present<Value>(size2);
      if (!v1 || !v2)
        return false;

      // Case 2: Both values are identical AffineMinOps. (Should not happen if
      // CSE is run.)
      auto minOp1 = v1.getDefiningOp<affine::AffineMinOp>();
      auto minOp2 = v2.getDefiningOp<affine::AffineMinOp>();
      if (minOp1 && minOp2 && minOp1.getAffineMap() == minOp2.getAffineMap() &&
          minOp1.getOperands() == minOp2.getOperands())
        continue;

      // Add additional cases as needed.
    }

    // All tests passed.
    return true;
  }
};

/// Returns the effective Pad value for the input op, provided it's a scalar.
///
/// Many Ops exhibit pad-like behaviour, but this isn't always explicit. If
/// this Op performs padding, retrieve the padding value provided that it's
/// a scalar and static/fixed for all the padded values. Returns an empty value
/// otherwise.
///
/// TODO: This is used twice (when checking vectorization pre-conditions and
/// when vectorizing). Cache results instead of re-running.
static Value getStaticPadVal(Operation *op) {
  if (!op)
    return {};

  // 1. vector.broadcast (f32 -> vector <...xf32>) - return the value that's
  // being broadcast, provided that it's a scalar.
  if (auto bcast = llvm::dyn_cast<vector::BroadcastOp>(op)) {
    auto source = bcast.getSource();
    if (llvm::dyn_cast<VectorType>(source.getType()))
      return {};

    return source;
  }

  // 2. linalg.fill - use the scalar input value that used to fill the output
  // tensor.
  if (auto fill = llvm::dyn_cast<linalg::FillOp>(op)) {
    return fill.getInputs()[0];
  }

  // 3. tensor.generateOp - can't guarantee the value is fixed without
  // analysing, bail out.
  if (auto generate = llvm::dyn_cast<tensor::GenerateOp>(op)) {
    return {};
  }

  // 4. vector.transfer_write - inspect the input vector that's written from. If
  // if contains a single value that has been broadcast (e.g. via
  // vector.broadcast), extract it, fail otherwise.
  if (auto xferWrite = llvm::dyn_cast<vector::TransferWriteOp>(op))
    return getStaticPadVal(xferWrite.getVector().getDefiningOp());

  // 5. tensor.insert_slice - inspect the destination tensor. If it's larger
  // than the input tensor, then, provided it's constant, we'll extract the
  // value that was used to generate it (via e.g. linalg.fill), fail otherwise.
  // TODO: Clarify the semantics when the input tensor is larger than the
  // destination.
  if (auto slice = llvm::dyn_cast<tensor::InsertSliceOp>(op))
    return getStaticPadVal(slice.getDest().getDefiningOp());

  return {};
}

static LogicalResult
vectorizeAsInsertSliceOp(RewriterBase &rewriter, tensor::InsertSliceOp sliceOp,
                         ArrayRef<int64_t> inputVectorSizes,
                         SmallVectorImpl<Value> &newResults) {
  // TODO: Introduce a parent class that will handle the insertion point update.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(sliceOp);

  TypedValue<RankedTensorType> source = sliceOp.getSource();
  auto sourceType = source.getType();
  auto resultType = sliceOp.getResultType();

  Value padValue = getStaticPadVal(sliceOp);

  if (!padValue) {
    auto elemType = sourceType.getElementType();
    padValue = arith::ConstantOp::create(rewriter, sliceOp.getLoc(), elemType,
                                         rewriter.getZeroAttr(elemType));
  }

  // 2. Get the vector shape
  SmallVector<int64_t> vecShape;
  size_t rankDiff = resultType.getRank() - sourceType.getRank();
  for (int64_t i = 0, end = sourceType.getRank(); i < end; ++i) {
    if (!inputVectorSizes.empty()) {
      vecShape.push_back(inputVectorSizes[i]);
    } else if (!sourceType.isDynamicDim(i)) {
      vecShape.push_back(sourceType.getDimSize(i));
    } else if (!resultType.isDynamicDim(i)) {
      // Source shape is not statically known, but result shape is.
      // Vectorize with size of result shape. This may be larger than the
      // source size.
      // FIXME: Using rankDiff implies that the source tensor is inserted at
      // the end of the destination tensor. However, that's not required.
      vecShape.push_back(resultType.getDimSize(rankDiff + i));
    } else {
      // Neither source nor result dim of padOp is static. Cannot vectorize
      // the copy.
      return failure();
    }
  }
  auto vecType = VectorType::get(vecShape, sourceType.getElementType());

  // 3. Generate TransferReadOp + TransferWriteOp
  auto loc = sliceOp.getLoc();

  // Create read
  SmallVector<Value> readIndices(
      vecType.getRank(), arith::ConstantIndexOp::create(rewriter, loc, 0));
  Value read = mlir::vector::createReadOrMaskedRead(
      rewriter, loc, source, vecType, padValue,
      /*useInBoundsInsteadOfMasking=*/inputVectorSizes.empty());

  // Create write
  auto writeIndices =
      getValueOrCreateConstantIndexOp(rewriter, loc, sliceOp.getMixedOffsets());
  Operation *write =
      vector::createWriteOrMaskedWrite(rewriter, loc, read, sliceOp.getDest(),
                                       writeIndices, inputVectorSizes.empty());

  // 4. Finalize
  newResults.push_back(write->getResult(0));

  return success();
}

/// Rewrite use of tensor::PadOp result in InsertSliceOp. E.g.:
/// ```
/// %0 = tensor.pad %src ... : tensor<?x?xf32> to tensor<17x5xf32>
/// %r = tensor.insert_slice %0
///     into %dest[%a, %b, 0, 0] [1, 1, 17, 5] [1, 1, 1, 1]
///     : tensor<17x5xf32> into tensor<?x?x17x5xf32>
/// ```
/// is rewritten to:
/// ```
/// %0 = vector.transfer_read %src[%c0, %c0], %padding
///     : tensor<?x?xf32>, vector<17x5xf32>
/// %r = vector.transfer_write %0, %dest[%a, %b, %c0, %c0]
///     {in_bounds = [true, true]} : vector<17x5xf32>, tensor<?x?x17x5xf32>
/// ```
///
/// This rewrite is possible if:
/// - Low padding is static 0.
/// - `padOp` result shape is static.
/// - The entire padded tensor is inserted.
///   (Implies that sizes of `insertOp` are all static.)
/// - Only unit strides in `insertOp`.
/// - Single, scalar padding value.
/// - `padOp` result not used as destination.
struct PadOpVectorizationWithInsertSlicePattern
    : public VectorizePadOpUserPattern<tensor::InsertSliceOp> {
  using VectorizePadOpUserPattern<
      tensor::InsertSliceOp>::VectorizePadOpUserPattern;

  LogicalResult rewriteUser(PatternRewriter &rewriter, tensor::PadOp padOp,
                            tensor::InsertSliceOp insertOp) const override {
    // Low padding must be static 0.
    if (!padOp.hasZeroLowPad())
      return failure();
    // Only unit stride supported.
    if (!insertOp.hasUnitStride())
      return failure();
    // Pad value must be a constant.
    auto padValue = padOp.getConstantPaddingValue();
    if (!padValue)
      return failure();
    // Dynamic shapes not supported.
    if (!cast<ShapedType>(padOp.getResult().getType()).hasStaticShape())
      return failure();
    // Pad result not used as destination.
    if (insertOp.getDest() == padOp.getResult())
      return failure();

    auto vecType = VectorType::get(padOp.getType().getShape(),
                                   padOp.getType().getElementType());
    unsigned vecRank = vecType.getRank();
    unsigned tensorRank = insertOp.getType().getRank();

    // Check if sizes match: Insert the entire tensor into most minor dims.
    // (No permutations allowed.)
    SmallVector<int64_t> expectedSizes(tensorRank - vecRank, 1);
    expectedSizes.append(vecType.getShape().begin(), vecType.getShape().end());
    if (!llvm::all_of(
            llvm::zip(insertOp.getMixedSizes(), expectedSizes), [](auto it) {
              return getConstantIntValue(std::get<0>(it)) == std::get<1>(it);
            }))
      return failure();

    // Insert the TransferReadOp and TransferWriteOp at the position of the
    // InsertSliceOp.
    rewriter.setInsertionPoint(insertOp);

    // Generate TransferReadOp: Read entire source tensor and add high
    // padding.
    SmallVector<Value> readIndices(
        vecRank, arith::ConstantIndexOp::create(rewriter, padOp.getLoc(), 0));
    auto read = vector::TransferReadOp::create(rewriter, padOp.getLoc(),
                                               vecType, padOp.getSource(),
                                               readIndices, padValue);

    // Generate TransferWriteOp: Write to InsertSliceOp's dest tensor at
    // specified offsets. Write is fully in-bounds because a InsertSliceOp's
    // source must fit into the destination at the specified offsets.
    auto writeIndices = getValueOrCreateConstantIndexOp(
        rewriter, padOp.getLoc(), insertOp.getMixedOffsets());
    SmallVector<bool> inBounds(vecRank, true);
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        insertOp, read, insertOp.getDest(), writeIndices,
        ArrayRef<bool>{inBounds});

    return success();
  }
};

void mlir::linalg::populatePadOpVectorizationPatterns(
    RewritePatternSet &patterns, PatternBenefit baseBenefit) {
  patterns.add<PadOpVectorizationWithTransferReadPattern,
               PadOpVectorizationWithTransferWritePattern,
               PadOpVectorizationWithInsertSlicePattern>(
      patterns.getContext(), baseBenefit.getBenefit() + 1);
}

//----------------------------------------------------------------------------//
// Forwarding patterns
//----------------------------------------------------------------------------//

/// Check whether there is any interleaved use of any `values` between
/// `firstOp` and `secondOp`. Conservatively return `true` if any op or value
/// is in a different block.
static bool mayExistInterleavedUses(Operation *firstOp, Operation *secondOp,
                                    ValueRange values) {
  if (firstOp->getBlock() != secondOp->getBlock() ||
      !firstOp->isBeforeInBlock(secondOp)) {
    LDBG() << "interleavedUses precondition failed, firstOp: " << *firstOp
           << ", second op: " << *secondOp;
    return true;
  }
  for (auto v : values) {
    for (auto &u : v.getUses()) {
      Operation *owner = u.getOwner();
      if (owner == firstOp || owner == secondOp)
        continue;
      // TODO: this is too conservative, use dominance info in the future.
      if (owner->getBlock() == firstOp->getBlock() &&
          (owner->isBeforeInBlock(firstOp) || secondOp->isBeforeInBlock(owner)))
        continue;
      LDBG() << " found interleaved op " << *owner << ", firstOp: " << *firstOp
             << ", second op: " << *secondOp;
      return true;
    }
  }
  return false;
}

/// Return the unique subview use of `v` if it is indeed unique, null
/// otherwise.
static memref::SubViewOp getSubViewUseIfUnique(Value v) {
  memref::SubViewOp subViewOp;
  for (auto &u : v.getUses()) {
    if (auto newSubViewOp = dyn_cast<memref::SubViewOp>(u.getOwner())) {
      if (subViewOp)
        return memref::SubViewOp();
      subViewOp = newSubViewOp;
    }
  }
  return subViewOp;
}

/// TODO: use interfaces, side-effects and aliasing analysis as appropriate,
/// when available.
LogicalResult LinalgCopyVTRForwardingPattern::matchAndRewrite(
    vector::TransferReadOp xferOp, PatternRewriter &rewriter) const {

  // TODO: support mask.
  if (xferOp.getMask())
    return rewriter.notifyMatchFailure(xferOp, "unsupported mask");

  // Transfer into `view`.
  Value viewOrAlloc = xferOp.getBase();
  if (!viewOrAlloc.getDefiningOp<memref::ViewOp>() &&
      !viewOrAlloc.getDefiningOp<memref::AllocOp>())
    return rewriter.notifyMatchFailure(xferOp, "source not a view or alloc");

  // Ensure there is exactly one subview of `viewOrAlloc` defining `subView`.
  memref::SubViewOp subViewOp = getSubViewUseIfUnique(viewOrAlloc);
  if (!subViewOp)
    return rewriter.notifyMatchFailure(xferOp, "no subview found");
  Value subView = subViewOp.getResult();

  // Find the copy into `subView` without interleaved uses.
  memref::CopyOp copyOp;
  for (auto &u : subView.getUses()) {
    if (auto newCopyOp = dyn_cast<memref::CopyOp>(u.getOwner())) {
      assert(isa<MemRefType>(newCopyOp.getTarget().getType()));
      if (newCopyOp.getTarget() != subView)
        continue;
      if (mayExistInterleavedUses(newCopyOp, xferOp, {viewOrAlloc, subView}))
        continue;
      copyOp = newCopyOp;
      break;
    }
  }
  if (!copyOp)
    return rewriter.notifyMatchFailure(xferOp, "no copy found");

  // Find the fill into `viewOrAlloc` without interleaved uses before the
  // copy.
  FillOp maybeFillOp;
  for (auto &u : viewOrAlloc.getUses()) {
    if (auto newFillOp = dyn_cast<FillOp>(u.getOwner())) {
      assert(isa<MemRefType>(newFillOp.output().getType()));
      if (newFillOp.output() != viewOrAlloc)
        continue;
      if (mayExistInterleavedUses(newFillOp, copyOp, {viewOrAlloc, subView}))
        continue;
      maybeFillOp = newFillOp;
      break;
    }
  }
  // Ensure padding matches.
  if (maybeFillOp && xferOp.getPadding() != maybeFillOp.value())
    return rewriter.notifyMatchFailure(xferOp,
                                       "padding value does not match fill");

  // `in` is the subview that memref.copy reads. Replace it.
  Value in = copyOp.getSource();

  // memref.copy + linalg.fill can be used to create a padded local buffer.
  // The `masked` attribute is only valid on this padded buffer.
  // When forwarding to vector.transfer_read, the attribute must be reset
  // conservatively.
  auto vectorType = xferOp.getVectorType();
  Value res = vector::TransferReadOp::create(
      rewriter, xferOp.getLoc(), vectorType, in, xferOp.getIndices(),
      xferOp.getPermutationMapAttr(), xferOp.getPadding(), xferOp.getMask(),
      rewriter.getBoolArrayAttr(
          SmallVector<bool>(vectorType.getRank(), false)));

  if (maybeFillOp)
    rewriter.eraseOp(maybeFillOp);
  rewriter.eraseOp(copyOp);
  rewriter.replaceOp(xferOp, res);

  return success();
}

/// TODO: use interfaces, side-effects and aliasing analysis as appropriate,
/// when available.
LogicalResult LinalgCopyVTWForwardingPattern::matchAndRewrite(
    vector::TransferWriteOp xferOp, PatternRewriter &rewriter) const {
  // TODO: support mask.
  if (xferOp.getMask())
    return rewriter.notifyMatchFailure(xferOp, "unsupported mask");

  // Transfer into `viewOrAlloc`.
  Value viewOrAlloc = xferOp.getBase();
  if (!viewOrAlloc.getDefiningOp<memref::ViewOp>() &&
      !viewOrAlloc.getDefiningOp<memref::AllocOp>())
    return rewriter.notifyMatchFailure(xferOp, "source not a view or alloc");

  // Ensure there is exactly one subview of `viewOrAlloc` defining `subView`.
  memref::SubViewOp subViewOp = getSubViewUseIfUnique(viewOrAlloc);
  if (!subViewOp)
    return rewriter.notifyMatchFailure(xferOp, "no subview found");
  Value subView = subViewOp.getResult();

  // Find the copy from `subView` without interleaved uses.
  memref::CopyOp copyOp;
  for (auto &u : subViewOp.getResult().getUses()) {
    if (auto newCopyOp = dyn_cast<memref::CopyOp>(u.getOwner())) {
      if (newCopyOp.getSource() != subView)
        continue;
      if (mayExistInterleavedUses(xferOp, newCopyOp, {viewOrAlloc, subView}))
        continue;
      copyOp = newCopyOp;
      break;
    }
  }
  if (!copyOp)
    return rewriter.notifyMatchFailure(xferOp, "no copy found");

  // `out` is the subview copied into that we replace.
  assert(isa<MemRefType>(copyOp.getTarget().getType()));
  Value out = copyOp.getTarget();

  // Forward vector.transfer into copy.
  // memref.copy + linalg.fill can be used to create a padded local buffer.
  // The `masked` attribute is only valid on this padded buffer.
  // When forwarding to vector.transfer_write, the attribute must be reset
  // conservatively.
  auto vector = xferOp.getVector();
  vector::TransferWriteOp::create(
      rewriter, xferOp.getLoc(), vector, out, xferOp.getIndices(),
      xferOp.getPermutationMapAttr(), xferOp.getMask(),
      rewriter.getBoolArrayAttr(SmallVector<bool>(
          dyn_cast<VectorType>(vector.getType()).getRank(), false)));

  rewriter.eraseOp(copyOp);
  rewriter.eraseOp(xferOp);

  return success();
}

//===----------------------------------------------------------------------===//
// Convolution vectorization patterns
//===----------------------------------------------------------------------===//

template <int N>
static void bindShapeDims(ShapedType shapedType) {}

template <int N, typename IntTy, typename... IntTy2>
static void bindShapeDims(ShapedType shapedType, IntTy &val, IntTy2 &...vals) {
  val = shapedType.getShape()[N];
  bindShapeDims<N + 1, IntTy2 &...>(shapedType, vals...);
}

/// Bind a pack of int& to the leading dimensions of shapedType.getShape().
template <typename... IntTy>
static void bindShapeDims(ShapedType shapedType, IntTy &...vals) {
  bindShapeDims<0>(shapedType, vals...);
}

/// Compute permutations to transpose each operand from its current layout to
/// the canonical NWC form:
///   lhs = [N?, iw, inputChannel?, depth?, C_like_batch?]
///   rhs = [kw, inputChannel?, outputChannel?, depth?]
///   res = [N?, w,  outputChannel?, depth?, C_like_batch?]
/// An empty result vector means the permutation is identity.
static Conv1DPerms computeCanonicalPerms(LinalgOp linalgOp,
                                         const Conv1DConfig &config) {
  const ConvolutionDimensions &dims = config.dims;
  ArrayRef<unsigned> nLikeBatch = config.nLikeBatch;
  ArrayRef<unsigned> cLikeBatch = config.cLikeBatch;
  AffineMap lhsMap = linalgOp.getIndexingMapsArray()[0];
  AffineMap rhsMap = linalgOp.getIndexingMapsArray()[1];
  AffineMap resMap = linalgOp.getIndexingMapsArray()[2];

  DenseMap<unsigned, unsigned> loopToLhs = buildLoopToTensorDimMap(lhsMap);
  DenseMap<unsigned, unsigned> loopToRhs = buildLoopToTensorDimMap(rhsMap);
  DenseMap<unsigned, unsigned> loopToRes = buildLoopToTensorDimMap(resMap);

  // The spatial (outputImage) tensor dim in LHS is the result that is NOT a
  // plain AffineDimExpr: its expression is `strideW*ow + dilationW*kw`.
  std::optional<unsigned> lhsSpatialTensorDim;
  for (unsigned i = 0; i < lhsMap.getNumResults(); ++i) {
    if (!isa<AffineDimExpr>(lhsMap.getResult(i))) {
      lhsSpatialTensorDim = i;
      break;
    }
  }
  assert((dims.outputImage.empty() || lhsSpatialTensorDim) &&
         "expected a non-plain LHS dim expr for the spatial dim");

  Conv1DPerms perms;

  // Result: [N_like_batch, outputImage, outputChannel, depth, C_like_batch].
  for (unsigned d : nLikeBatch)
    perms.res.push_back(loopToRes[d]);
  for (unsigned d : dims.outputImage)
    perms.res.push_back(loopToRes[d]);
  for (unsigned d : dims.outputChannel)
    perms.res.push_back(loopToRes[d]);
  for (unsigned d : dims.depth)
    perms.res.push_back(loopToRes[d]);
  for (unsigned d : cLikeBatch)
    perms.res.push_back(loopToRes[d]);

  // LHS: [N_like_batch, outputImage->spatial, inputChannel, depth,
  //       C_like_batch]. The spatial position is the non-plain dim expr.
  for (unsigned d : nLikeBatch)
    perms.lhs.push_back(loopToLhs[d]);
  if (lhsSpatialTensorDim)
    perms.lhs.push_back(*lhsSpatialTensorDim);
  for (unsigned d : dims.inputChannel)
    perms.lhs.push_back(loopToLhs[d]);
  for (unsigned d : dims.depth)
    perms.lhs.push_back(loopToLhs[d]);
  for (unsigned d : cLikeBatch)
    perms.lhs.push_back(loopToLhs[d]);

  // RHS: [filterLoop, inputChannel, outputChannel, depth]. For pooling RHS
  // only has filterLoop.
  for (unsigned d : dims.filterLoop)
    perms.rhs.push_back(loopToRhs[d]);
  for (unsigned d : dims.inputChannel)
    perms.rhs.push_back(loopToRhs[d]);
  for (unsigned d : dims.outputChannel)
    perms.rhs.push_back(loopToRhs[d]);
  for (unsigned d : dims.depth)
    perms.rhs.push_back(loopToRhs[d]);

  if (isIdentityPermutation(perms.lhs))
    perms.lhs.clear();
  if (isIdentityPermutation(perms.rhs))
    perms.rhs.clear();
  if (isIdentityPermutation(perms.res))
    perms.res.clear();

  return perms;
}

namespace {
/// Generate a vector implementation for either:
/// ```
///   Op def: (     w,     kw  )
///    Iters: ({Par(), Red()})
///   Layout: {{w + kw}, {kw}, {w}}
/// ```
/// kw is unrolled.
///
/// or
///
/// ```
///   Op def: (     n,     w,     c,    kw,    f  )
///    Iters: ({Par(), Par(), Par(), Red(), Red()})
///   Layout: {{n, strideW * w + dilationW * kw, c}, {kw, c, f}, {n, w, f}}
/// ```
/// kw is unrolled, w is unrolled iff dilationW > 1.
///
/// or
///
/// ```
///   Op def: (     n,     c,     w,    f,    kw )
///    Iters: ({Par(), Par(), Par(), Red(), Red()})
///   Layout: {{n, c, strideW * w + dilationW * kw}, {f, c, kw}, {n, f, w}}
/// ```
/// kw is unrolled, w is unrolled iff dilationW > 1.
///
/// or
///
/// ```
///   Op def: (     n,     w,     c,    kw )
///    Iters: ({Par(), Par(), Par(), Red()})
///   Layout: {{n, strideW * w + dilationW * kw, c}, {kw, c}, {n, w, c}}
/// ```
/// kw is unrolled, w is unrolled iff dilationW > 1.
struct Conv1DGenerator
    : public StructuredGenerator<LinalgOp, utils::IteratorType> {
  /// Factory method to create a Conv1DGenerator. Uses the shared
  /// `classifyAs1DConv` so the precondition and the matcher cannot drift
  /// apart. Returns failure if the op is not a supported 1D conv/pool op.
  static FailureOr<Conv1DGenerator> create(RewriterBase &rewriter,
                                           LinalgOp linalgOp) {
    std::string reason;
    FailureOr<Conv1DConfig> maybeConfig = classifyAs1DConv(linalgOp, &reason);
    if (failed(maybeConfig))
      return rewriter.notifyMatchFailure(linalgOp, reason);
    return Conv1DGenerator(rewriter, linalgOp, std::move(*maybeConfig));
  }

private:
  Conv1DGenerator(RewriterBase &rewriter, LinalgOp linalgOp,
                  Conv1DConfig config)
      : StructuredGenerator<LinalgOp, utils::IteratorType>(rewriter, linalgOp),
        config(std::move(config)) {
    strideW = static_cast<int>(this->config.dims.strides.front());
    dilationW = static_cast<int>(this->config.dims.dilations.front());

    lhsShaped = linalgOp.getDpsInputOperand(0)->get();
    rhsShaped = linalgOp.getDpsInputOperand(1)->get();
    resShaped = linalgOp.getDpsInitOperand(0)->get();
    lhsShapedType = dyn_cast<ShapedType>(lhsShaped.getType());
    rhsShapedType = dyn_cast<ShapedType>(rhsShaped.getType());
    resShapedType = dyn_cast<ShapedType>(resShaped.getType());

    Operation *reduceOp = matchLinalgReduction(linalgOp.getDpsInitOperand(0));
    redOp = reduceOp->getName().getIdentifier();
    setPoolExtFromReduce(reduceOp);
    reductionKind = getCombinerOpKind(reduceOp).value();
  }

public:
  // Take a value and widen to have the same element type as `ty`.
  Value promote(RewriterBase &rewriter, Location loc, Value val, Type ty) {
    const Type srcElementType = getElementTypeOrSelf(val.getType());
    const Type dstElementType = getElementTypeOrSelf(ty);
    assert(isa<IntegerType>(dstElementType) || isa<FloatType>(dstElementType));
    if (srcElementType == dstElementType)
      return val;

    const int64_t srcWidth = srcElementType.getIntOrFloatBitWidth();
    const int64_t dstWidth = dstElementType.getIntOrFloatBitWidth();
    // Handle both shaped as well as scalar types.
    Type dstType;
    if (auto shapedType = dyn_cast<ShapedType>(val.getType()))
      dstType = shapedType.cloneWith(std::nullopt, dstElementType);
    else
      dstType = dstElementType;

    if (isa<IntegerType>(srcElementType) && isa<FloatType>(dstElementType)) {
      return arith::SIToFPOp::create(rewriter, loc, dstType, val);
    }

    if (isa<FloatType>(srcElementType) && isa<FloatType>(dstElementType) &&
        srcWidth < dstWidth)
      return arith::ExtFOp::create(rewriter, loc, dstType, val);

    if (isa<IntegerType>(srcElementType) && isa<IntegerType>(dstElementType) &&
        srcWidth < dstWidth)
      return arith::ExtSIOp::create(rewriter, loc, dstType, val);

    assert(false && "unhandled promotion case");
    return nullptr;
  }

  // Create a contraction: lhs{n, w, c} * rhs{c, f} -> res{n, w, f}
  Value conv1dSliceAsContraction(RewriterBase &rewriter, Location loc,
                                 Value lhs, Value rhs, Value res) {
    vector::IteratorType par = vector::IteratorType::parallel;
    vector::IteratorType red = vector::IteratorType::reduction;
    AffineExpr n, w, f, c;
    bindDims(ctx, n, w, f, c);
    lhs = promote(rewriter, loc, lhs, res.getType());
    rhs = promote(rewriter, loc, rhs, res.getType());
    auto contrationOp = vector::ContractionOp::create(
        rewriter, loc, lhs, rhs, res,
        /*indexingMaps=*/MapList{{n, w, c}, {c, f}, {n, w, f}},
        /*iteratorTypes=*/ArrayRef<vector::IteratorType>{par, par, par, red});
    contrationOp.setKind(reductionKind);
    return contrationOp;
  }

  // Create an outerproduct: lhs{w} * rhs{1} -> res{w} for single channel
  // convolution.
  Value conv1dSliceAsOuterProduct(RewriterBase &rewriter, Location loc,
                                  Value lhs, Value rhs, Value res) {
    lhs = promote(rewriter, loc, lhs, res.getType());
    rhs = promote(rewriter, loc, rhs, res.getType());
    return vector::OuterProductOp::create(rewriter, loc, res.getType(), lhs,
                                          rhs, res, vector::CombiningKind::ADD);
  }

  // Create a reduction: lhs{n, w, c} -> res{n, w, c}
  Value pool1dSlice(RewriterBase &rewriter, Location loc, Value lhs,
                    Value res) {
    if (isPoolExt)
      lhs = rewriter.create(loc, poolExtOp, lhs, res.getType())->getResult(0);
    return rewriter
        .create(loc, redOp, ArrayRef<Value>{lhs, res}, res.getType())
        ->getResult(0);
  }

  /// Generate a vector implementation for:
  /// ```
  ///   Op def: (     n,     w,     c,    kw)
  ///    Iters: ({Par(), Par(), Par(), Red()})
  ///   Layout: {{n, strideW * w + dilationW * kw, c}, {kw, c}, {n, w, c}}
  /// ```
  /// kw is always unrolled.
  /// TODO: w (resp. kw) is unrolled when the strideW ( resp. dilationW) is
  /// > 1.
  FailureOr<Operation *> depthwiseConv(uint64_t channelDimVecSize,
                                       bool channelDimScalableFlag,
                                       bool flatten) {
    bool scalableChDim = false;
    bool useMasking = false;
    int64_t nSize, wSize, cSize, kwSize;
    // kernel{kw, c}
    bindShapeDims(rhsShapedType, kwSize, cSize);
    if (ShapedType::isDynamic(cSize)) {
      assert(channelDimVecSize != 0 && "Channel dim vec size must be > 0");
      cSize = channelDimVecSize;
      // Scalable vectors are only used when both conditions are met:
      //  1. channel dim is dynamic
      //  2. channelDimScalableFlag is set
      scalableChDim = channelDimScalableFlag;
      useMasking = true;
    }

    assert(!(useMasking && flatten) &&
           "Unsupported flattened conv with dynamic shapes");

    // out{n, w, c}
    bindShapeDims(resShapedType, nSize, wSize);

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

    // w is unrolled (i.e. wSizeStep == 1) iff strideW > 1.
    // When strideW == 1, we can batch the contiguous loads and avoid
    // unrolling
    int64_t wSizeStep = strideW == 1 ? wSize : 1;

    Type lhsEltType = lhsShapedType.getElementType();
    Type rhsEltType = rhsShapedType.getElementType();
    Type resEltType = resShapedType.getElementType();
    VectorType lhsType = VectorType::get(
        {nSize,
         // iw = ow * sw + kw *  dw - 1
         //   (i.e. 16 convolved with 3 (@stride 1 dilation 1) -> 14)
         ((wSize - 1) * strideW + 1) + ((kwSize - 1) * dilationW + 1) - 1,
         cSize},
        lhsEltType, /*scalableDims=*/{false, false, scalableChDim});
    VectorType rhsType =
        VectorType::get({kwSize, cSize}, rhsEltType,
                        /*scalableDims=*/{false, scalableChDim});
    VectorType resType =
        VectorType::get({nSize, wSize, cSize}, resEltType,
                        /*scalableDims=*/{false, false, scalableChDim});

    // Masks the input xfer Op along the channel dim, iff the corresponding
    // scalable flag is set.
    auto maybeMaskXferOp = [&](ArrayRef<int64_t> maskShape,
                               ArrayRef<bool> scalableDims,
                               Operation *opToMask) {
      if (!useMasking)
        return opToMask;
      auto maskType =
          VectorType::get(maskShape, rewriter.getI1Type(), scalableDims);

      SmallVector<bool> inBounds(maskShape.size(), true);
      auto xferOp = cast<VectorTransferOpInterface>(opToMask);
      xferOp->setAttr(xferOp.getInBoundsAttrName(),
                      rewriter.getBoolArrayAttr(inBounds));

      SmallVector<OpFoldResult> mixedDims = vector::getMixedSizesXfer(
          cast<LinalgOp>(op).hasPureTensorSemantics(), opToMask, rewriter);

      Value maskOp =
          vector::CreateMaskOp::create(rewriter, loc, maskType, mixedDims);

      return mlir::vector::maskOperation(rewriter, opToMask, maskOp);
    };

    // Read lhs slice of size {n, w * strideW + kw * dilationW, c} @ [0, 0,
    // 0].
    Value lhs = vector::TransferReadOp::create(
        rewriter, loc, lhsType, lhsShaped, ValueRange{zero, zero, zero},
        /*padding=*/arith::getZeroConstant(rewriter, loc, lhsEltType));
    auto *maybeMaskedLhs = maybeMaskXferOp(
        lhsType.getShape(), lhsType.getScalableDims(), lhs.getDefiningOp());

    // Read rhs slice of size {kw, c} @ [0, 0].
    Value rhs = vector::TransferReadOp::create(
        rewriter, loc, rhsType, rhsShaped, ValueRange{zero, zero},
        /*padding=*/arith::getZeroConstant(rewriter, loc, rhsEltType));
    auto *maybeMaskedRhs = maybeMaskXferOp(
        rhsType.getShape(), rhsType.getScalableDims(), rhs.getDefiningOp());

    // Read res slice of size {n, w, c} @ [0, 0, 0].
    Value res = vector::TransferReadOp::create(
        rewriter, loc, resType, resShaped, ValueRange{zero, zero, zero},
        /*padding=*/arith::getZeroConstant(rewriter, loc, resEltType));
    auto *maybeMaskedRes = maybeMaskXferOp(
        resType.getShape(), resType.getScalableDims(), res.getDefiningOp());

    FailureOr<Value> kernelRes = depthwise1DKernel(
        maybeMaskedLhs->getResult(0), maybeMaskedRhs->getResult(0),
        maybeMaskedRes->getResult(0), nSize, wSize, cSize, kwSize, wSizeStep,
        flatten);
    if (failed(kernelRes)) {
      // Best-effort: erase the original transfer_reads we created above.
      // (Mask wrapper ops, if any, will be handled by the failed-pattern
      // rollback in the parent rewrite driver.)
      for (Value v : {res, rhs, lhs, zero})
        if (v)
          rewriter.eraseOp(v.getDefiningOp());
      return failure();
    }

    Operation *resOut = vector::TransferWriteOp::create(
        rewriter, loc, *kernelRes, resShaped, ValueRange{zero, zero, zero});
    return maybeMaskXferOp(resType.getShape(), resType.getScalableDims(),
                           resOut);
  }

  /// Vector-level depthwise compute kernel: extracts kw x w tiles from the
  /// canonical NWC operand vectors, runs MulAcc, and re-inserts the tiles
  /// back into the result vector. All inputs must be in canonical NWC form
  /// (lhs=[n,iw,c], rhs=[kw,c], res=[n,w,c]).
  ///
  /// On MulAcc failure the kernel erases only the IR it created; the caller
  /// is responsible for cleaning up its own pre-kernel IR.
  FailureOr<Value> depthwise1DKernel(Value lhs, Value rhs, Value res,
                                     int64_t nSize, int64_t wSize,
                                     int64_t cSize, int64_t kwSize,
                                     int64_t wSizeStep, bool flatten) {
    Type lhsEltType = cast<VectorType>(lhs.getType()).getElementType();
    Type resEltType = cast<VectorType>(res.getType()).getElementType();

    SmallVector<int64_t> inOutSliceSizes = {nSize, wSizeStep, cSize};
    SmallVector<int64_t> inOutStrides = {1, 1, 1};

    SmallVector<Value> lhsVals, rhsVals, resVals;
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      for (int64_t w = 0; w < wSize; w += wSizeStep) {
        lhsVals.push_back(vector::ExtractStridedSliceOp::create(
            rewriter, loc, lhs,
            /*offsets=*/ArrayRef<int64_t>{0, w * strideW + kw * dilationW, 0},
            inOutSliceSizes, inOutStrides));
      }
    }
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      rhsVals.push_back(vector::ExtractOp::create(
          rewriter, loc, rhs, /*offsets=*/ArrayRef<int64_t>{kw}));
    }
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      resVals.push_back(vector::ExtractStridedSliceOp::create(
          rewriter, loc, res,
          /*offsets=*/ArrayRef<int64_t>{0, w, 0}, inOutSliceSizes,
          inOutStrides));
    }

    auto linearIndex = [&](int64_t kw, int64_t w) {
      return kw * (wSize / wSizeStep) + w;
    };

    // Note: the scalable flags are ignored because flattening combined with
    // scalable vectorization is not supported.
    SmallVector<int64_t> inOutFlattenSliceSizes = {nSize, wSizeStep * cSize};
    auto lhsTypeAfterFlattening =
        VectorType::get(inOutFlattenSliceSizes, lhsEltType);
    auto resTypeAfterFlattening =
        VectorType::get(inOutFlattenSliceSizes, resEltType);

    for (int64_t kw = 0; kw < kwSize; ++kw) {
      for (int64_t w = 0; w < wSize; w += wSizeStep) {
        Value lhsVal = lhsVals[linearIndex(kw, w)];
        Value resVal = resVals[w];
        if (flatten) {
          lhsVal = vector::ShapeCastOp::create(rewriter, loc,
                                               lhsTypeAfterFlattening, lhsVal);
          resVal = vector::ShapeCastOp::create(rewriter, loc,
                                               resTypeAfterFlattening, resVal);
        }
        resVals[w] = depthwiseConv1dSliceAsMulAcc(rewriter, loc, lhsVal,
                                                  rhsVals[kw], resVal, flatten);
        if (flatten) {
          resVals[w] = vector::ShapeCastOp::create(
              rewriter, loc, VectorType::get(inOutSliceSizes, resEltType),
              resVals[w]);
        }
      }
    }

    if (!llvm::all_of(resVals, [](Value v) { return v; })) {
      for (auto &collection : {resVals, rhsVals, lhsVals})
        for (Value v : collection)
          if (v)
            rewriter.eraseOp(v.getDefiningOp());
      return rewriter.notifyMatchFailure(op, "failed to create FMA");
    }

    Value out = res;
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      out = vector::InsertStridedSliceOp::create(
          rewriter, loc, resVals[w], out,
          /*offsets=*/ArrayRef<int64_t>{0, w, 0},
          /*strides=*/ArrayRef<int64_t>{1, 1, 1});
    }
    return out;
  }

  /// Depthwise convolution for non-canonical layouts: read the operands in
  /// their original layout, transpose the loaded vectors to canonical NWC,
  /// run the kernel, inverse-transpose the result, and write it back.
  /// Static shapes only (no masking).
  FailureOr<Operation *> depthwiseConvViaTranspose(const Conv1DPerms &perms,
                                                   bool flatten) {
    // Canonical sizes: kernel is [kw, c], result is [n, w, c].
    int64_t kwSize = 0, cSize = 0, nSize = 0, wSize = 0;
    {
      DenseMap<unsigned, unsigned> loopToRhs =
          buildLoopToTensorDimMap(cast<LinalgOp>(op).getIndexingMapsArray()[1]);
      DenseMap<unsigned, unsigned> loopToRes =
          buildLoopToTensorDimMap(cast<LinalgOp>(op).getIndexingMapsArray()[2]);
      kwSize = rhsShapedType.getShape()[loopToRhs[dims().filterLoop[0]]];
      cSize = rhsShapedType.getShape()[loopToRhs[dims().depth[0]]];
      nSize = config.nLikeBatch.empty()
                  ? 1
                  : resShapedType.getShape()[loopToRes[config.nLikeBatch[0]]];
      wSize = resShapedType.getShape()[loopToRes[dims().outputImage[0]]];
    }
    int64_t iwSize =
        ((wSize - 1) * strideW + 1) + ((kwSize - 1) * dilationW + 1) - 1;

    Conv1DShapes canon{
        /*lhs=*/{nSize, iwSize, cSize},
        /*rhs=*/{kwSize, cSize},
        /*res=*/{nSize, wSize, cSize},
    };
    Conv1DShapes read = computeReadShapes(canon, perms);

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Type lhsEltType = lhsShapedType.getElementType();
    Type rhsEltType = rhsShapedType.getElementType();
    Type resEltType = resShapedType.getElementType();
    SmallVector<Value> lhsIdx(read.lhs.size(), zero);
    SmallVector<Value> rhsIdx(read.rhs.size(), zero);
    SmallVector<Value> resIdx(read.res.size(), zero);

    Value lhs = vector::TransferReadOp::create(
        rewriter, loc, VectorType::get(read.lhs, lhsEltType), lhsShaped, lhsIdx,
        arith::getZeroConstant(rewriter, loc, lhsEltType));
    Value rhs = vector::TransferReadOp::create(
        rewriter, loc, VectorType::get(read.rhs, rhsEltType), rhsShaped, rhsIdx,
        arith::getZeroConstant(rewriter, loc, rhsEltType));
    Value res = vector::TransferReadOp::create(
        rewriter, loc, VectorType::get(read.res, resEltType), resShaped, resIdx,
        arith::getZeroConstant(rewriter, loc, resEltType));

    if (!perms.lhs.empty())
      lhs = vector::TransposeOp::create(rewriter, loc, lhs, perms.lhs);
    if (!perms.rhs.empty())
      rhs = vector::TransposeOp::create(rewriter, loc, rhs, perms.rhs);
    if (!perms.res.empty())
      res = vector::TransposeOp::create(rewriter, loc, res, perms.res);

    int64_t wSizeStep = strideW == 1 ? wSize : 1;
    FailureOr<Value> kernelRes = depthwise1DKernel(
        lhs, rhs, res, nSize, wSize, cSize, kwSize, wSizeStep, flatten);
    if (failed(kernelRes))
      return failure();

    return writeAndPostTranspose(*kernelRes, read.res, perms.res);
  }

  /// Lower:
  ///   *  lhs{n, w, c} * rhs{c} -> res{n, w, c} (flatten = false)
  ///   *  lhs{n, w * c} * rhs{c} -> res{n, w * c} (flatten = true)
  /// to MulAcc.
  Value depthwiseConv1dSliceAsMulAcc(RewriterBase &rewriter, Location loc,
                                     Value lhs, Value rhs, Value res,
                                     bool flatten) {
    auto rhsTy = cast<ShapedType>(rhs.getType());
    auto resTy = cast<ShapedType>(res.getType());

    // TODO(suderman): Change this to use a vector.ima intrinsic.
    lhs = promote(rewriter, loc, lhs, resTy);

    if (flatten) {
      // NOTE: This following logic won't work for scalable vectors. For this
      // reason, "flattening" is not supported when shapes are dynamic (this
      // should be captured by one of the pre-conditions).

      // There are two options for handling the filter:
      //  * shape_cast(broadcast(filter))
      //  * broadcast(shuffle(filter))
      // Opt for the option without shape_cast to simplify the codegen.
      auto rhsSize = cast<VectorType>(rhs.getType()).getShape()[0];
      auto resSize = cast<VectorType>(res.getType()).getShape()[1];

      SmallVector<int64_t, 16> indices;
      for (int i = 0; i < resSize / rhsSize; ++i) {
        for (int j = 0; j < rhsSize; ++j)
          indices.push_back(j);
      }

      rhs = vector::ShuffleOp::create(rewriter, loc, rhs, rhs, indices);
    }
    // Broadcast the filter to match the output vector
    rhs = vector::BroadcastOp::create(rewriter, loc,
                                      resTy.clone(rhsTy.getElementType()), rhs);

    rhs = promote(rewriter, loc, rhs, resTy);

    if (!lhs || !rhs)
      return nullptr;

    if (isa<FloatType>(resTy.getElementType()))
      return vector::FMAOp::create(rewriter, loc, lhs, rhs, res);

    auto mul = arith::MulIOp::create(rewriter, loc, lhs, rhs);
    return arith::AddIOp::create(rewriter, loc, mul, res);
  }

  /// Create a contraction: lhs{w, c} * rhs{c, f} -> res{w, f}.
  /// Used for batchless channeled convolution.
  Value conv1dBatchlessSliceAsContraction(RewriterBase &rewriter, Location loc,
                                          Value lhs, Value rhs, Value res) {
    vector::IteratorType par = vector::IteratorType::parallel;
    vector::IteratorType red = vector::IteratorType::reduction;
    AffineExpr w, f, c;
    bindDims(ctx, w, f, c);
    lhs = promote(rewriter, loc, lhs, res.getType());
    rhs = promote(rewriter, loc, rhs, res.getType());
    auto contractionOp = vector::ContractionOp::create(
        rewriter, loc, lhs, rhs, res,
        /*indexingMaps=*/MapList{{w, c}, {c, f}, {w, f}},
        /*iteratorTypes=*/ArrayRef<vector::IteratorType>{par, par, red});
    contractionOp.setKind(reductionKind);
    return contractionOp;
  }

  //===--------------------------------------------------------------------===//
  // generate() pipeline helpers
  //===--------------------------------------------------------------------===//

  /// Handle depthwise convolution. For canonical (NWC) layout, delegate to
  /// the existing depthwiseConv() which supports masking/scalable channel
  /// dim. For non-canonical layouts (e.g. NcwCw), pre-transpose at the
  /// vector boundaries and run the canonical kernel; this path requires
  /// fully-static shapes (no masking).
  FailureOr<Operation *> handleDepthwise(ArrayRef<int64_t> inputVecSizes,
                                         ArrayRef<bool> inputScalableVecDims,
                                         bool flatten1DDepthwiseConv) {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    Conv1DPerms perms = computeCanonicalPerms(linalgOp, config);

    if (perms.allIdentity()) {
      // Canonical NWC: existing path handles masking + scalable dims.
      int64_t vecChDimSize = 0;
      bool vecChScalableFlag = false;
      if (!inputVecSizes.empty()) {
        DenseMap<unsigned, unsigned> loopToRes =
            buildLoopToTensorDimMap(linalgOp.getIndexingMapsArray()[2]);
        unsigned chTensorDim = loopToRes[dims().depth[0]];
        vecChDimSize = inputVecSizes[chTensorDim];
        vecChScalableFlag = inputScalableVecDims[chTensorDim];
      }
      return depthwiseConv(vecChDimSize, vecChScalableFlag,
                           flatten1DDepthwiseConv);
    }

    // Non-canonical depthwise: only static shapes are supported because
    // boundary masking interacts with the post-transpose layout in ways we
    // don't yet handle.
    if (!lhsShapedType.hasStaticShape() || !rhsShapedType.hasStaticShape() ||
        !resShapedType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op,
          "depthwise conv with non-canonical layout requires static shapes");
    return depthwiseConvViaTranspose(perms, flatten1DDepthwiseConv);
  }

  /// Extract canonical loop sizes from ConvolutionDimensions.
  Conv1DSizes computeConvSizes(LinalgOp linalgOp) const {
    SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
    Conv1DSizes sizes;
    sizes.kw = loopRanges[dims().filterLoop[0]];
    sizes.w = loopRanges[dims().outputImage[0]];
    if (!config.nLikeBatch.empty())
      sizes.n = loopRanges[config.nLikeBatch[0]];
    if (config.isPooling) {
      // Pooling has no separate filter channel: the C-like batch dim doubles
      // as both input and output channel size.
      if (!config.cLikeBatch.empty())
        sizes.c = sizes.f = loopRanges[config.cLikeBatch[0]];
    } else if (config.layout != ConvLayoutKind::Scalar) {
      sizes.c = loopRanges[dims().inputChannel[0]];
      sizes.f = loopRanges[dims().outputChannel[0]];
    }
    return sizes;
  }

  /// Build the canonical (post-transpose) NWC operand shapes.
  Conv1DShapes computeCanonicalShapes(const Conv1DSizes &sizes) const {
    int64_t iwSize = config.layout == ConvLayoutKind::Scalar
                         ? (sizes.w + sizes.kw - 1)
                         : ((sizes.w - 1) * strideW + 1) +
                               ((sizes.kw - 1) * dilationW + 1) - 1;
    Conv1DShapes shapes;
    switch (config.layout) {
    case ConvLayoutKind::Scalar:
      shapes.lhs = {iwSize};
      shapes.rhs = {sizes.kw};
      shapes.res = {sizes.w};
      break;
    case ConvLayoutKind::Batched:
      shapes.lhs = {sizes.n, iwSize, sizes.c};
      shapes.rhs = config.isPooling
                       ? SmallVector<int64_t>{sizes.kw}
                       : SmallVector<int64_t>{sizes.kw, sizes.c, sizes.f};
      shapes.res = {sizes.n, sizes.w, sizes.f};
      break;
    case ConvLayoutKind::Batchless:
      shapes.lhs = {iwSize, sizes.c};
      shapes.rhs = config.isPooling
                       ? SmallVector<int64_t>{sizes.kw}
                       : SmallVector<int64_t>{sizes.kw, sizes.c, sizes.f};
      shapes.res = {sizes.w, sizes.f};
      break;
    }
    return shapes;
  }

  /// Compute the pre-read (transfer_read) shapes by inverting the
  /// permutations on the canonical shapes. If a permutation is identity, the
  /// canonical shape is returned unchanged.
  Conv1DShapes computeReadShapes(const Conv1DShapes &canonical,
                                 const Conv1DPerms &perms) const {
    auto inverseApply = [](ArrayRef<int64_t> canonShape,
                           ArrayRef<int64_t> perm) -> SmallVector<int64_t> {
      if (perm.empty())
        return SmallVector<int64_t>(canonShape);
      return applyPermutation(canonShape, invertPermutationVector(perm));
    };
    return Conv1DShapes{inverseApply(canonical.lhs, perms.lhs),
                        inverseApply(canonical.rhs, perms.rhs),
                        inverseApply(canonical.res, perms.res)};
  }

  /// Read operand tensors and pre-transpose to canonical NWC form. `rhs` is
  /// nullptr for pooling.
  Conv1DOperandVecs readAndTranspose(const Conv1DShapes &readShapes,
                                     const Conv1DPerms &perms) {
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Type lhsEltType = lhsShapedType.getElementType();
    Type rhsEltType = rhsShapedType.getElementType();
    Type resEltType = resShapedType.getElementType();

    SmallVector<Value> lhsIndices(readShapes.lhs.size(), zero);
    SmallVector<Value> rhsIndices(readShapes.rhs.size(), zero);
    SmallVector<Value> resIndices(readShapes.res.size(), zero);

    Conv1DOperandVecs vecs;
    vecs.lhs = vector::TransferReadOp::create(
        rewriter, loc, VectorType::get(readShapes.lhs, lhsEltType), lhsShaped,
        lhsIndices, arith::getZeroConstant(rewriter, loc, lhsEltType));
    if (convKind() == ConvOperationKind::Conv)
      vecs.rhs = vector::TransferReadOp::create(
          rewriter, loc, VectorType::get(readShapes.rhs, rhsEltType), rhsShaped,
          rhsIndices, arith::getZeroConstant(rewriter, loc, rhsEltType));
    vecs.res = vector::TransferReadOp::create(
        rewriter, loc, VectorType::get(readShapes.res, resEltType), resShaped,
        resIndices, arith::getZeroConstant(rewriter, loc, resEltType));

    if (!perms.lhs.empty())
      vecs.lhs =
          vector::TransposeOp::create(rewriter, loc, vecs.lhs, perms.lhs);
    if (!perms.rhs.empty() && vecs.rhs)
      vecs.rhs =
          vector::TransposeOp::create(rewriter, loc, vecs.rhs, perms.rhs);
    if (!perms.res.empty())
      vecs.res =
          vector::TransposeOp::create(rewriter, loc, vecs.res, perms.res);

    return vecs;
  }

  /// Extract slices, perform the contraction/pooling, and re-insert slices.
  /// All operands must already be in canonical NWC form.
  Value computeConv1D(const Conv1DOperandVecs &vecs, const Conv1DSizes &sizes,
                      int64_t wSizeStep) {
    auto linearIndex = [&](int64_t kw, int64_t w) {
      return kw * (sizes.w / wSizeStep) + w / wSizeStep;
    };

    SmallVector<Value> lhsVals = extractConvInputSlices(
        rewriter, loc, vecs.lhs, sizes.n, sizes.w, sizes.c, sizes.kw, strideW,
        dilationW, wSizeStep, config.layout);
    SmallVector<Value> rhsVals;
    if (convKind() == ConvOperationKind::Conv)
      rhsVals = extractConvFilterSlices(rewriter, loc, vecs.rhs, sizes.kw);
    SmallVector<Value> resVals =
        extractConvResultSlices(rewriter, loc, vecs.res, sizes.n, sizes.w,
                                sizes.f, wSizeStep, config.layout);

    for (int64_t kw = 0; kw < sizes.kw; ++kw) {
      for (int64_t w = 0; w < sizes.w; w += wSizeStep) {
        int64_t idx = linearIndex(kw, w);
        int64_t wIdx = w / wSizeStep;
        if (convKind() == ConvOperationKind::Pool) {
          resVals[wIdx] =
              pool1dSlice(rewriter, loc, lhsVals[idx], resVals[wIdx]);
          continue;
        }
        switch (config.layout) {
        case ConvLayoutKind::Scalar:
          resVals[wIdx] = conv1dSliceAsOuterProduct(rewriter, loc, lhsVals[idx],
                                                    rhsVals[kw], resVals[wIdx]);
          break;
        case ConvLayoutKind::Batched:
          resVals[wIdx] = conv1dSliceAsContraction(rewriter, loc, lhsVals[idx],
                                                   rhsVals[kw], resVals[wIdx]);
          break;
        case ConvLayoutKind::Batchless:
          resVals[wIdx] = conv1dBatchlessSliceAsContraction(
              rewriter, loc, lhsVals[idx], rhsVals[kw], resVals[wIdx]);
          break;
        }
      }
    }

    return insertConvResultSlices(rewriter, loc, vecs.res, sizes.w, wSizeStep,
                                  resVals, config.layout);
  }

  /// Post-transpose the result back to the original layout and write it.
  FailureOr<Operation *> writeAndPostTranspose(Value res,
                                               ArrayRef<int64_t> writeShape,
                                               ArrayRef<int64_t> resPerm) {
    if (!resPerm.empty()) {
      SmallVector<int64_t> invResPerm = invertPermutationVector(resPerm);
      res = vector::TransposeOp::create(rewriter, loc, res, invResPerm);
    }
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> resIndices(writeShape.size(), zero);
    return vector::TransferWriteOp::create(rewriter, loc, res, resShaped,
                                           resIndices)
        .getOperation();
  }

  /// Generic entry point that handles any 1D convolution/pooling layout.
  FailureOr<Operation *> generate(ArrayRef<int64_t> inputVecSizes = {},
                                  ArrayRef<bool> inputScalableVecDims = {},
                                  bool flatten1DDepthwiseConv = false) {
    if (config.isDepthwise)
      return handleDepthwise(inputVecSizes, inputScalableVecDims,
                             flatten1DDepthwiseConv);

    LinalgOp linalgOp = cast<LinalgOp>(op);
    Conv1DSizes sizes = computeConvSizes(linalgOp);
    Conv1DPerms perms = computeCanonicalPerms(linalgOp, config);
    Conv1DShapes canonShapes = computeCanonicalShapes(sizes);
    Conv1DShapes readShapes = computeReadShapes(canonShapes, perms);

    Conv1DOperandVecs vecs = readAndTranspose(readShapes, perms);

    int64_t wSizeStep = strideW == 1 ? sizes.w : 1;
    Value res = computeConv1D(vecs, sizes, wSizeStep);

    return writeAndPostTranspose(res, readShapes.res, perms.res);
  }

private:
  Conv1DConfig config;
  StringAttr redOp;
  StringAttr poolExtOp;
  bool isPoolExt = false;
  int strideW = 0;
  int dilationW = 0;
  Value lhsShaped, rhsShaped, resShaped;
  ShapedType lhsShapedType, rhsShapedType, resShapedType;
  vector::CombiningKind reductionKind;

  /// Convenience accessors.
  const ConvolutionDimensions &dims() const { return config.dims; }
  ConvOperationKind convKind() const { return config.convKind; }

  /// Inspect the reduce op body and set `isPoolExt`/`poolExtOp` if this is a
  /// pooling op whose value is fed by a cast of a block argument.
  void setPoolExtFromReduce(Operation *reduceOp) {
    int numBlockArguments =
        llvm::count_if(reduceOp->getOperands(), llvm::IsaPred<BlockArgument>);
    if (numBlockArguments != 1)
      return;
    auto feedValIt = llvm::find_if_not(reduceOp->getOperands(),
                                       llvm::IsaPred<BlockArgument>);
    Operation *feedOp = (*feedValIt).getDefiningOp();
    if (isCastOfBlockArgument(feedOp)) {
      isPoolExt = true;
      poolExtOp = feedOp->getName().getIdentifier();
    }
  }
};
} // namespace

/// Helper function to vectorize a LinalgOp with convolution semantics.
// TODO: extend the generic vectorization to support windows and drop this.
static FailureOr<Operation *> vectorizeConvolution(
    RewriterBase &rewriter, LinalgOp op, ArrayRef<int64_t> inputVecSizes,
    ArrayRef<bool> inputScalableVecDims, bool flatten1DDepthwiseConv) {
  FailureOr<Conv1DGenerator> conv1dGen = Conv1DGenerator::create(rewriter, op);
  if (failed(conv1dGen))
    return failure();
  return conv1dGen->generate(inputVecSizes, inputScalableVecDims,
                             flatten1DDepthwiseConv);
}

struct VectorizeConvolution : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<Operation *> resultOrFail = vectorizeConvolution(rewriter, op);
    if (failed(resultOrFail))
      return failure();
    Operation *newOp = *resultOrFail;
    if (newOp->getNumResults() == 0) {
      rewriter.eraseOp(op.getOperation());
      return success();
    }
    assert(newOp->getNumResults() == 1 && "expected single result");
    rewriter.replaceOp(op.getOperation(), newOp->getResult(0));
    return success();
  }
};

void mlir::linalg::populateConvolutionVectorizationPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<VectorizeConvolution>(patterns.getContext(), benefit);
}
