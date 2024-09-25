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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Interfaces/MaskableOpInterface.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-vectorization"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

/// Try to vectorize `convOp` as a convolution.
static FailureOr<Operation *>
vectorizeConvolution(RewriterBase &rewriter, LinalgOp convOp,
                     ArrayRef<int64_t> inputVecSizes = {},
                     ArrayRef<bool> inputVecScalableFlags = {},
                     bool flatten1DDepthwiseConv = false);

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

/// Helper function to extract the input slices after filter is unrolled along
/// kw.
static SmallVector<Value>
extractConvInputSlices(RewriterBase &rewriter, Location loc, Value input,
                       int64_t nSize, int64_t wSize, int64_t cSize,
                       int64_t kwSize, int strideW, int dilationW,
                       int64_t wSizeStep, bool isSingleChanneled) {
  SmallVector<Value> result;
  if (isSingleChanneled) {
    // Extract input slice of size {wSizeStep} @ [w + kw] for non-channeled
    // convolution.
    SmallVector<int64_t> sizes{wSizeStep};
    SmallVector<int64_t> strides{1};
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      for (int64_t w = 0; w < wSize; w += wSizeStep) {
        result.push_back(rewriter.create<vector::ExtractStridedSliceOp>(
            loc, input, /*offsets=*/ArrayRef<int64_t>{w + kw}, sizes, strides));
      }
    }
  } else {
    // Extract lhs slice of size {n, wSizeStep, c} @ [0, sw * w + dw * kw, 0]
    // for channeled convolution.
    SmallVector<int64_t> sizes{nSize, wSizeStep, cSize};
    SmallVector<int64_t> strides{1, 1, 1};
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      for (int64_t w = 0; w < wSize; w += wSizeStep) {
        result.push_back(rewriter.create<vector::ExtractStridedSliceOp>(
            loc, input,
            /*offsets=*/ArrayRef<int64_t>{0, w * strideW + kw * dilationW, 0},
            sizes, strides));
      }
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
    result.push_back(rewriter.create<vector::ExtractOp>(
        loc, filter, /*offsets=*/ArrayRef<int64_t>{kw}));
  }
  return result;
}

/// Helper function to extract the result slices after filter is unrolled along
/// kw.
static SmallVector<Value>
extractConvResultSlices(RewriterBase &rewriter, Location loc, Value res,
                        int64_t nSize, int64_t wSize, int64_t fSize,
                        int64_t wSizeStep, bool isSingleChanneled) {
  SmallVector<Value> result;
  if (isSingleChanneled) {
    // Extract res slice: {wSizeStep} @ [w] for non-channeled convolution.
    SmallVector<int64_t> sizes{wSizeStep};
    SmallVector<int64_t> strides{1};
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      result.push_back(rewriter.create<vector::ExtractStridedSliceOp>(
          loc, res, /*offsets=*/ArrayRef<int64_t>{w}, sizes, strides));
    }
  } else {
    // Extract res slice: {n, wSizeStep, f} @ [0, w, 0] for channeled
    // convolution.
    SmallVector<int64_t> sizes{nSize, wSizeStep, fSize};
    SmallVector<int64_t> strides{1, 1, 1};
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      result.push_back(rewriter.create<vector::ExtractStridedSliceOp>(
          loc, res, /*offsets=*/ArrayRef<int64_t>{0, w, 0}, sizes, strides));
    }
  }
  return result;
}

/// Helper function to insert the computed result slices.
static Value insertConvResultSlices(RewriterBase &rewriter, Location loc,
                                    Value res, int64_t wSize, int64_t wSizeStep,
                                    SmallVectorImpl<Value> &resVals,
                                    bool isSingleChanneled) {

  if (isSingleChanneled) {
    // Write back res slice: {wSizeStep} @ [w] for non-channeled convolution.
    // This does not depend on kw.
    SmallVector<int64_t> strides{1};
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      res = rewriter.create<vector::InsertStridedSliceOp>(
          loc, resVals[w], res, /*offsets=*/ArrayRef<int64_t>{w}, strides);
    }
  } else {
    // Write back res slice: {n, wSizeStep, f} @ [0, w, 0] for channeled
    // convolution. This does not depend on kw.
    SmallVector<int64_t> strides{1, 1, 1};
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      res = rewriter.create<vector::InsertStridedSliceOp>(
          loc, resVals[w], res, /*offsets=*/ArrayRef<int64_t>{0, w, 0},
          strides);
    }
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
                          ArrayRef<bool> inputScalableVecDims);

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
  /// permuted using `maybeMaskingMap`.
  Operation *
  maskOperation(RewriterBase &rewriter, Operation *opToMask, LinalgOp linalgOp,
                std::optional<AffineMap> maybeMaskingMap = std::nullopt);

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
};

LogicalResult
VectorizationState::precomputeIterSpaceValueSizes(RewriterBase &rewriter,
                                                  LinalgOp linalgOp) {
  // TODO: Support 0-d vectors.
  for (int vecDim = 0, end = canonicalVecShape.size(); vecDim < end; ++vecDim) {
    if (!ShapedType::isDynamic(iterSpaceStaticSizes[vecDim])) {
      // Create constant index op for static dimensions.
      iterSpaceValueSizes.push_back(rewriter.create<arith::ConstantIndexOp>(
          linalgOp.getLoc(), iterSpaceStaticSizes[vecDim]));
      continue;
    }

    // Find an operand defined on this dimension of the iteration space to
    // extract the runtime dimension size.
    Value operand;
    unsigned operandDimPos;
    if (failed(linalgOp.mapIterationSpaceDimToOperandDim(vecDim, operand,
                                                         operandDimPos)))
      return failure();

    Value dynamicDim = linalgOp.hasPureTensorSemantics()
                           ? (Value)rewriter.create<tensor::DimOp>(
                                 linalgOp.getLoc(), operand, operandDimPos)
                           : (Value)rewriter.create<memref::DimOp>(
                                 linalgOp.getLoc(), operand, operandDimPos);
    iterSpaceValueSizes.push_back(dynamicDim);
  }

  return success();
}

/// Initializes the vectorization state, including the computation of the
/// canonical vector shape for vectorization.
// TODO: Move this to the constructor when we can remove the failure cases.
LogicalResult
VectorizationState::initState(RewriterBase &rewriter, LinalgOp linalgOp,
                              ArrayRef<int64_t> inputVectorSizes,
                              ArrayRef<bool> inputScalableVecDims) {
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

  LDBG("Canonical vector shape: ");
  LLVM_DEBUG(llvm::interleaveComma(canonicalVecShape, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");
  LDBG("Scalable vector dims: ");
  LLVM_DEBUG(llvm::interleaveComma(scalableVecDims, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

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

  LDBG("Masking map: " << maskingMap << "\n");

  // Return the active mask for the masking map of this operation if it was
  // already created.
  auto activeMaskIt = activeMaskCache.find(maskingMap);
  if (activeMaskIt != activeMaskCache.end()) {
    Value mask = activeMaskIt->second;
    LDBG("Reusing mask: " << mask << "\n");
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

  LDBG("Mask shape: ");
  LLVM_DEBUG(llvm::interleaveComma(maskShape, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  if (permutedStaticSizes == maskShape) {
    LDBG("Masking is not needed for masking map: " << maskingMap << "\n");
    activeMaskCache[maskingMap] = Value();
    return Value();
  }

  // Permute the iteration space value sizes to compute the mask upper bounds.
  SmallVector<Value> upperBounds =
      applyPermutationMap(maskingMap, ArrayRef<Value>(iterSpaceValueSizes));
  assert(!maskShape.empty() && !upperBounds.empty() &&
         "Masked 0-d vectors are not supported yet");

  // Create the mask based on the dimension values.
  Value mask = rewriter.create<vector::CreateMaskOp>(linalgOp.getLoc(),
                                                     maskType, upperBounds);
  LDBG("Creating new mask: " << mask << "\n");
  activeMaskCache[maskingMap] = mask;
  return mask;
}

/// Masks an operation with the canonical vector mask if the operation needs
/// masking. Returns the masked operation or the original operation if masking
/// is not needed. If provided, the canonical mask for this operation is
/// permuted using `maybeMaskingMap`.
Operation *
VectorizationState::maskOperation(RewriterBase &rewriter, Operation *opToMask,
                                  LinalgOp linalgOp,
                                  std::optional<AffineMap> maybeMaskingMap) {
  LDBG("Trying to mask: " << *opToMask << "\n");

  // Create or retrieve mask for this operation.
  Value mask =
      getOrCreateMaskFor(rewriter, opToMask, linalgOp, maybeMaskingMap);

  if (!mask) {
    LDBG("No mask required\n");
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

  LDBG("Masked operation: " << *maskOp << "\n");
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
  assert(res.getNumDims() == res.getNumResults() &&
         "expected reindexed map with same number of dims and results");
  return res;
}

/// Helper enum to represent conv1d input traversal order.
enum class Conv1DOpOrder {
  W,   // Corresponds to non-channeled 1D convolution operation.
  Ncw, // Corresponds to operation that traverses the input in (n, c, w) order.
  Nwc  // Corresponds to operation that traverses the input in (n, w, c) order.
};

/// Helper data structure to represent the result of vectorization.
/// In certain specific cases, like terminators, we do not want to propagate/
enum VectorizationStatus {
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
struct VectorizationResult {
  /// Return status from vectorizing the current op.
  enum VectorizationStatus status = VectorizationStatus::Failure;
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
      .Case<arith::AndIOp>([&](auto op) { return CombiningKind::AND; })
      .Case<arith::MaxSIOp>([&](auto op) { return CombiningKind::MAXSI; })
      .Case<arith::MaxUIOp>([&](auto op) { return CombiningKind::MAXUI; })
      .Case<arith::MaximumFOp>([&](auto op) { return CombiningKind::MAXIMUMF; })
      .Case<arith::MaxNumFOp>([&](auto op) { return CombiningKind::MAXNUMF; })
      .Case<arith::MinSIOp>([&](auto op) { return CombiningKind::MINSI; })
      .Case<arith::MinUIOp>([&](auto op) { return CombiningKind::MINUI; })
      .Case<arith::MinimumFOp>([&](auto op) { return CombiningKind::MINIMUMF; })
      .Case<arith::MinNumFOp>([&](auto op) { return CombiningKind::MINNUMF; })
      .Case<arith::MulIOp, arith::MulFOp>(
          [&](auto op) { return CombiningKind::MUL; })
      .Case<arith::OrIOp>([&](auto op) { return CombiningKind::OR; })
      .Case<arith::XOrIOp>([&](auto op) { return CombiningKind::XOR; })
      .Default([&](auto op) { return std::nullopt; });
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
  return b.create<vector::MultiDimReductionOp>(
      reduceOp->getLoc(), valueToReduce, acc, dimsToMask, *maybeKind);
}

static SmallVector<bool> getDimsToReduce(LinalgOp linalgOp) {
  return llvm::to_vector(
      llvm::map_range(linalgOp.getIteratorTypesArray(), isReductionIterator));
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

  Operation *write;
  if (vectorType.getRank() > 0) {
    AffineMap writeMap = inversePermutation(reindexIndexingMap(opOperandMap));
    SmallVector<Value> indices(linalgOp.getRank(outputOperand),
                               rewriter.create<arith::ConstantIndexOp>(loc, 0));
    value = broadcastIfNeeded(rewriter, value, vectorType);
    assert(value.getType() == vectorType && "Incorrect type");
    write = rewriter.create<vector::TransferWriteOp>(
        loc, value, outputOperand->get(), indices, writeMap);
  } else {
    // 0-d case is still special: do not invert the reindexing writeMap.
    if (!isa<VectorType>(value.getType()))
      value = rewriter.create<vector::BroadcastOp>(loc, vectorType, value);
    assert(value.getType() == vectorType && "Incorrect type");
    write = rewriter.create<vector::TransferWriteOp>(
        loc, value, outputOperand->get(), ValueRange{});
  }

  write = state.maskOperation(rewriter, write, linalgOp, opOperandMap);

  // If masked, set in-bounds to true. Masking guarantees that the access will
  // be in-bounds.
  if (auto maskOp = dyn_cast<vector::MaskingOpInterface>(write)) {
    auto maskedWriteOp = cast<vector::TransferWriteOp>(maskOp.getMaskableOp());
    SmallVector<bool> inBounds(maskedWriteOp.getVectorType().getRank(), true);
    maskedWriteOp.setInBoundsAttr(rewriter.getBoolArrayAttr(inBounds));
  }

  LDBG("vectorized op: " << *write << "\n");
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
    std::function<VectorizationResult(Operation *, const IRMapping &)>;

/// Helper function to vectorize the terminator of a `linalgOp`. New result
/// vector values are appended to `newResults`. Return
/// VectorizationStatus::NoReplace to signal the vectorization algorithm that it
/// should not try to map produced operations and instead return the results
/// using the `newResults` vector making them available to the vectorization
/// algorithm for RAUW. This function is meant to be used as a
/// CustomVectorizationHook.
static VectorizationResult
vectorizeLinalgYield(RewriterBase &rewriter, Operation *op,
                     const IRMapping &bvm, VectorizationState &state,
                     LinalgOp linalgOp, SmallVectorImpl<Value> &newResults) {
  auto yieldOp = dyn_cast<linalg::YieldOp>(op);
  if (!yieldOp)
    return VectorizationResult{VectorizationStatus::Failure, nullptr};
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

  return VectorizationResult{VectorizationStatus::NoReplace, nullptr};
}

/// Helper function to vectorize the index operations of a `linalgOp`. Return
/// VectorizationStatus::NewOp to signal the vectorization algorithm that it
/// should map the produced operations. This function is meant to be used as a
/// CustomVectorizationHook.
static VectorizationResult vectorizeLinalgIndex(RewriterBase &rewriter,
                                                VectorizationState &state,
                                                Operation *op,
                                                LinalgOp linalgOp) {
  IndexOp indexOp = dyn_cast<linalg::IndexOp>(op);
  if (!indexOp)
    return VectorizationResult{VectorizationStatus::Failure, nullptr};
  auto loc = indexOp.getLoc();
  // Compute the static loop sizes of the index op.
  ArrayRef<int64_t> targetShape = state.getCanonicalVecShape();
  auto dim = indexOp.getDim();
  // Compute a one-dimensional index vector for the index op dimension.
  auto indexVectorType =
      VectorType::get({targetShape[dim]}, rewriter.getIndexType(),
                      state.getScalableVecDims()[dim]);
  auto indexSteps = rewriter.create<vector::StepOp>(loc, indexVectorType);
  // Return the one-dimensional index vector if it lives in the trailing
  // dimension of the iteration space since the vectorization algorithm in this
  // case can handle the broadcast.
  if (dim == targetShape.size() - 1)
    return VectorizationResult{VectorizationStatus::NewOp, indexSteps};
  // Otherwise permute the targetShape to move the index dimension last,
  // broadcast the one-dimensional index vector to the permuted shape, and
  // finally transpose the broadcasted index vector to undo the permutation.
  auto permPattern =
      llvm::to_vector(llvm::seq<unsigned>(0, targetShape.size()));
  std::swap(permPattern[dim], permPattern.back());
  auto permMap =
      AffineMap::getPermutationMap(permPattern, linalgOp.getContext());

  auto broadCastOp = rewriter.create<vector::BroadcastOp>(
      loc, state.getCanonicalVecType(rewriter.getIndexType(), permMap),
      indexSteps);
  SmallVector<int64_t> transposition =
      llvm::to_vector<16>(llvm::seq<int64_t>(0, linalgOp.getNumLoops()));
  std::swap(transposition.back(), transposition[dim]);
  auto transposeOp =
      rewriter.create<vector::TransposeOp>(loc, broadCastOp, transposition);
  return VectorizationResult{VectorizationStatus::NewOp, transposeOp};
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

  if (llvm::any_of(extractOp->getResultTypes(), [](Type type) {
        return !VectorType::isValidElementType(type);
      })) {
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
    Value dimIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);

    auto dimSize = broadcastIfNeeded(
        rewriter,
        rewriter.create<tensor::DimOp>(loc, extractOp.getTensor(), dimIdx),
        indexVecType);

    offset = rewriter.create<arith::MulIOp>(loc, offset, dimSize);

    auto extractOpIndex = broadcastIfNeeded(
        rewriter, bvm.lookup(extractOp.getIndices()[i]), indexVecType);

    offset = rewriter.create<arith::AddIOp>(loc, extractOpIndex, offset);
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
  assert(linalgOp.hasDynamicShape() ||
         llvm::count_if(loopRanges, [](int64_t dim) { return dim != 1; }) ==
                 1 &&
             "For statically shaped Linalg Ops, only one "
             "non-unit loop dim is expected");

  size_t idx = loopRanges.size() - 1;
  for (; idx >= 0; idx--)
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
    return llvm::all_of(block->getArguments(),
                        [&val](Value v) { return (v != val); });

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
    return llvm::all_of(block->getArguments(),
                        [&val](Value v) { return (v != val); });

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
    LDBG("Found gather load: " << extractOp);
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
    LDBG("Found scalar broadcast load: " << extractOp);

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
    LDBG("Found contigous load: " << extractOp);
    return VectorMemoryAccessKind::Contiguous;
  }

  // 4. Fallback case - gather load.
  LDBG("Found gather load: " << extractOp);
  return VectorMemoryAccessKind::Gather;
}

/// Helper function to vectorize the tensor.extract operations. Returns
/// VectorizationStatus::NewOp to signal the vectorization algorithm that it
/// should map the produced operations. This function is meant to be used as a
/// CustomVectorizationHook.
static VectorizationResult
vectorizeTensorExtract(RewriterBase &rewriter, VectorizationState &state,
                       Operation *op, LinalgOp linalgOp, const IRMapping &bvm) {
  tensor::ExtractOp extractOp = dyn_cast<tensor::ExtractOp>(op);
  if (!extractOp)
    return VectorizationResult{VectorizationStatus::Failure, nullptr};
  auto loc = extractOp.getLoc();

  // Compute the static loop sizes of the extract op.
  auto resultType = state.getCanonicalVecType(extractOp.getResult().getType());
  auto maskConstantOp = rewriter.create<arith::ConstantOp>(
      loc,
      DenseIntElementsAttr::get(state.getCanonicalVecType(rewriter.getI1Type()),
                                /*value=*/true));
  auto passThruConstantOp =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resultType));

  // Base indices are currently set to 0. We will need to re-visit if more
  // generic scenarios are to be supported.
  SmallVector<Value> baseIndices(
      extractOp.getIndices().size(),
      rewriter.create<arith::ConstantIndexOp>(loc, 0));

  VectorMemoryAccessKind memAccessKind =
      getTensorExtractMemoryAccessPattern(extractOp, linalgOp, resultType);

  // 1. Handle gather access
  if (memAccessKind == VectorMemoryAccessKind::Gather) {
    Value offset = calculateGatherOffset(rewriter, state, extractOp, bvm);

    // Generate the gather load
    Operation *gatherOp = rewriter.create<vector::GatherOp>(
        loc, resultType, extractOp.getTensor(), baseIndices, offset,
        maskConstantOp, passThruConstantOp);
    gatherOp = state.maskOperation(rewriter, gatherOp, linalgOp);

    LDBG("Vectorised as gather load: " << extractOp << "\n");
    return VectorizationResult{VectorizationStatus::NewOp, gatherOp};
  }

  // 2. Handle:
  //  a. scalar loads + broadcast,
  //  b. contiguous loads.
  // Both cases use vector.transfer_read.

  assert(llvm::count_if(resultType.getShape(),
                        [](uint64_t dim) { return dim != 1; }) &&
         "Contiguous loads and scalar loads + broadcast only support 1-D "
         "vectors ATM!");

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
  auto zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getZeroAttr(rewriter.getI32Type()));
  for (size_t i = 0; i < extractOp.getIndices().size(); i++) {
    Value idx = bvm.lookup(extractOp.getIndices()[i]);
    if (idx.getType().isIndex()) {
      transferReadIdxs.push_back(idx);
      continue;
    }

    auto indexAs1dVector = rewriter.create<vector::ShapeCastOp>(
        loc,
        VectorType::get(resultType.getShape().back(), rewriter.getIndexType(),
                        resultType.getScalableDims().back()),
        idx);
    transferReadIdxs.push_back(
        rewriter.create<vector::ExtractElementOp>(loc, indexAs1dVector, zero));
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

    auto transferReadOp = rewriter.create<vector::TransferReadOp>(
        loc, resultType, extractOp.getTensor(), transferReadIdxs,
        permutationMap, inBounds);

    LDBG("Vectorised as scalar broadcast load: " << extractOp << "\n");
    return VectorizationResult{VectorizationStatus::NewOp, transferReadOp};
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

  auto transferReadOp = rewriter.create<vector::TransferReadOp>(
      loc, resultType, extractOp.getTensor(), transferReadIdxs, permutationMap,
      inBounds);

  LDBG("Vectorised as contiguous load: " << extractOp);
  return VectorizationResult{VectorizationStatus::NewOp, transferReadOp};
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
/// This function does not update `bvm` but returns a VectorizationStatus that
/// instructs the caller what `bvm` update needs to occur.
static VectorizationResult
vectorizeOneOp(RewriterBase &rewriter, VectorizationState &state,
               LinalgOp linalgOp, Operation *op, const IRMapping &bvm,
               ArrayRef<CustomVectorizationHook> customVectorizationHooks) {
  LDBG("vectorize op " << *op << "\n");

  // 1. Try to apply any CustomVectorizationHook.
  if (!customVectorizationHooks.empty()) {
    for (auto &customFunc : customVectorizationHooks) {
      VectorizationResult result = customFunc(op, bvm);
      if (result.status == VectorizationStatus::Failure)
        continue;
      return result;
    }
  }

  // 2. Constant ops don't get vectorized but rather broadcasted at their users.
  // Clone so that the constant is not confined to the linalgOp block .
  if (isa<arith::ConstantOp, func::ConstantOp>(op))
    return VectorizationResult{VectorizationStatus::NewOp, rewriter.clone(*op)};

  // 3. Only ElementwiseMappable are allowed in the generic vectorization.
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return VectorizationResult{VectorizationStatus::Failure, nullptr};

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
      return VectorizationResult{VectorizationStatus::NewOp, reduceOp};
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
  return VectorizationResult{
      VectorizationStatus::NewOp,
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
/// broadcasting makes it trivial to detrmine where broadcast, transposes and
/// reductions should occur, without any bookkeeping. The tradeoff is that, in
/// the absence of good canonicalizations, the amount of work increases.
/// This is not deemed a problem as we expect canonicalizations and foldings to
/// aggressively clean up the useless work.
static LogicalResult
vectorizeAsLinalgGeneric(RewriterBase &rewriter, VectorizationState &state,
                         LinalgOp linalgOp,
                         SmallVectorImpl<Value> &newResults) {
  LDBG("Vectorizing operation as linalg generic\n");
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
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (OpOperand *opOperand : linalgOp.getOpOperandsMatchingBBargs()) {
    BlockArgument bbarg = linalgOp.getMatchingBlockArgument(opOperand);
    if (linalgOp.isScalar(opOperand)) {
      bvm.map(bbarg, opOperand->get());
      continue;
    }

    // 3.a. Convert the indexing map for this input/output to a transfer read
    // permutation map and masking map.
    AffineMap indexingMap = linalgOp.getMatchingIndexingMap(opOperand);

    // Remove zeros from indexing map to use it as masking map.
    SmallVector<int64_t> zeroPos;
    auto results = indexingMap.getResults();
    for (const auto &result : llvm::enumerate(results)) {
      if (isa<AffineConstantExpr>(result.value())) {
        zeroPos.push_back(result.index());
      }
    }
    AffineMap maskingMap = indexingMap.dropResults(zeroPos);

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

    // Make sure that the in_bounds attribute corresponding to a broadcast dim
    // is `true`
    SmallVector<unsigned> broadcastedDims = readMap.getBroadcastDims();
    SmallVector<bool> inBounds(readType.getRank(), false);

    for (auto idx : broadcastedDims)
      inBounds[idx] = true;

    Operation *read = rewriter.create<vector::TransferReadOp>(
        loc, readType, opOperand->get(), indices, readMap,
        ArrayRef<bool>(inBounds));
    read = state.maskOperation(rewriter, read, linalgOp, maskingMap);
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
      readValue = rewriter.create<vector::ExtractElementOp>(loc, readValue);

    LDBG("New vectorized bbarg(" << bbarg.getArgNumber() << "): " << readValue
                                 << "\n");
    bvm.map(bbarg, readValue);
    bvm.map(opOperand->get(), readValue);
  }

  SmallVector<CustomVectorizationHook> hooks;
  // 4a. Register CustomVectorizationHook for yieldOp.
  CustomVectorizationHook vectorizeYield =
      [&](Operation *op, const IRMapping &bvm) -> VectorizationResult {
    return vectorizeLinalgYield(rewriter, op, bvm, state, linalgOp, newResults);
  };
  hooks.push_back(vectorizeYield);

  // 4b. Register CustomVectorizationHook for indexOp.
  CustomVectorizationHook vectorizeIndex =
      [&](Operation *op, const IRMapping &bvm) -> VectorizationResult {
    return vectorizeLinalgIndex(rewriter, state, op, linalgOp);
  };
  hooks.push_back(vectorizeIndex);

  // 4c. Register CustomVectorizationHook for extractOp.
  CustomVectorizationHook vectorizeExtract =
      [&](Operation *op, const IRMapping &bvm) -> VectorizationResult {
    return vectorizeTensorExtract(rewriter, state, op, linalgOp, bvm);
  };
  hooks.push_back(vectorizeExtract);

  // 5. Iteratively call `vectorizeOneOp` to each op in the slice.
  for (Operation &op : block->getOperations()) {
    VectorizationResult result =
        vectorizeOneOp(rewriter, state, linalgOp, &op, bvm, hooks);
    if (result.status == VectorizationStatus::Failure) {
      LDBG("failed to vectorize: " << op << "\n");
      return failure();
    }
    if (result.status == VectorizationStatus::NewOp) {
      Operation *maybeMaskedOp =
          state.maskOperation(rewriter, result.newOp, linalgOp);
      LDBG("New vector op: " << *maybeMaskedOp << "\n");
      bvm.map(op.getResults(), maybeMaskedOp->getResults());
    }
  }

  return success();
}

/// Given a tensor::PackOp, return the `dest` shape before any packing
/// permutations.
static SmallVector<int64_t> getTiledPackShape(tensor::PackOp packOp,
                                              ArrayRef<int64_t> destShape) {
  return applyPermutation(destShape, tensor::getPackInverseDestPerm(packOp));
}

/// Given an input, the mixed destSizes, and the vector sizes for vectorization,
/// create an empty destination tensor and create a TransferWriteOp from the
/// input to the empty tensor. If the destination shape is not the same as the
/// inputVectorSizes for the first rank(inputVectorSizes) dims, then create a
/// mask for the write. If `useInBoundsInsteadOfMasking` is set, then update the
/// inBounds attribute of the transfer write op instead of masking.
static Operation *createWriteOrMaskedWrite(OpBuilder &builder, Location loc,
                                           Value input,
                                           SmallVector<OpFoldResult> destSizes,
                                           ArrayRef<int64_t> inputVectorSizes,
                                           bool useInBoundsInsteadOfMasking) {

  auto inputType = cast<VectorType>(input.getType());
  Value dest = builder.create<tensor::EmptyOp>(loc, destSizes,
                                               inputType.getElementType());
  int64_t rank = cast<ShapedType>(dest.getType()).getRank();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto destShape = cast<ShapedType>(dest.getType()).getShape();
  SmallVector<bool> inBoundsVal(rank, true);
  if (useInBoundsInsteadOfMasking) {
    // Update the inBounds attribute.
    for (unsigned i = 0; i < rank; i++)
      inBoundsVal[i] = (destShape[i] == inputVectorSizes[i]) &&
                       !ShapedType::isDynamic(destShape[i]);
  }
  Operation *write = builder.create<vector::TransferWriteOp>(
      loc,
      /*vector=*/input,
      /*source=*/dest,
      /*indices=*/SmallVector<Value>(rank, zero),
      /*inBounds=*/inBoundsVal);
  assert(llvm::none_of(
             destShape.drop_front(inputVectorSizes.size()),
             [](int64_t size) { return size == ShapedType::kDynamic; }) &&
         "Only dims aligned with inputVectorSizes may be dynamic");
  if (useInBoundsInsteadOfMasking)
    return write;
  bool needMaskForWrite = !llvm::equal(
      inputVectorSizes, destShape.take_front(inputVectorSizes.size()));
  if (needMaskForWrite) {
    SmallVector<int64_t> writeMaskShape;
    writeMaskShape.append(inputVectorSizes.begin(), inputVectorSizes.end());
    writeMaskShape.append(destShape.begin() + inputVectorSizes.size(),
                          destShape.end());
    auto writeMaskType = VectorType::get(writeMaskShape, builder.getI1Type());
    Value maskForWrite =
        builder.create<vector::CreateMaskOp>(loc, writeMaskType, destSizes);
    write = mlir::vector::maskOperation(builder, write, maskForWrite);
  }
  return write;
}

/// Vectorize tensor::PackOp with (1) static innerTiles (2) constant
/// padding value and (3) input vector sizes into:
/// masked_transfer_read->shape_cast->transpose->transfer_write_in_bounds
/// As in the following example:
/// %pack = tensor.pack %src inner_dims_pos = [2, 1] inner_tiles = [16, 2]
///     into %dst : tensor<32x8x16xf32> -> tensor<32x4x1x16x2xf32>
///
/// This pack would be vectorized to:
///
/// %load = vector.mask %mask {
///     vector.transfer_read %arg0[%c0, %c0, %c0], %cst
///         {in_bounds = [true, true, true]} :
///         tensor<32x7x16xf32>, vector<32x8x16xf32>
/// } : vector<32x8x16xi1> -> vector<32x8x16xf32>
/// %shape_cast = vector.shape_cast %load : vector<32x8x16xf32>
///                                         to vector<32x4x2x1x16xf32>
/// %transpose = vector.transpose %shape_cast, [0, 1, 3, 4, 2]
///     : vector<32x4x2x1x16xf32> to vector<32x4x1x16x2xf32>
/// %write = vector.transfer_write %transpose,
///     %empty[%c0_0, %c0_0, %c0_0, %c0_0, %c0_0]
///     {in_bounds = [true, true, true, true, true]}
///     : vector<32x4x1x16x2xf32>, tensor<32x4x1x16x2xf32>
///
/// If the (3) input vector sizes are not provided, the vector sizes are
/// determined by the result tensor shape. Also, we update the inBounds
/// attribute instead of masking.
static LogicalResult
vectorizeAsTensorPackOp(RewriterBase &rewriter, tensor::PackOp packOp,
                        ArrayRef<int64_t> inputVectorSizes,
                        SmallVectorImpl<Value> &newResults) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(packOp);

  Location loc = packOp.getLoc();
  auto padValue = packOp.getPaddingValue();
  if (!padValue) {
    padValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(packOp.getSourceType().getElementType()));
  }
  ReifiedRankedShapedTypeDims reifiedReturnShapes;
  LogicalResult status =
      cast<ReifyRankedShapedTypeOpInterface>(packOp.getOperation())
          .reifyResultShapes(rewriter, reifiedReturnShapes);
  (void)status; // prevent unused variable warning on non-assert builds.
  assert(succeeded(status) && "failed to reify result shapes");

  // If the input vector sizes are not provided, then the vector sizes are
  // determined by the result tensor shape. In case the vector sizes aren't
  // provided, we update the inBounds attribute instead of masking.
  bool useInBoundsInsteadOfMasking = false;
  if (inputVectorSizes.empty()) {
    ArrayRef<int64_t> resultTensorShape = packOp.getDestType().getShape();
    inputVectorSizes = resultTensorShape.take_front(packOp.getSourceRank());
    useInBoundsInsteadOfMasking = true;
  }

  // Create masked TransferReadOp.
  SmallVector<int64_t> inputShape(inputVectorSizes);
  auto innerTiles = packOp.getStaticInnerTiles();
  auto innerDimsPos = packOp.getInnerDimsPos();
  auto outerDimsPerm = packOp.getOuterDimsPerm();
  if (!outerDimsPerm.empty())
    applyPermutationToVector(inputShape,
                             invertPermutationVector(outerDimsPerm));
  for (auto [idx, size] : enumerate(innerTiles))
    inputShape[innerDimsPos[idx]] *= size;
  auto maskedRead = vector::createReadOrMaskedRead(
      rewriter, loc, packOp.getSource(), inputShape, padValue,
      useInBoundsInsteadOfMasking);

  // Create ShapeCastOp.
  SmallVector<int64_t> destShape(inputVectorSizes);
  destShape.append(innerTiles.begin(), innerTiles.end());
  auto tiledPackType = VectorType::get(getTiledPackShape(packOp, destShape),
                                       packOp.getDestType().getElementType());
  auto shapeCastOp =
      rewriter.create<vector::ShapeCastOp>(loc, tiledPackType, maskedRead);

  // Create TransposeOp.
  auto destPermutation =
      invertPermutationVector(tensor::getPackInverseDestPerm(packOp));
  auto transposeOp = rewriter.create<vector::TransposeOp>(
      loc, shapeCastOp.getResult(), destPermutation);

  // Create TransferWriteOp.
  Operation *write = createWriteOrMaskedWrite(
      rewriter, loc, transposeOp.getResult(), reifiedReturnShapes[0],
      inputVectorSizes, /*useInBoundsInsteadOfMasking=*/false);
  newResults.push_back(write->getResult(0));
  return success();
}

/// Vectorize a `tensor::UnPackOp` to these 4 Ops:
///   Vector::TransferReadOp - Reads a vector from the source tensor
///   vector::TransposeOp - Transpose the Source tensor
///   ShapeCastOp - Reshape the data based on the target.
///   vector::TransferWriteOp. - Write the result vector back to the destination
///   tensor.
///   If the vector sizes are not provided:
///   * the vector sizes are determined by the input operand and attributes,
///   * update the inBounds attribute instead of masking.
static LogicalResult
vectorizeAsTensorUnpackOp(RewriterBase &rewriter, tensor::UnPackOp unpackOp,
                          ArrayRef<int64_t> inputVectorSizes,
                          SmallVectorImpl<Value> &newResults) {

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(unpackOp);

  RankedTensorType unpackTensorType = unpackOp.getSourceType();

  ArrayRef<int64_t> innerDimPos = unpackOp.getInnerDimsPos();
  ArrayRef<int64_t> innerTiles = unpackOp.getStaticInnerTiles();
  ArrayRef<int64_t> sourceShape = unpackTensorType.getShape();
  bool useInBoundsInsteadOfMasking = false;
  ArrayRef<int64_t> outerDimsPerm = unpackOp.getOuterDimsPerm();

  auto destSize = unpackOp.getDestRank();

  if (!inputVectorSizes.empty())
    assert(inputVectorSizes.size() == destSize &&
           "Incorrect number of input vector sizes");

  // vectorSizes is the shape of the vector that will be used to do final
  // write on the destination tensor. It is set like this: Let's say the
  // source tensor is rank 'M' and the dest tensor rank 'N', where N <= M.
  // Thus:
  // 1. vectorSizes = sourceShape.take_front(N)
  // 2. if outer_dims_perms is present: do that permutation on vectorSizes.
  // 3. multiply all the locations in vectorSize pointed by innerDimPos by the
  //    innerTiles attribute value.
  SmallVector<int64_t> vectorSizes(inputVectorSizes);
  if (vectorSizes.empty()) {
    llvm::append_range(vectorSizes, sourceShape.take_front(destSize));
    if (!outerDimsPerm.empty())
      applyPermutationToVector(vectorSizes, outerDimsPerm);
    for (auto [i, pos] : llvm::enumerate(innerDimPos))
      vectorSizes[pos] *= innerTiles[i];

    useInBoundsInsteadOfMasking = true;
  }

  // readVectorSizes is the size of tensor used to read and apply mask. It is
  // set like this: Let's say the vectorSize (VS) array is size 'N' and
  // the sourceShape(SS) is 'M' where M >= N and InnerTileSizes (IT) of
  // size M-N
  // Thus:
  // - initially: readVectorSizes = vectorInputSizes
  // - Divide all the readMaskShape locations pointed by innerDimPos
  //   by the innerTileSize attribute value.
  // - if outer_dims_perms is present: do that permutation on readVectorSizes.
  // - Append the remaining shape from SS
  // E.g. let's say let's say unpackTensorType.getShape() = <8x8x32x16>
  // inner Dim Pos = [0, 1] and Inner Tiles = [32, 16], vector_sizes are [512,
  // 128] and outer_dims_perm is [1, 0] then read shape is:
  //   ReadVectorSizes(initial): [512, 128]
  //   Final Value(after innerDim Adjustment): [512/32, 128/16]
  //                                           = [16, 8]
  //   After applying outer_dims_perm: [8, 16]
  //   After appending the rest of the sourceShape: [8, 16, 32, 16]

  SmallVector<int64_t> readVectorSizes(vectorSizes.begin(), vectorSizes.end());

  for (auto [index, size] : enumerate(innerTiles)) {
    readVectorSizes[innerDimPos[index]] =
        llvm::divideCeil(readVectorSizes[innerDimPos[index]], size);
  }
  if (!outerDimsPerm.empty()) {
    applyPermutationToVector(readVectorSizes, outerDimsPerm);
  }
  readVectorSizes.append(sourceShape.begin() + vectorSizes.size(),
                         sourceShape.end());

  ReifiedRankedShapedTypeDims reifiedRetShapes;
  LogicalResult status =
      cast<ReifyRankedShapedTypeOpInterface>(unpackOp.getOperation())
          .reifyResultShapes(rewriter, reifiedRetShapes);
  if (status.failed()) {
    LDBG("Unable to reify result shapes of " << unpackOp);
    return failure();
  }
  Location loc = unpackOp->getLoc();

  auto padValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(unpackOp.getSourceType().getElementType()));

  // Read result, mask if necessary. If transferReadOp shape is not equal
  // to shape of source, then a mask is necessary.
  Value readResult = vector::createReadOrMaskedRead(
      rewriter, loc, unpackOp.getSource(), readVectorSizes, padValue,
      /*useInBoundsInsteadOfMasking=*/false);

  PackingMetadata packMetadata;
  SmallVector<int64_t> lastDimToInsertPosPerm =
      tensor::getUnPackInverseSrcPerm(unpackOp, packMetadata);
  ShapedType maskedOpShapedType = cast<ShapedType>(readResult.getType());
  SmallVector<int64_t> stripMineShape(maskedOpShapedType.getShape());
  mlir::Type stripMineElemType = maskedOpShapedType.getElementType();
  applyPermutationToVector(stripMineShape, lastDimToInsertPosPerm);
  RankedTensorType stripMineTensorType =
      RankedTensorType::get(stripMineShape, stripMineElemType);
  // Transpose the appropriate rows to match output.
  vector::TransposeOp transposeOp = rewriter.create<vector::TransposeOp>(
      loc, readResult, lastDimToInsertPosPerm);

  // Collapse the vector to the size required by result.
  RankedTensorType collapsedType = tensor::CollapseShapeOp::inferCollapsedType(
      stripMineTensorType, packMetadata.reassociations);
  mlir::VectorType vecCollapsedType =
      VectorType::get(collapsedType.getShape(), collapsedType.getElementType());
  vector::ShapeCastOp shapeCastOp = rewriter.create<vector::ShapeCastOp>(
      loc, vecCollapsedType, transposeOp->getResult(0));

  // writeVectorSizes had to match the shapecast shape for dynamic sizes,
  // otherwise the validator complains that the mask size is invalid.
  SmallVector<int64_t> writeVectorSizes(
      unpackOp.getDestType().hasStaticShape()
          ? vectorSizes
          : shapeCastOp.getResultVectorType().getShape());
  Operation *write = createWriteOrMaskedWrite(
      rewriter, loc, shapeCastOp.getResult(), reifiedRetShapes[0],
      writeVectorSizes, useInBoundsInsteadOfMasking);
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

  // transfer_write_in_bounds(transfer_read_masked(pad_source, pad_value))
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(padOp);

  ReifiedRankedShapedTypeDims reifiedReturnShapes;
  LogicalResult status =
      cast<ReifyRankedShapedTypeOpInterface>(padOp.getOperation())
          .reifyResultShapes(rewriter, reifiedReturnShapes);
  (void)status; // prevent unused variable warning on non-assert builds
  assert(succeeded(status) && "failed to reify result shapes");
  auto maskedRead = vector::createReadOrMaskedRead(
      rewriter, loc, padOp.getSource(), inputVectorSizes, padValue,
      /*useInBoundsInsteadOfMasking=*/false);
  Operation *write = createWriteOrMaskedWrite(
      rewriter, loc, maskedRead, reifiedReturnShapes[0], inputVectorSizes,
      /*useInBoundsInsteadOfMasking=*/false);
  newResults.push_back(write->getResult(0));
  return success();
}

// TODO: probably need some extra checks for reduction followed by consumer
// ops that may not commute (e.g. linear reduction + non-linear instructions).
static LogicalResult reductionPreconditions(LinalgOp op) {
  if (llvm::none_of(op.getIteratorTypesArray(), isReductionIterator)) {
    LDBG("reduction precondition failed: no reduction iterator\n");
    return failure();
  }
  for (OpOperand &opOperand : op.getDpsInitsMutable()) {
    AffineMap indexingMap = op.getMatchingIndexingMap(&opOperand);
    if (indexingMap.isPermutation())
      continue;

    Operation *reduceOp = matchLinalgReduction(&opOperand);
    if (!reduceOp || !getCombinerOpKind(reduceOp)) {
      LDBG("reduction precondition failed: reduction detection failed\n");
      return failure();
    }
  }
  return success();
}

static LogicalResult
vectorizeDynamicConvOpPrecondition(linalg::LinalgOp conv,
                                   bool flatten1DDepthwiseConv) {
  if (flatten1DDepthwiseConv) {
    LDBG("Vectorization of flattened convs with dynamic shapes is not "
         "supported\n");
    return failure();
  }

  if (!isa<linalg::DepthwiseConv1DNwcWcOp>(conv)) {
    LDBG("Not a 1D depth-wise WC conv, dynamic shapes are not supported\n");
    return failure();
  }

  // Support dynamic shapes in 1D depthwise convolution, but only in the
  // _channel_ dimension.
  Value lhs = conv.getDpsInputOperand(0)->get();
  ArrayRef<int64_t> lhsShape = cast<ShapedType>(lhs.getType()).getShape();
  auto shapeWithoutCh = lhsShape.drop_back(1);
  if (ShapedType::isDynamicShape(shapeWithoutCh)) {
    LDBG("Dynamically-shaped op vectorization precondition failed: only "
         "channel dim can be dynamic\n");
    return failure();
  }

  return success();
}

static LogicalResult
vectorizeDynamicLinalgOpPrecondition(linalg::LinalgOp op,
                                     bool flatten1DDepthwiseConv) {
  if (isa<ConvolutionOpInterface>(op.getOperation()))
    return vectorizeDynamicConvOpPrecondition(op, flatten1DDepthwiseConv);

  if (hasReductionIterator(op))
    return reductionPreconditions(op);

  // TODO: Masking only supports dynamic element-wise ops, linalg.generic ops,
  // linalg.copy ops and ops that implement ContractionOpInterface for now.
  if (!isElementwise(op) &&
      !isa<linalg::GenericOp, linalg::CopyOp, linalg::ContractionOpInterface>(
          op.getOperation()))
    return failure();

  LDBG("Dynamically-shaped op meets vectorization pre-conditions\n");
  return success();
}

/// Need to check if the inner-tiles are static/constant.
static LogicalResult
vectorizeUnPackOpPrecondition(tensor::UnPackOp unpackOp,
                              ArrayRef<int64_t> inputVectorSizes) {

  if (llvm::any_of(unpackOp.getInnerTiles(), [](OpFoldResult res) {
        return !getConstantIntValue(res).has_value();
      })) {
    LDBG("Inner-tiles must be constant: " << unpackOp << "\n");
    return failure();
  }
  ArrayRef<int64_t> resultShape = unpackOp.getDestType().getShape();
  bool satisfyEmptyCond = inputVectorSizes.empty() &&
                          unpackOp.getDestType().hasStaticShape() &&
                          unpackOp.getSourceType().hasStaticShape();
  if (!satisfyEmptyCond &&
      failed(vector::isValidMaskedInputVector(resultShape, inputVectorSizes)))
    return failure();

  return success();
}

static LogicalResult vectorizeLinalgOpPrecondition(
    LinalgOp linalgOp, ArrayRef<int64_t> inputVectorSizes,
    bool vectorizeNDExtract, bool flatten1DDepthwiseConv) {
  // tensor with dimension of 0 cannot be vectorized.
  if (llvm::is_contained(linalgOp.getStaticShape(), 0))
    return failure();
  // Check API contract for input vector sizes.
  if (!inputVectorSizes.empty() &&
      failed(vector::isValidMaskedInputVector(linalgOp.getStaticLoopRanges(),
                                              inputVectorSizes)))
    return failure();

  if (linalgOp.hasDynamicShape() && failed(vectorizeDynamicLinalgOpPrecondition(
                                        linalgOp, flatten1DDepthwiseConv))) {
    LDBG("Dynamically-shaped op failed vectorization pre-conditions\n");
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
    if (llvm::any_of(innerOp.getOperandTypes(), [](Type type) {
          return !VectorType::isValidElementType(type);
        })) {
      return failure();
    }
    if (llvm::any_of(innerOp.getResultTypes(), [](Type type) {
          return !VectorType::isValidElementType(type);
        })) {
      return failure();
    }
  }
  if (isElementwise(linalgOp))
    return success();

  // TODO: isaConvolutionOpInterface that can also infer from generic
  // features. But we will still need stride/dilation attributes that will be
  // annoying to reverse-engineer...
  if (isa<ConvolutionOpInterface>(linalgOp.getOperation()))
    return success();
  // TODO: the common vector shape is equal to the static loop sizes only when
  // all indexing maps are projected permutations. For convs and stencils the
  // logic will need to evolve.
  if (!allIndexingsAreProjectedPermutation(linalgOp)) {
    LDBG("precondition failed: not projected permutations\n");
    return failure();
  }
  if (failed(reductionPreconditions(linalgOp))) {
    LDBG("precondition failed: reduction preconditions\n");
    return failure();
  }
  return success();
}

static LogicalResult
vectorizePackOpPrecondition(tensor::PackOp packOp,
                            ArrayRef<int64_t> inputVectorSizes) {
  auto padValue = packOp.getPaddingValue();
  Attribute cstAttr;
  if (padValue && !matchPattern(padValue, m_Constant(&cstAttr))) {
    LDBG("pad value is not constant: " << packOp << "\n");
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
    LDBG("inner_tiles must be constant: " << packOp << "\n");
    return failure();
  }

  return success();
}

static LogicalResult
vectorizePadOpPrecondition(tensor::PadOp padOp,
                           ArrayRef<int64_t> inputVectorSizes) {
  auto padValue = padOp.getConstantPaddingValue();
  if (!padValue) {
    LDBG("pad value is not constant: " << padOp << "\n");
    return failure();
  }

  ArrayRef<int64_t> resultTensorShape = padOp.getResultType().getShape();
  if (failed(vector::isValidMaskedInputVector(resultTensorShape,
                                              inputVectorSizes)))
    return failure();

  if (llvm::any_of(padOp.getLow(), [](Value v) {
        std::optional<int64_t> res = getConstantIntValue(v);
        return !res.has_value() || res.value() != 0;
      })) {
    LDBG("low pad must all be zero: " << padOp << "\n");
    return failure();
  }

  return success();
}

/// Preconditions for scalable vectors. This is quite restrictive - it models
/// the fact that in practice we would only make selected dimensions scalable.
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

  // Cond 1: There's been no need for scalable vectorisation of
  // non-linalg Ops so far
  if (!linalgOp)
    return failure();

  // Cond 2: There's been no need for more than 2 scalable dims so far
  if (numOfScalableDims > 2)
    return failure();

  // Cond 3: Look at the configuration in `inputScalableVecDims` and verify that
  // it matches one of the supported cases:
  //  1. exactly 1 dim is scalable and that's the _last_ parallel dim
  //  2. exactly 2 dims are scalable and those are the _last two adjacent_
  //     parallel dims
  //  3. exactly 1 reduction dim is scalable and that's the last (innermost) dim
  // The 2nd restriction above means that only Matmul-like Ops are supported
  // when 2 dims are scalable, e.g. :
  //    * iterators = [parallel, parallel, reduction]
  //    * scalable flags = [true, true, false]

  // Find the first scalable flag
  bool seenParalell = false;
  auto iterators = linalgOp.getIteratorTypesArray();
  SmallVector<bool> scalableFlags(inputScalableVecDims);
  while (!scalableFlags.back()) {
    seenParalell |= (iterators.back() == utils::IteratorType::parallel);

    iterators.pop_back();
    scalableFlags.pop_back();
  }

  switch (iterators.back()) {
  case utils::IteratorType::reduction: {
    // Check 3. above is met.
    if (iterators.size() != inputVectorSizes.size()) {
      LDBG("Non-trailing reduction dim requested for scalable "
           "vectorization\n");
      return failure();
    }
    if (isa<linalg::MatmulOp>(op) || isa<linalg::MatmulTransposeAOp>(op)) {
      LDBG("Scalable vectorization of the reduction dim in Matmul-like ops "
           "is not supported\n");
      return failure();
    }
    break;
  }
  case utils::IteratorType::parallel: {
    // Check 1. and 2. above are met.
    if (seenParalell) {
      LDBG("Inner parallel dim not requested for scalable "
           "vectorization\n");
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
      LDBG("Higher dim than the trailing reduction dim requested for scalable "
           "vectorization\n");
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
  return success(isElementwise(linalgOp) || isa<linalg::MatmulOp>(op) ||
                 isa<linalg::MatmulTransposeAOp>(op) ||
                 isa<linalg::DepthwiseConv1DNwcWcOp>(op) ||
                 isa<linalg::MatvecOp>(op) || hasReductionIterator(linalgOp));
}

LogicalResult mlir::linalg::vectorizeOpPrecondition(
    Operation *op, ArrayRef<int64_t> inputVectorSizes,
    ArrayRef<bool> inputScalableVecDims, bool vectorizeNDExtract,
    bool flatten1DDepthwiseConv) {
  if (failed(vectorizeScalableVectorPrecondition(op, inputVectorSizes,
                                                 inputScalableVecDims)))
    return failure();

  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<linalg::LinalgOp>([&](auto linalgOp) {
        return vectorizeLinalgOpPrecondition(linalgOp, inputVectorSizes,
                                             vectorizeNDExtract,
                                             flatten1DDepthwiseConv);
      })
      .Case<tensor::PadOp>([&](auto padOp) {
        return vectorizePadOpPrecondition(padOp, inputVectorSizes);
      })
      .Case<tensor::PackOp>([&](auto packOp) {
        return vectorizePackOpPrecondition(packOp, inputVectorSizes);
      })
      .Case<tensor::UnPackOp>([&](auto unpackOp) {
        return vectorizeUnPackOpPrecondition(unpackOp, inputVectorSizes);
      })
      .Default([](auto) { return failure(); });
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

/// Emit a suitable vector form for an operation. If provided,
/// `inputVectorSizes` are used to vectorize this operation.
/// `inputVectorSizes` must match the rank of the iteration space of the
/// operation and the input vector sizes must be greater than or equal to
/// their counterpart iteration space sizes, if static. `inputVectorShapes`
/// also allows the vectorization of operations with dynamic shapes.
LogicalResult mlir::linalg::vectorize(RewriterBase &rewriter, Operation *op,
                                      ArrayRef<int64_t> inputVectorSizes,
                                      ArrayRef<bool> inputScalableVecDims,
                                      bool vectorizeNDExtract,
                                      bool flatten1DDepthwiseConv) {
  LDBG("Attempting to vectorize:\n" << *op << "\n");
  LDBG("Input vector sizes: ");
  LLVM_DEBUG(llvm::interleaveComma(inputVectorSizes, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");
  LDBG("Input scalable vector dims: ");
  LLVM_DEBUG(llvm::interleaveComma(inputScalableVecDims, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  if (failed(vectorizeOpPrecondition(op, inputVectorSizes, inputScalableVecDims,
                                     vectorizeNDExtract,
                                     flatten1DDepthwiseConv))) {
    LDBG("Vectorization pre-conditions failed\n");
    return failure();
  }

  // Initialize vectorization state.
  VectorizationState state(rewriter);
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (failed(state.initState(rewriter, linalgOp, inputVectorSizes,
                               inputScalableVecDims))) {
      LDBG("Vectorization state couldn't be initialized\n");
      return failure();
    }
  }

  SmallVector<Value> results;
  auto vectorizeResult =
      TypeSwitch<Operation *, LogicalResult>(op)
          .Case<linalg::LinalgOp>([&](auto linalgOp) {
            // TODO: isaConvolutionOpInterface that can also infer from
            // generic features. Will require stride/dilation attributes
            // inference.
            if (isa<ConvolutionOpInterface>(linalgOp.getOperation())) {
              FailureOr<Operation *> convOr = vectorizeConvolution(
                  rewriter, linalgOp, inputVectorSizes, inputScalableVecDims,
                  flatten1DDepthwiseConv);
              if (succeeded(convOr)) {
                llvm::append_range(results, (*convOr)->getResults());
                return success();
              }

              LDBG("Unsupported convolution can't be vectorized.\n");
              return failure();
            }

            LDBG("Vectorize generic by broadcasting to the canonical vector "
                 "shape\n");

            // Pre-process before proceeding.
            convertAffineApply(rewriter, linalgOp);

            // TODO: 'vectorize' takes in a 'RewriterBase' which is up-casted
            // to 'OpBuilder' when it is passed over to some methods like
            // 'vectorizeAsLinalgGeneric'. This is highly problematic: if we
            // erase an op within these methods, the actual rewriter won't be
            // notified and we will end up with read-after-free issues!
            return vectorizeAsLinalgGeneric(rewriter, state, linalgOp, results);
          })
          .Case<tensor::PadOp>([&](auto padOp) {
            return vectorizeAsTensorPadOp(rewriter, padOp, inputVectorSizes,
                                          results);
          })
          .Case<tensor::PackOp>([&](auto packOp) {
            return vectorizeAsTensorPackOp(rewriter, packOp, inputVectorSizes,
                                           results);
          })
          .Case<tensor::UnPackOp>([&](auto unpackOp) {
            return vectorizeAsTensorUnpackOp(rewriter, unpackOp,
                                             inputVectorSizes, results);
          })
          .Default([](auto) { return failure(); });

  if (failed(vectorizeResult)) {
    LDBG("Vectorization failed\n");
    return failure();
  }

  if (!results.empty())
    rewriter.replaceOp(op, results);
  else
    rewriter.eraseOp(op);

  return success();
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
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(srcType.getRank(), zero);

  Value readValue = rewriter.create<vector::TransferReadOp>(
      loc, readType, copyOp.getSource(), indices,
      rewriter.getMultiDimIdentityMap(srcType.getRank()));
  if (cast<VectorType>(readValue.getType()).getRank() == 0) {
    readValue = rewriter.create<vector::ExtractElementOp>(loc, readValue);
    readValue = rewriter.create<vector::BroadcastOp>(loc, writeType, readValue);
  }
  Operation *writeValue = rewriter.create<vector::TransferWriteOp>(
      loc, readValue, copyOp.getTarget(), indices,
      rewriter.getMultiDimIdentityMap(srcType.getRank()));
  rewriter.replaceOp(copyOp, writeValue->getResults());
  return success();
}

//----------------------------------------------------------------------------//
// Misc. vectorization patterns.
//----------------------------------------------------------------------------//

/// Helper function that retrieves the value of an IntegerAttr.
static int64_t getIntFromAttr(Attribute attr) {
  return cast<IntegerAttr>(attr).getInt();
}

/// Given an ArrayRef of OpFoldResults, return a vector of Values.
/// IntegerAttrs are converted to ConstantIndexOps. Other attribute types are
/// not supported.
static SmallVector<Value> ofrToIndexValues(RewriterBase &rewriter, Location loc,
                                           ArrayRef<OpFoldResult> ofrs) {
  SmallVector<Value> result;
  for (auto o : ofrs) {
    if (auto val = llvm::dyn_cast_if_present<Value>(o)) {
      result.push_back(val);
    } else {
      result.push_back(rewriter.create<arith::ConstantIndexOp>(
          loc, getIntFromAttr(o.template get<Attribute>())));
    }
  }
  return result;
}

/// Rewrite a tensor::PadOp into a sequence of EmptyOp, FillOp and
/// InsertSliceOp. For now, only constant padding values are supported.
/// If there is enough static type information, TransferReadOps and
/// TransferWriteOps may be generated instead of InsertSliceOps.
struct GenericPadOpVectorizationPattern : public GeneralizePadOpPattern {
  GenericPadOpVectorizationPattern(MLIRContext *context,
                                   PatternBenefit benefit = 1)
      : GeneralizePadOpPattern(context, tryVectorizeCopy, benefit) {}
  /// Vectorize the copying of a tensor::PadOp's source. This is possible if
  /// each dimension size is statically know in the source type or the result
  /// type (or both).
  static LogicalResult tryVectorizeCopy(RewriterBase &rewriter,
                                        tensor::PadOp padOp, Value dest) {
    auto sourceType = padOp.getSourceType();
    auto resultType = padOp.getResultType();
    if (!VectorType::isValidElementType(sourceType.getElementType()))
      return failure();

    // Copy cannot be vectorized if pad value is non-constant and source shape
    // is dynamic. In case of a dynamic source shape, padding must be appended
    // by TransferReadOp, but TransferReadOp supports only constant padding.
    auto padValue = padOp.getConstantPaddingValue();
    if (!padValue) {
      if (!sourceType.hasStaticShape())
        return failure();
      // Create dummy padding value.
      auto elemType = sourceType.getElementType();
      padValue = rewriter.create<arith::ConstantOp>(
          padOp.getLoc(), elemType, rewriter.getZeroAttr(elemType));
    }

    SmallVector<int64_t> vecShape;
    SmallVector<bool> readInBounds;
    SmallVector<bool> writeInBounds;
    for (unsigned i = 0; i < sourceType.getRank(); ++i) {
      if (!sourceType.isDynamicDim(i)) {
        vecShape.push_back(sourceType.getDimSize(i));
        // Source shape is statically known: Neither read nor write are
        // out-of- bounds.
        readInBounds.push_back(true);
        writeInBounds.push_back(true);
      } else if (!resultType.isDynamicDim(i)) {
        // Source shape is not statically known, but result shape is.
        // Vectorize with size of result shape. This may be larger than the
        // source size.
        vecShape.push_back(resultType.getDimSize(i));
        // Read may be out-of-bounds because the result size could be larger
        // than the source size.
        readInBounds.push_back(false);
        // Write is out-of-bounds if low padding > 0.
        writeInBounds.push_back(
            getConstantIntValue(padOp.getMixedLowPad()[i]) ==
            static_cast<int64_t>(0));
      } else {
        // Neither source nor result dim of padOp is static. Cannot vectorize
        // the copy.
        return failure();
      }
    }
    auto vecType = VectorType::get(vecShape, sourceType.getElementType());

    // Generate TransferReadOp.
    SmallVector<Value> readIndices(
        vecType.getRank(),
        rewriter.create<arith::ConstantIndexOp>(padOp.getLoc(), 0));
    auto read = rewriter.create<vector::TransferReadOp>(
        padOp.getLoc(), vecType, padOp.getSource(), readIndices, padValue,
        ArrayRef<bool>{readInBounds});

    // If `dest` is a FillOp and the TransferWriteOp would overwrite the
    // entire tensor, write directly to the FillOp's operand.
    if (llvm::equal(vecShape, resultType.getShape()) &&
        llvm::all_of(writeInBounds, [](bool b) { return b; }))
      if (auto fill = dest.getDefiningOp<FillOp>())
        dest = fill.output();

    // Generate TransferWriteOp.
    auto writeIndices =
        ofrToIndexValues(rewriter, padOp.getLoc(), padOp.getMixedLowPad());
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        padOp, read, dest, writeIndices, ArrayRef<bool>{writeInBounds});

    return success();
  }
};

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
      xferOp.getSourceMutable().assign(padOp.getSource());
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
        vecRank, rewriter.create<arith::ConstantIndexOp>(padOp.getLoc(), 0));
    auto read = rewriter.create<vector::TransferReadOp>(
        padOp.getLoc(), vecType, padOp.getSource(), readIndices, padValue);

    // Generate TransferWriteOp: Write to InsertSliceOp's dest tensor at
    // specified offsets. Write is fully in-bounds because a InsertSliceOp's
    // source must fit into the destination at the specified offsets.
    auto writeIndices =
        ofrToIndexValues(rewriter, padOp.getLoc(), insertOp.getMixedOffsets());
    SmallVector<bool> inBounds(vecRank, true);
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        insertOp, read, insertOp.getDest(), writeIndices,
        ArrayRef<bool>{inBounds});

    return success();
  }
};

void mlir::linalg::populatePadOpVectorizationPatterns(
    RewritePatternSet &patterns, PatternBenefit baseBenefit) {
  patterns.add<GenericPadOpVectorizationPattern>(patterns.getContext(),
                                                 baseBenefit);
  // Try these specialized patterns first before resorting to the generic one.
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
    LDBG("interleavedUses precondition failed, firstOp: "
         << *firstOp << ", second op: " << *secondOp << "\n");
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
      LDBG(" found interleaved op " << *owner << ", firstOp: " << *firstOp
                                    << ", second op: " << *secondOp << "\n");
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
  Value viewOrAlloc = xferOp.getSource();
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
  Value res = rewriter.create<vector::TransferReadOp>(
      xferOp.getLoc(), vectorType, in, xferOp.getIndices(),
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
  Value viewOrAlloc = xferOp.getSource();
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
  rewriter.create<vector::TransferWriteOp>(
      xferOp.getLoc(), vector, out, xferOp.getIndices(),
      xferOp.getPermutationMapAttr(), xferOp.getMask(),
      rewriter.getBoolArrayAttr(
          SmallVector<bool>(vector.getType().getRank(), false)));

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

namespace {
bool isCastOfBlockArgument(Operation *op) {
  return isa<CastOpInterface>(op) && op->getNumOperands() == 1 &&
         isa<BlockArgument>(op->getOperand(0));
}

bool isSupportedPoolKind(vector::CombiningKind kind) {
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
  Conv1DGenerator(RewriterBase &rewriter, LinalgOp linalgOp, int strideW,
                  int dilationW)
      : StructuredGenerator<LinalgOp, utils::IteratorType>(rewriter, linalgOp),
        strideW(strideW), dilationW(dilationW) {
    // Determine whether `linalgOp` can be generated with this generator
    if (linalgOp.getNumDpsInputs() != 2 || linalgOp.getNumDpsInits() != 1)
      return;
    lhsShaped = linalgOp.getDpsInputOperand(0)->get();
    rhsShaped = linalgOp.getDpsInputOperand(1)->get();
    resShaped = linalgOp.getDpsInitOperand(0)->get();
    lhsShapedType = dyn_cast<ShapedType>(lhsShaped.getType());
    rhsShapedType = dyn_cast<ShapedType>(rhsShaped.getType());
    resShapedType = dyn_cast<ShapedType>(resShaped.getType());
    if (!lhsShapedType || !rhsShapedType || !resShapedType)
      return;
    // (LHS has dimension NCW/NWC and RES has dimension NFW/NCW/NWF/NWC) OR
    // (non-channeled convolution -> LHS and RHS both have single dimensions).
    if ((lhsShapedType.getRank() != 3 || resShapedType.getRank() != 3) &&
        (lhsShapedType.getRank() != 1 || resShapedType.getRank() != 1))
      return;

    Operation *reduceOp = matchLinalgReduction(linalgOp.getDpsInitOperand(0));
    if (!reduceOp)
      return;
    redOp = reduceOp->getName().getIdentifier();

    if (!setOperKind(reduceOp))
      return;
    auto maybeKind = getCombinerOpKind(reduceOp);
    // Typically convolution will have a `Add` CombiningKind but for i1 type it
    // can get strength reduced to `OR` which is also supported. This strength
    // reduction logic is in `buildBinaryFn` helper in the Linalg dialect.
    if (!maybeKind || ((*maybeKind != vector::CombiningKind::ADD &&
                        *maybeKind != vector::CombiningKind::OR) &&
                       (oper != Pool || !isSupportedPoolKind(*maybeKind)))) {
      return;
    }
    reductionKind = maybeKind.value();

    auto rhsRank = rhsShapedType.getRank();
    switch (oper) {
    case Conv:
      if (rhsRank != 1 && rhsRank != 2 && rhsRank != 3)
        return;
      break;
    case Pool:
      if (rhsRank != 1)
        return;
      break;
    }
    // The op is now known to be valid.
    valid = true;
  }

  /// Generate a vector implementation for:
  /// ```
  ///   Op def: (     w,     kw  )
  ///    Iters: ({Par(), Red()})
  ///   Layout: {{w + kw}, {kw}, {w}}
  /// ```
  /// kw is always unrolled.
  ///
  /// or
  ///
  /// ```
  ///   Op def: (     n,     w,     c,    kw,    f  )
  ///    Iters: ({Par(), Par(), Par(), Red(), Red()})
  ///   Layout: {{n, strideW * w + dilationW * kw, c}, {kw, c, f}, {n, w, f}}
  /// ```
  /// kw is always unrolled.
  /// TODO: w (resp. kw) is unrolled when the strideW ( resp. dilationW) is
  /// > 1.
  FailureOr<Operation *> conv(Conv1DOpOrder conv1DOpOrder) {
    if (!valid)
      return rewriter.notifyMatchFailure(op, "unvectorizable 1-D conv/pool");

    int64_t nSize, wSize, cSize, kwSize, fSize;
    SmallVector<int64_t, 3> lhsShape, rhsShape, resShape;
    bool isSingleChanneled = (conv1DOpOrder == Conv1DOpOrder::W);
    switch (conv1DOpOrder) {
    case Conv1DOpOrder::W:
      // Initialize unused dimensions
      nSize = fSize = cSize = 0;
      // out{W}
      bindShapeDims(resShapedType, wSize);
      // kernel{kw}
      bindShapeDims(rhsShapedType, kwSize);
      lhsShape = {// iw = ow + kw - 1
                  //   (i.e. 16 convolved with 3 -> 14)
                  (wSize + kwSize - 1)};
      rhsShape = {kwSize};
      resShape = {wSize};
      break;
    case Conv1DOpOrder::Nwc:
      // out{n, w, f}
      bindShapeDims(resShapedType, nSize, wSize, fSize);
      switch (oper) {
      case Conv:
        // kernel{kw, c, f}
        bindShapeDims(rhsShapedType, kwSize, cSize);
        break;
      case Pool:
        // kernel{kw}
        bindShapeDims(rhsShapedType, kwSize);
        cSize = fSize;
        break;
      }
      lhsShape = {nSize,
                  // iw = ow * sw + kw *  dw - 1
                  //   (i.e. 16 convolved with 3 (@stride 1 dilation 1) -> 14)
                  // Perform the proper inclusive -> exclusive -> inclusive.
                  ((wSize - 1) * strideW + 1) + ((kwSize - 1) * dilationW + 1) -
                      1,
                  cSize};
      switch (oper) {
      case Conv:
        rhsShape = {kwSize, cSize, fSize};
        break;
      case Pool:
        rhsShape = {kwSize};
        break;
      }
      resShape = {nSize, wSize, fSize};
      break;
    case Conv1DOpOrder::Ncw:
      // out{n, f, w}
      bindShapeDims(resShapedType, nSize, fSize, wSize);
      switch (oper) {
      case Conv:
        // kernel{f, c, kw}
        bindShapeDims(rhsShapedType, fSize, cSize, kwSize);
        break;
      case Pool:
        // kernel{kw}
        bindShapeDims(rhsShapedType, kwSize);
        cSize = fSize;
        break;
      }
      lhsShape = {nSize, cSize,
                  // iw = ow * sw + kw *  dw - 1
                  //   (i.e. 16 convolved with 3 (@stride 1 dilation 1) -> 14)
                  // Perform the proper inclusive -> exclusive -> inclusive.
                  ((wSize - 1) * strideW + 1) + ((kwSize - 1) * dilationW + 1) -
                      1};
      switch (oper) {
      case Conv:
        rhsShape = {fSize, cSize, kwSize};
        break;
      case Pool:
        rhsShape = {kwSize};
        break;
      }
      resShape = {nSize, fSize, wSize};
      break;
    }

    vector::TransferWriteOp write;
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // w is unrolled (i.e. wSizeStep == 1) iff strideW > 1.
    // When strideW == 1, we can batch the contiguous loads and avoid
    // unrolling
    int64_t wSizeStep = strideW == 1 ? wSize : 1;

    Type lhsEltType = lhsShapedType.getElementType();
    Type rhsEltType = rhsShapedType.getElementType();
    Type resEltType = resShapedType.getElementType();
    auto lhsType = VectorType::get(lhsShape, lhsEltType);
    auto rhsType = VectorType::get(rhsShape, rhsEltType);
    auto resType = VectorType::get(resShape, resEltType);
    // Zero padding with the corresponding dimensions for lhs, rhs and res.
    SmallVector<Value> lhsPadding(lhsShape.size(), zero);
    SmallVector<Value> rhsPadding(rhsShape.size(), zero);
    SmallVector<Value> resPadding(resShape.size(), zero);

    // Read the whole lhs, rhs and res in one shot (with zero padding).
    Value lhs = rewriter.create<vector::TransferReadOp>(loc, lhsType, lhsShaped,
                                                        lhsPadding);
    // This is needed only for Conv.
    Value rhs = nullptr;
    if (oper == Conv)
      rhs = rewriter.create<vector::TransferReadOp>(loc, rhsType, rhsShaped,
                                                    rhsPadding);
    Value res = rewriter.create<vector::TransferReadOp>(loc, resType, resShaped,
                                                        resPadding);

    // The base vectorization case for channeled convolution is input:
    // {n,w,c}, weight: {kw,c,f}, output: {n,w,f}. To reuse the base pattern
    // vectorization case, we do pre transpose on input, weight, and output.
    switch (conv1DOpOrder) {
    case Conv1DOpOrder::W:
    case Conv1DOpOrder::Nwc:
      // Base case, so no transposes necessary.
      break;
    case Conv1DOpOrder::Ncw: {
      // To match base vectorization case, we pre-transpose current case.
      // ncw -> nwc
      static constexpr std::array<int64_t, 3> permLhs = {0, 2, 1};
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, permLhs);
      // fcw -> wcf
      static constexpr std::array<int64_t, 3> permRhs = {2, 1, 0};

      // This is needed only for Conv.
      if (oper == Conv)
        rhs = rewriter.create<vector::TransposeOp>(loc, rhs, permRhs);
      // nfw -> nwf
      static constexpr std::array<int64_t, 3> permRes = {0, 2, 1};
      res = rewriter.create<vector::TransposeOp>(loc, res, permRes);
      break;
    }
    }

    //===------------------------------------------------------------------===//
    // Begin vector-only rewrite part
    //===------------------------------------------------------------------===//
    // Unroll along kw and read slices of lhs and rhs.
    SmallVector<Value> lhsVals, rhsVals, resVals;
    lhsVals = extractConvInputSlices(rewriter, loc, lhs, nSize, wSize, cSize,
                                     kwSize, strideW, dilationW, wSizeStep,
                                     isSingleChanneled);
    // Do not do for pooling.
    if (oper == Conv)
      rhsVals = extractConvFilterSlices(rewriter, loc, rhs, kwSize);
    resVals = extractConvResultSlices(rewriter, loc, res, nSize, wSize, fSize,
                                      wSizeStep, isSingleChanneled);

    auto linearIndex = [&](int64_t kw, int64_t w) {
      return kw * (wSize / wSizeStep) + w;
    };

    // Compute contraction: O{n, w, f} += I{n, sw * w + dw * kw, c} * F{c, f}
    // or perform outerproduct for non-channeled convolution or perform simple
    // arith operation for pooling
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      for (int64_t w = 0; w < wSize; w += wSizeStep) {
        switch (oper) {
        case Conv:
          if (isSingleChanneled) {
            resVals[w] = conv1dSliceAsOuterProduct(rewriter, loc,
                                                   lhsVals[linearIndex(kw, w)],
                                                   rhsVals[kw], resVals[w]);
          } else {
            resVals[w] = conv1dSliceAsContraction(rewriter, loc,
                                                  lhsVals[linearIndex(kw, w)],
                                                  rhsVals[kw], resVals[w]);
          }
          break;
        case Pool:
          resVals[w] = pool1dSlice(rewriter, loc, lhsVals[linearIndex(kw, w)],
                                   resVals[w]);
          break;
        }
      }
    }

    res = insertConvResultSlices(rewriter, loc, res, wSize, wSizeStep, resVals,
                                 isSingleChanneled);
    //===------------------------------------------------------------------===//
    // End vector-only rewrite part
    //===------------------------------------------------------------------===//

    // The base vectorization case for channeled convolution is output:
    // {n,w,f} To reuse the result from base pattern vectorization case, we
    // post transpose the base case result.
    switch (conv1DOpOrder) {
    case Conv1DOpOrder::W:
    case Conv1DOpOrder::Nwc:
      // Base case, so no transposes necessary.
      break;
    case Conv1DOpOrder::Ncw: {
      // nwf -> nfw
      static constexpr std::array<int64_t, 3> perm = {0, 2, 1};
      res = rewriter.create<vector::TransposeOp>(loc, res, perm);
      break;
    }
    }

    return rewriter
        .create<vector::TransferWriteOp>(loc, res, resShaped, resPadding)
        .getOperation();
  }

  // Take a value and widen to have the same element type as `ty`.
  Value promote(RewriterBase &rewriter, Location loc, Value val, Type ty) {
    const Type srcElementType = getElementTypeOrSelf(val.getType());
    const Type dstElementType = getElementTypeOrSelf(ty);
    assert(isa<IntegerType>(dstElementType) || isa<FloatType>(dstElementType));
    if (srcElementType == dstElementType)
      return val;

    const int64_t srcWidth = srcElementType.getIntOrFloatBitWidth();
    const int64_t dstWidth = dstElementType.getIntOrFloatBitWidth();
    const Type dstType =
        cast<ShapedType>(val.getType()).cloneWith(std::nullopt, dstElementType);

    if (isa<IntegerType>(srcElementType) && isa<FloatType>(dstElementType)) {
      return rewriter.create<arith::SIToFPOp>(loc, dstType, val);
    }

    if (isa<FloatType>(srcElementType) && isa<FloatType>(dstElementType) &&
        srcWidth < dstWidth)
      return rewriter.create<arith::ExtFOp>(loc, dstType, val);

    if (isa<IntegerType>(srcElementType) && isa<IntegerType>(dstElementType) &&
        srcWidth < dstWidth)
      return rewriter.create<arith::ExtSIOp>(loc, dstType, val);

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
    auto contrationOp = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, res,
        /*indexingMaps=*/MapList{{n, w, c}, {c, f}, {n, w, f}},
        /*iteratorTypes=*/ArrayRef<vector::IteratorType>{par, par, par, red});
    contrationOp.setKind(reductionKind);
    return contrationOp;
  }

  // Create an outerproduct: lhs{w} * rhs{1} -> res{w} for single channel
  // convolution.
  Value conv1dSliceAsOuterProduct(RewriterBase &rewriter, Location loc,
                                  Value lhs, Value rhs, Value res) {
    return rewriter.create<vector::OuterProductOp>(
        loc, res.getType(), lhs, rhs, res, vector::CombiningKind::ADD);
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
    if (!valid)
      return rewriter.notifyMatchFailure(op, "unvectorizable depthwise conv");

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

    vector::TransferWriteOp write;
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

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
          rewriter.create<vector::CreateMaskOp>(loc, maskType, mixedDims);

      return mlir::vector::maskOperation(rewriter, opToMask, maskOp);
    };

    // Read lhs slice of size {n, w * strideW + kw * dilationW, c} @ [0, 0,
    // 0].
    Value lhs = rewriter.create<vector::TransferReadOp>(
        loc, lhsType, lhsShaped, ValueRange{zero, zero, zero});
    auto maybeMaskedLhs = maybeMaskXferOp(
        lhsType.getShape(), lhsType.getScalableDims(), lhs.getDefiningOp());

    // Read rhs slice of size {kw, c} @ [0, 0].
    Value rhs = rewriter.create<vector::TransferReadOp>(loc, rhsType, rhsShaped,
                                                        ValueRange{zero, zero});
    auto maybeMaskedRhs = maybeMaskXferOp(
        rhsType.getShape(), rhsType.getScalableDims(), rhs.getDefiningOp());

    // Read res slice of size {n, w, c} @ [0, 0, 0].
    Value res = rewriter.create<vector::TransferReadOp>(
        loc, resType, resShaped, ValueRange{zero, zero, zero});
    auto maybeMaskedRes = maybeMaskXferOp(
        resType.getShape(), resType.getScalableDims(), res.getDefiningOp());

    //===------------------------------------------------------------------===//
    // Begin vector-only rewrite part
    //===------------------------------------------------------------------===//
    // Unroll along kw and read slices of lhs and rhs.
    SmallVector<Value> lhsVals, rhsVals, resVals;
    auto inOutSliceSizes = SmallVector<int64_t>{nSize, wSizeStep, cSize};
    auto inOutStrides = SmallVector<int64_t>{1, 1, 1};

    // Extract lhs slice of size {n, wSizeStep, c}
    //   @ [0, sw * w + dw * kw, 0].
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      for (int64_t w = 0; w < wSize; w += wSizeStep) {
        lhsVals.push_back(rewriter.create<vector::ExtractStridedSliceOp>(
            loc, maybeMaskedLhs->getResult(0),
            /*offsets=*/ArrayRef<int64_t>{0, w * strideW + kw * dilationW, 0},
            inOutSliceSizes, inOutStrides));
      }
    }
    // Extract rhs slice of size {c} @ [kw].
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      rhsVals.push_back(rewriter.create<vector::ExtractOp>(
          loc, maybeMaskedRhs->getResult(0),
          /*offsets=*/ArrayRef<int64_t>{kw}));
    }
    // Extract res slice: {n, wSizeStep, c} @ [0, w, 0].
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      resVals.push_back(rewriter.create<vector::ExtractStridedSliceOp>(
          loc, maybeMaskedRes->getResult(0),
          /*offsets=*/ArrayRef<int64_t>{0, w, 0}, inOutSliceSizes,
          inOutStrides));
    }

    auto linearIndex = [&](int64_t kw, int64_t w) {
      return kw * (wSize / wSizeStep) + w;
    };

    // Note - the scalable flags are ignored as flattening combined with
    // scalable vectorization is not supported.
    auto inOutFlattenSliceSizes =
        SmallVector<int64_t>{nSize, wSizeStep * cSize};
    auto lhsTypeAfterFlattening =
        VectorType::get(inOutFlattenSliceSizes, lhsEltType);
    auto resTypeAfterFlattening =
        VectorType::get(inOutFlattenSliceSizes, resEltType);

    // Compute contraction: O{n, w, c} += I{n, sw * w + dw * kw, c} * F{c}
    for (int64_t kw = 0; kw < kwSize; ++kw) {
      for (int64_t w = 0; w < wSize; w += wSizeStep) {
        Value lhsVal = lhsVals[linearIndex(kw, w)];
        Value resVal = resVals[w];
        if (flatten) {
          // Flatten the input and output vectors (collapse the channel
          // dimension)
          lhsVal = rewriter.create<vector::ShapeCastOp>(
              loc, lhsTypeAfterFlattening, lhsVals[linearIndex(kw, w)]);
          resVal = rewriter.create<vector::ShapeCastOp>(
              loc, resTypeAfterFlattening, resVals[w]);
        }
        resVals[w] = depthwiseConv1dSliceAsMulAcc(rewriter, loc, lhsVal,
                                                  rhsVals[kw], resVal, flatten);
        if (flatten) {
          // Un-flatten the output vector (restore the channel dimension)
          resVals[w] = rewriter.create<vector::ShapeCastOp>(
              loc, VectorType::get(inOutSliceSizes, resEltType), resVals[w]);
        }
      }
    }

    // Its possible we failed to create the Fma.
    if (!llvm::all_of(resVals, [](Value v) { return v; })) {
      // Manually revert (in reverse order) to avoid leaving a bad IR state.
      for (auto &collection :
           {resVals, rhsVals, lhsVals, {res, rhs, lhs, zero}})
        for (Value v : collection)
          rewriter.eraseOp(v.getDefiningOp());
      return rewriter.notifyMatchFailure(op, "failed to create FMA");
    }

    // Write back res slice: {n, wSizeStep, c} @ [0, w, 0].
    // This does not depend on kw.
    for (int64_t w = 0; w < wSize; w += wSizeStep) {
      maybeMaskedRes = rewriter.create<vector::InsertStridedSliceOp>(
          loc, resVals[w], maybeMaskedRes->getResult(0),
          /*offsets=*/ArrayRef<int64_t>{0, w, 0},
          /*strides=*/ArrayRef<int64_t>{1, 1, 1});
    }
    //===------------------------------------------------------------------===//
    // End vector-only rewrite part
    //===------------------------------------------------------------------===//

    // Write back res slice of size {n, w, c} @ [0, 0, 0].
    Operation *resOut = rewriter.create<vector::TransferWriteOp>(
        loc, maybeMaskedRes->getResult(0), resShaped,
        ValueRange{zero, zero, zero});
    return maybeMaskXferOp(resType.getShape(), resType.getScalableDims(),
                           resOut);
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

      rhs = rewriter.create<vector::ShuffleOp>(loc, rhs, rhs, indices);
    }
    // Broadcast the filter to match the output vector
    rhs = rewriter.create<vector::BroadcastOp>(
        loc, resTy.clone(rhsTy.getElementType()), rhs);

    rhs = promote(rewriter, loc, rhs, resTy);

    if (!lhs || !rhs)
      return nullptr;

    if (isa<FloatType>(resTy.getElementType()))
      return rewriter.create<vector::FMAOp>(loc, lhs, rhs, res);

    auto mul = rewriter.create<arith::MulIOp>(loc, lhs, rhs);
    return rewriter.create<arith::AddIOp>(loc, mul, res);
  }

  /// Entry point for non-channeled convolution:
  ///   {{w + kw}, {kw}, {w}}
  FailureOr<Operation *> generateNonChanneledConv() {
    AffineExpr w, kw;
    bindDims(ctx, w, kw);
    if (!iters({Par(), Red()}))
      return rewriter.notifyMatchFailure(op,
                                         "failed to match conv::W 1-par 1-red");

    // No transposition needed.
    if (layout({/*lhsIndex*/ {w + kw},
                /*rhsIndex*/ {kw},
                /*resIndex*/ {w}}))
      return conv(Conv1DOpOrder::W);

    return rewriter.notifyMatchFailure(op, "not a conv::W layout");
  }

  /// Entry point that transposes into the common form:
  ///   {{n, strideW * w + dilationW * kw, c}, {kw, c, f}, {n, w, f}}
  FailureOr<Operation *> generateNwcConv() {
    AffineExpr n, w, f, kw, c;
    bindDims(ctx, n, w, f, kw, c);
    if (!iters({Par(), Par(), Par(), Red(), Red()}))
      return rewriter.notifyMatchFailure(
          op, "failed to match conv::Nwc 3-par 2-red");

    // No transposition needed.
    if (layout({/*lhsIndex*/ {n, strideW * w + dilationW * kw, c},
                /*rhsIndex*/ {kw, c, f},
                /*resIndex*/ {n, w, f}}))
      return conv(Conv1DOpOrder::Nwc);

    return rewriter.notifyMatchFailure(op, "not a conv::Nwc layout");
  }

  /// Entry point that transposes into the common form:
  ///   {{n, c, strideW * w + dilationW * kw}, {f, c, kw}, {n, f, w}}
  FailureOr<Operation *> generateNcwConv() {
    AffineExpr n, w, f, kw, c;
    bindDims(ctx, n, f, w, c, kw);
    if (!iters({Par(), Par(), Par(), Red(), Red()}))
      return rewriter.notifyMatchFailure(
          op, "failed to match conv::Ncw 3-par 2-red");

    if (layout({/*lhsIndex*/ {n, c, strideW * w + dilationW * kw},
                /*rhsIndex*/ {f, c, kw},
                /*resIndex*/ {n, f, w}}))
      return conv(Conv1DOpOrder::Ncw);

    return rewriter.notifyMatchFailure(op, "not a conv::Ncw layout");
  }

  /// Entry point that transposes into the common form:
  ///   {{n, strideW * w + dilationW * kw, c}, {kw}, {n, w, c}} for pooling
  FailureOr<Operation *> generateNwcPooling() {
    AffineExpr n, w, c, kw;
    bindDims(ctx, n, w, c, kw);
    if (!iters({Par(), Par(), Par(), Red()}))
      return rewriter.notifyMatchFailure(op,
                                         "failed to match pooling 3-par 1-red");

    // No transposition needed.
    if (layout({/*lhsIndex*/ {n, strideW * w + dilationW * kw, c},
                /*rhsIndex*/ {kw},
                /*resIndex*/ {n, w, c}}))
      return conv(Conv1DOpOrder::Nwc);

    return rewriter.notifyMatchFailure(op, "not a pooling::Nwc layout");
  }

  /// Entry point that transposes into the common form:
  ///   {{n, c, strideW * w + dilationW * kw}, {kw}, {n, c, w}} for pooling
  FailureOr<Operation *> generateNcwPooling() {
    AffineExpr n, w, c, kw;
    bindDims(ctx, n, c, w, kw);
    if (!iters({Par(), Par(), Par(), Red()}))
      return rewriter.notifyMatchFailure(op,
                                         "failed to match pooling 3-par 1-red");

    if (layout({/*lhsIndex*/ {n, c, strideW * w + dilationW * kw},
                /*rhsIndex*/ {kw},
                /*resIndex*/ {n, c, w}}))
      return conv(Conv1DOpOrder::Ncw);

    return rewriter.notifyMatchFailure(op, "not a pooling::Ncw layout");
  }

  /// Entry point that transposes into the common form:
  ///   {{n, strideW * w + dilationW * kw, c}, {kw, c}, {n, w, c}}
  FailureOr<Operation *> generateDilatedConv(uint64_t vecChDimSize = 0,
                                             bool vecChDimScalableFlag = false,
                                             bool flatten = false) {
    AffineExpr n, w, c, kw;
    bindDims(ctx, n, w, c, kw);
    if (!iters({Par(), Par(), Par(), Red()}))
      return rewriter.notifyMatchFailure(
          op, "failed to match depthwise::Nwc conv 3-par 1-red");

    // No transposition needed.
    if (layout({/*lhsIndex*/ {n, strideW * w + dilationW * kw, c},
                /*rhsIndex*/ {kw, c},
                /*resIndex*/ {n, w, c}}))
      return depthwiseConv(vecChDimSize, vecChDimScalableFlag, flatten);

    return rewriter.notifyMatchFailure(op, "not a depthwise::Nwc layout");
  }

private:
  enum OperKind { Conv, Pool };
  bool valid = false;
  OperKind oper = Conv;
  StringAttr redOp;
  StringAttr poolExtOp;
  bool isPoolExt = false;
  int strideW, dilationW;
  Value lhsShaped, rhsShaped, resShaped;
  ShapedType lhsShapedType, rhsShapedType, resShapedType;
  vector::CombiningKind reductionKind;

  // Sets oper, poolExtOp and isPoolExt for valid conv/pooling ops.
  // Returns true iff it is a valid conv/pooling op.
  // If (region has 2 ops (reduction + yield) or 3 ops (extension + reduction
  // + yield) and rhs is not used) then it is the body of a pooling
  // If conv, check for single `mul` predecessor. The `mul` operands must be
  // block arguments or extension of block arguments.
  // Otherwise, check for one or zero `ext` predecessor. The `ext` operands
  // must be block arguments or extension of block arguments.
  bool setOperKind(Operation *reduceOp) {
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
      Operation *feedOp = (*feedValIt).getDefiningOp();
      if (isCastOfBlockArgument(feedOp)) {
        oper = Pool;
        isPoolExt = true;
        poolExtOp = feedOp->getName().getIdentifier();
      } else if (!((isa<arith::MulIOp, arith::MulFOp>(feedOp) ||
                    (isa<arith::AndIOp>(feedOp) &&
                     feedOp->getResultTypes()[0].isInteger(1))) &&
                   llvm::all_of(feedOp->getOperands(), [](Value v) {
                     if (isa<BlockArgument>(v))
                       return true;
                     if (Operation *op = v.getDefiningOp())
                       return isCastOfBlockArgument(op);
                     return false;
                   }))) {
        return false;
      }
      return true;
    }
    case 2:
      // Must be pooling
      oper = Pool;
      isPoolExt = false;
      return true;
    default:
      return false;
    }
  }
};
} // namespace

/// Helper function to vectorize a LinalgOp with convolution semantics.
// TODO: extend the generic vectorization to support windows and drop this.
static FailureOr<Operation *> vectorizeConvolution(
    RewriterBase &rewriter, LinalgOp op, ArrayRef<int64_t> inputVecSizes,
    ArrayRef<bool> inputScalableVecDims, bool flatten1DDepthwiseConv) {
  // The ConvolutionOpInterface gives us guarantees of existence for
  // strides/dilations. However, we do not need to rely on those, we can
  // simply use them if present, otherwise use the default and let the generic
  // conv. matcher in the ConvGenerator succeed or fail.
  auto strides = op->getAttrOfType<DenseIntElementsAttr>("strides");
  auto dilations = op->getAttrOfType<DenseIntElementsAttr>("dilations");
  auto stride = strides ? *strides.getValues<uint64_t>().begin() : 1;
  auto dilation = dilations ? *dilations.getValues<uint64_t>().begin() : 1;
  Conv1DGenerator e(rewriter, op, stride, dilation);
  auto res = e.generateNonChanneledConv();
  if (succeeded(res))
    return res;
  res = e.generateNwcConv();
  if (succeeded(res))
    return res;
  res = e.generateNcwConv();
  if (succeeded(res))
    return res;
  res = e.generateNwcPooling();
  if (succeeded(res))
    return res;
  res = e.generateNcwPooling();
  if (succeeded(res))
    return res;

  // Only depthwise 1D NWC convs are left - these can be vectorized using masks
  // and scalable vectors. Note that ATM the only dim that can be dynamic (i.e.
  // masked/scalable) is the channel dim (i.e. the trailing dim).
  uint64_t vecChDimSize = ShapedType::kDynamic;
  bool vecChDimScalableFlag = false;
  if (!inputVecSizes.empty()) {
    // Only use the input vector size corresponding to the channel dim. Other
    // vector dims will be inferred from the Ops.
    assert((isa<linalg::DepthwiseConv1DNwcWcOp>(*op) ||
            isa<linalg::DepthwiseConv1DNcwCwOp>(*op)) &&
           "Not a 1D depthwise conv!");
    size_t chDimIdx =
        TypeSwitch<Operation *, size_t>(op)
            .Case<linalg::DepthwiseConv1DNwcWcOp>([](auto conv) { return 2; })
            .Case<linalg::DepthwiseConv1DNcwCwOp>([](auto conv) { return 1; });

    vecChDimSize = inputVecSizes[chDimIdx];
    vecChDimScalableFlag = inputScalableVecDims[chDimIdx];
  }
  return e.generateDilatedConv(vecChDimSize, vecChDimScalableFlag,
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
