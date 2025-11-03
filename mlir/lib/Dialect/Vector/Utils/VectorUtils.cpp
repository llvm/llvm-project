//===- VectorUtils.cpp - MLIR Utilities for VectorOps   ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility methods for working with the Vector dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Utils/VectorUtils.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"

#define DEBUG_TYPE "vector-utils"

using namespace mlir;

/// Helper function that creates a memref::DimOp or tensor::DimOp depending on
/// the type of `source`.
Value mlir::vector::createOrFoldDimOp(OpBuilder &b, Location loc, Value source,
                                      int64_t dim) {
  if (isa<UnrankedMemRefType, MemRefType>(source.getType()))
    return b.createOrFold<memref::DimOp>(loc, source, dim);
  if (isa<UnrankedTensorType, RankedTensorType>(source.getType()))
    return b.createOrFold<tensor::DimOp>(loc, source, dim);
  llvm_unreachable("Expected MemRefType or TensorType");
}

/// Given the n-D transpose pattern 'transp', return true if 'dim0' and 'dim1'
/// should be transposed with each other within the context of their 2D
/// transposition slice.
///
/// Example 1: dim0 = 0, dim1 = 2, transp = [2, 1, 0]
///   Return true: dim0 and dim1 are transposed within the context of their 2D
///   transposition slice ([1, 0]).
///
/// Example 2: dim0 = 0, dim1 = 1, transp = [2, 1, 0]
///   Return true: dim0 and dim1 are transposed within the context of their 2D
///   transposition slice ([1, 0]). Paradoxically, note how dim1 (1) is *not*
///   transposed within the full context of the transposition.
///
/// Example 3: dim0 = 0, dim1 = 1, transp = [2, 0, 1]
///   Return false: dim0 and dim1 are *not* transposed within the context of
///   their 2D transposition slice ([0, 1]). Paradoxically, note how dim0 (0)
///   and dim1 (1) are transposed within the full context of the of the
///   transposition.
static bool areDimsTransposedIn2DSlice(int64_t dim0, int64_t dim1,
                                       ArrayRef<int64_t> transp) {
  // Perform a linear scan along the dimensions of the transposed pattern. If
  // dim0 is found first, dim0 and dim1 are not transposed within the context of
  // their 2D slice. Otherwise, 'dim1' is found first and they are transposed.
  for (int64_t permDim : transp) {
    if (permDim == dim0)
      return false;
    if (permDim == dim1)
      return true;
  }

  llvm_unreachable("Ill-formed transpose pattern");
}

FailureOr<std::pair<int, int>>
mlir::vector::isTranspose2DSlice(vector::TransposeOp op) {
  VectorType srcType = op.getSourceVectorType();
  SmallVector<int64_t> srcGtOneDims;
  for (auto [index, size] : llvm::enumerate(srcType.getShape()))
    if (size > 1)
      srcGtOneDims.push_back(index);

  if (srcGtOneDims.size() != 2)
    return failure();

  // Check whether the two source vector dimensions that are greater than one
  // must be transposed with each other so that we can apply one of the 2-D
  // transpose patterns. Otherwise, these patterns are not applicable.
  if (!areDimsTransposedIn2DSlice(srcGtOneDims[0], srcGtOneDims[1],
                                  op.getPermutation()))
    return failure();

  return std::pair<int, int>(srcGtOneDims[0], srcGtOneDims[1]);
}

/// Constructs a permutation map from memref indices to vector dimension.
///
/// The implementation uses the knowledge of the mapping of enclosing loop to
/// vector dimension. `enclosingLoopToVectorDim` carries this information as a
/// map with:
///   - keys representing "vectorized enclosing loops";
///   - values representing the corresponding vector dimension.
/// The algorithm traverses "vectorized enclosing loops" and extracts the
/// at-most-one MemRef index that is invariant along said loop. This index is
/// guaranteed to be at most one by construction: otherwise the MemRef is not
/// vectorizable.
/// If this invariant index is found, it is added to the permutation_map at the
/// proper vector dimension.
/// If no index is found to be invariant, 0 is added to the permutation_map and
/// corresponds to a vector broadcast along that dimension.
///
/// Returns an empty AffineMap if `enclosingLoopToVectorDim` is empty,
/// signalling that no permutation map can be constructed given
/// `enclosingLoopToVectorDim`.
///
/// Examples can be found in the documentation of `makePermutationMap`, in the
/// header file.
static AffineMap makePermutationMap(
    ArrayRef<Value> indices,
    const DenseMap<Operation *, unsigned> &enclosingLoopToVectorDim) {
  if (enclosingLoopToVectorDim.empty())
    return AffineMap();
  MLIRContext *context =
      enclosingLoopToVectorDim.begin()->getFirst()->getContext();
  SmallVector<AffineExpr> perm(enclosingLoopToVectorDim.size(),
                               getAffineConstantExpr(0, context));

  for (auto kvp : enclosingLoopToVectorDim) {
    assert(kvp.second < perm.size());
    auto invariants = affine::getInvariantAccesses(
        cast<affine::AffineForOp>(kvp.first).getInductionVar(), indices);
    unsigned numIndices = indices.size();
    unsigned countInvariantIndices = 0;
    for (unsigned dim = 0; dim < numIndices; ++dim) {
      if (!invariants.count(indices[dim])) {
        assert(perm[kvp.second] == getAffineConstantExpr(0, context) &&
               "permutationMap already has an entry along dim");
        perm[kvp.second] = getAffineDimExpr(dim, context);
      } else {
        ++countInvariantIndices;
      }
    }
    assert((countInvariantIndices == numIndices ||
            countInvariantIndices == numIndices - 1) &&
           "Vectorization prerequisite violated: at most 1 index may be "
           "invariant wrt a vectorized loop");
    (void)countInvariantIndices;
  }
  return AffineMap::get(indices.size(), 0, perm, context);
}

/// Implementation detail that walks up the parents and records the ones with
/// the specified type.
/// TODO: could also be implemented as a collect parents followed by a
/// filter and made available outside this file.
template <typename T>
static SetVector<Operation *> getParentsOfType(Block *block) {
  SetVector<Operation *> res;
  auto *current = block->getParentOp();
  while (current) {
    if ([[maybe_unused]] auto typedParent = dyn_cast<T>(current)) {
      assert(res.count(current) == 0 && "Already inserted");
      res.insert(current);
    }
    current = current->getParentOp();
  }
  return res;
}

/// Returns the enclosing AffineForOp, from closest to farthest.
static SetVector<Operation *> getEnclosingforOps(Block *block) {
  return getParentsOfType<affine::AffineForOp>(block);
}

AffineMap mlir::makePermutationMap(
    Block *insertPoint, ArrayRef<Value> indices,
    const DenseMap<Operation *, unsigned> &loopToVectorDim) {
  DenseMap<Operation *, unsigned> enclosingLoopToVectorDim;
  auto enclosingLoops = getEnclosingforOps(insertPoint);
  for (auto *forInst : enclosingLoops) {
    auto it = loopToVectorDim.find(forInst);
    if (it != loopToVectorDim.end()) {
      enclosingLoopToVectorDim.insert(*it);
    }
  }
  return ::makePermutationMap(indices, enclosingLoopToVectorDim);
}

AffineMap mlir::makePermutationMap(
    Operation *op, ArrayRef<Value> indices,
    const DenseMap<Operation *, unsigned> &loopToVectorDim) {
  return makePermutationMap(op->getBlock(), indices, loopToVectorDim);
}

bool matcher::operatesOnSuperVectorsOf(Operation &op,
                                       VectorType subVectorType) {
  // First, extract the vector type and distinguish between:
  //   a. ops that *must* lower a super-vector (i.e. vector.transfer_read,
  //      vector.transfer_write); and
  //   b. ops that *may* lower a super-vector (all other ops).
  // The ops that *may* lower a super-vector only do so if the super-vector to
  // sub-vector ratio exists. The ops that *must* lower a super-vector are
  // explicitly checked for this property.
  /// TODO: there should be a single function for all ops to do this so we
  /// do not have to special case. Maybe a trait, or just a method, unclear atm.
  bool mustDivide = false;
  (void)mustDivide;
  VectorType superVectorType;
  if (auto transfer = dyn_cast<VectorTransferOpInterface>(op)) {
    superVectorType = transfer.getVectorType();
    mustDivide = true;
  } else if (op.getNumResults() == 0) {
    if (!isa<func::ReturnOp>(op)) {
      op.emitError("NYI: assuming only return operations can have 0 "
                   " results at this point");
    }
    return false;
  } else if (op.getNumResults() == 1) {
    if (auto v = dyn_cast<VectorType>(op.getResult(0).getType())) {
      superVectorType = v;
    } else {
      // Not a vector type.
      return false;
    }
  } else {
    // Not a vector.transfer and has more than 1 result, fail hard for now to
    // wake us up when something changes.
    op.emitError("NYI: operation has more than 1 result");
    return false;
  }

  // Get the ratio.
  auto ratio =
      computeShapeRatio(superVectorType.getShape(), subVectorType.getShape());

  // Sanity check.
  assert((ratio || !mustDivide) &&
         "vector.transfer operation in which super-vector size is not an"
         " integer multiple of sub-vector size");

  // This catches cases that are not strictly necessary to have multiplicity but
  // still aren't divisible by the sub-vector shape.
  // This could be useful information if we wanted to reshape at the level of
  // the vector type (but we would have to look at the compute and distinguish
  // between parallel, reduction and possibly other cases.
  return ratio.has_value();
}

bool vector::isContiguousSlice(MemRefType memrefType, VectorType vectorType) {
  if (vectorType.isScalable())
    return false;

  // Ignore a leading sequence of adjacent unit dimensions in the vector.
  ArrayRef<int64_t> vectorShape =
      vectorType.getShape().drop_while([](auto v) { return v == 1; });
  auto vecRank = vectorShape.size();

  if (!memrefType.areTrailingDimsContiguous(vecRank))
    return false;

  // Extract the trailing dims of the input memref
  auto memrefShape = memrefType.getShape().take_back(vecRank);

  // Compare the dims of `vectorType` against `memrefType`.
  // All of the dimensions, except the first must match.
  return llvm::equal(vectorShape.drop_front(), memrefShape.drop_front());
}

std::optional<StaticTileOffsetRange>
vector::createUnrollIterator(VectorType vType, int64_t targetRank) {
  if (vType.getRank() <= targetRank)
    return {};
  // Attempt to unroll until targetRank or the first scalable dimension (which
  // cannot be unrolled).
  auto shapeToUnroll = vType.getShape().drop_back(targetRank);
  auto inputScalableVecDimsToUnroll =
      vType.getScalableDims().drop_back(targetRank);
  const auto *it = llvm::find(inputScalableVecDimsToUnroll, true);
  auto firstScalableDim = it - inputScalableVecDimsToUnroll.begin();
  if (firstScalableDim == 0)
    return {};
  // All scalable dimensions should be removed now.
  inputScalableVecDimsToUnroll =
      inputScalableVecDimsToUnroll.slice(0, firstScalableDim);
  assert(!llvm::is_contained(inputScalableVecDimsToUnroll, true) &&
         "unexpected leading scalable dimension");
  // Create an unroll iterator for leading dimensions.
  shapeToUnroll = shapeToUnroll.slice(0, firstScalableDim);
  return StaticTileOffsetRange(shapeToUnroll, /*unrollStep=*/1);
}

SmallVector<OpFoldResult> vector::getMixedSizesXfer(bool hasTensorSemantics,
                                                    Operation *xfer,
                                                    RewriterBase &rewriter) {
  auto loc = xfer->getLoc();

  Value base = TypeSwitch<Operation *, Value>(xfer)
                   .Case<vector::TransferReadOp>(
                       [&](auto readOp) { return readOp.getBase(); })
                   .Case<vector::TransferWriteOp>(
                       [&](auto writeOp) { return writeOp.getOperand(1); });

  SmallVector<OpFoldResult> mixedSourceDims =
      hasTensorSemantics ? tensor::getMixedSizes(rewriter, loc, base)
                         : memref::getMixedSizes(rewriter, loc, base);
  return mixedSourceDims;
}

bool vector::isLinearizableVector(VectorType type) {
  return (type.getRank() > 1) && (type.getNumScalableDims() <= 1);
}

Value vector::createReadOrMaskedRead(OpBuilder &builder, Location loc,
                                     Value source,
                                     ArrayRef<int64_t> inputVectorSizes,
                                     std::optional<Value> padValue,
                                     bool useInBoundsInsteadOfMasking,
                                     ArrayRef<bool> inputScalableVecDims) {
  assert(!llvm::is_contained(inputVectorSizes, ShapedType::kDynamic) &&
         "invalid input vector sizes");
  auto sourceShapedType = cast<ShapedType>(source.getType());
  auto sourceShape = sourceShapedType.getShape();
  assert(sourceShape.size() == inputVectorSizes.size() &&
         "expected same ranks.");
  auto vectorType =
      VectorType::get(inputVectorSizes, sourceShapedType.getElementType(),
                      inputScalableVecDims);
  assert((!padValue.has_value() ||
          padValue.value().getType() == sourceShapedType.getElementType()) &&
         "expected same pad element type to match source element type");
  int64_t readRank = inputVectorSizes.size();
  auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
  SmallVector<bool> inBoundsVal(readRank, true);

  if (useInBoundsInsteadOfMasking) {
    // Update the inBounds attribute.
    // FIXME: This computation is too weak - it ignores the read indices.
    for (unsigned i = 0; i < readRank; i++)
      inBoundsVal[i] = (sourceShape[i] == inputVectorSizes[i]) &&
                       ShapedType::isStatic(sourceShape[i]);
  }
  auto transferReadOp = vector::TransferReadOp::create(
      builder, loc,
      /*vectorType=*/vectorType,
      /*source=*/source,
      /*indices=*/SmallVector<Value>(readRank, zero),
      /*padding=*/padValue,
      /*inBounds=*/inBoundsVal);

  if (llvm::equal(inputVectorSizes, sourceShape) || useInBoundsInsteadOfMasking)
    return transferReadOp;
  SmallVector<OpFoldResult> mixedSourceDims =
      isa<MemRefType>(source.getType())
          ? memref::getMixedSizes(builder, loc, source)
          : tensor::getMixedSizes(builder, loc, source);

  auto maskType = VectorType::get(inputVectorSizes, builder.getI1Type(),
                                  inputScalableVecDims);
  Value mask =
      vector::CreateMaskOp::create(builder, loc, maskType, mixedSourceDims);
  return mlir::vector::maskOperation(builder, transferReadOp, mask)
      ->getResult(0);
}

LogicalResult
vector::isValidMaskedInputVector(ArrayRef<int64_t> shape,
                                 ArrayRef<int64_t> inputVectorSizes) {
  LDBG() << "Iteration space static sizes:" << llvm::interleaved(shape);

  if (inputVectorSizes.size() != shape.size()) {
    LDBG() << "Input vector sizes don't match the number of loops";
    return failure();
  }
  if (ShapedType::isDynamicShape(inputVectorSizes)) {
    LDBG() << "Input vector sizes can't have dynamic dimensions";
    return failure();
  }
  if (!llvm::all_of(llvm::zip(shape, inputVectorSizes),
                    [](std::tuple<int64_t, int64_t> sizePair) {
                      int64_t staticSize = std::get<0>(sizePair);
                      int64_t inputSize = std::get<1>(sizePair);
                      return ShapedType::isDynamic(staticSize) ||
                             staticSize <= inputSize;
                    })) {
    LDBG() << "Input vector sizes must be greater than or equal to iteration "
              "space static sizes";
    return failure();
  }
  return success();
}

/// Takes a 2+ dimensional vector as an input
/// returns n vector values produced by n vector.extract operations.
/// I.e. calling unrollVectorValue([[%v]], rewriter) such that
///
///   %v : vector<nxaxb...>
///
/// will produce the following IR changes
///
///   %v0 = vector.extract %v[0] : vector<axbx...> from vector<nxaxb...>
///   %v1 = vector.extract %v[1] : vector<axbx...> from vector<nxaxb...>
///   ...
///   %vnminusone = vector.extract %v[n-1] : vector<axbx...> from ...
///
/// and returns SmallVector<Value> r = {[[%v0]], [[%v1]], ..., [[%vnminusone]]}
FailureOr<SmallVector<Value>>
vector::unrollVectorValue(TypedValue<VectorType> vector,
                          RewriterBase &rewriter) {
  SmallVector<Value> subvectors;
  VectorType ty = cast<VectorType>(vector.getType());
  Location loc = vector.getLoc();
  if (ty.getRank() < 2)
    return rewriter.notifyMatchFailure(loc, "already 1-D");

  // Unrolling doesn't take vscale into account. Pattern is disabled for
  // vectors with leading scalable dim(s).
  if (ty.getScalableDims().front())
    return rewriter.notifyMatchFailure(loc, "cannot unroll scalable dim");

  for (int64_t i = 0, e = ty.getShape().front(); i < e; ++i) {
    subvectors.push_back(vector::ExtractOp::create(rewriter, loc, vector, i));
  }

  return subvectors;
}

LogicalResult vector::unrollVectorOp(Operation *op, PatternRewriter &rewriter,
                                     vector::UnrollVectorOpFn unrollFn) {
  assert(op->getNumResults() == 1 && "expected single result");
  assert(isa<VectorType>(op->getResult(0).getType()) && "expected vector type");
  VectorType resultTy = cast<VectorType>(op->getResult(0).getType());
  if (resultTy.getRank() < 2)
    return rewriter.notifyMatchFailure(op, "already 1-D");

  // Unrolling doesn't take vscale into account. Pattern is disabled for
  // vectors with leading scalable dim(s).
  if (resultTy.getScalableDims().front())
    return rewriter.notifyMatchFailure(op, "cannot unroll scalable dim");

  Location loc = op->getLoc();
  Value result = ub::PoisonOp::create(rewriter, loc, resultTy);
  VectorType subTy = VectorType::Builder(resultTy).dropDim(0);

  for (int64_t i = 0, e = resultTy.getShape().front(); i < e; ++i) {
    Value subVector = unrollFn(rewriter, loc, subTy, i);
    result = vector::InsertOp::create(rewriter, loc, subVector, result, i);
  }

  rewriter.replaceOp(op, result);
  return success();
}
