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

#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include <numeric>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using llvm::SetVector;

using namespace mlir;

/// Return the number of elements of basis, `0` if empty.
int64_t mlir::computeMaxLinearIndex(ArrayRef<int64_t> basis) {
  if (basis.empty())
    return 0;
  return std::accumulate(basis.begin(), basis.end(), 1,
                         std::multiplies<int64_t>());
}

/// Given a shape with sizes greater than 0 along all dimensions,
/// return the distance, in number of elements, between a slice in a dimension
/// and the next slice in the same dimension.
///   e.g. shape[3, 4, 5] -> linearization_basis[20, 5, 1]
SmallVector<int64_t, 8> mlir::computeStrides(ArrayRef<int64_t> shape) {
  if (shape.empty())
    return {};
  SmallVector<int64_t, 8> tmp;
  tmp.reserve(shape.size());
  int64_t running = 1;
  for (auto size : llvm::reverse(shape)) {
    assert(size > 0 && "size must be nonnegative");
    tmp.push_back(running);
    running *= size;
  }
  return SmallVector<int64_t, 8>(tmp.rbegin(), tmp.rend());
}

SmallVector<int64_t, 4> mlir::computeStrides(ArrayRef<int64_t> shape,
                                             ArrayRef<int64_t> sizes) {
  int64_t rank = shape.size();
  // Compute the count for each dimension.
  SmallVector<int64_t, 4> sliceDimCounts(rank);
  for (int64_t r = 0; r < rank; ++r)
    sliceDimCounts[r] = ceilDiv(shape[r], sizes[r]);
  // Use that to compute the slice stride for each dimension.
  SmallVector<int64_t, 4> sliceStrides(rank);
  sliceStrides[rank - 1] = 1;
  for (int64_t r = rank - 2; r >= 0; --r)
    sliceStrides[r] = sliceStrides[r + 1] * sliceDimCounts[r + 1];
  return sliceStrides;
}

int64_t mlir::linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis) {
  assert(offsets.size() == basis.size());
  int64_t linearIndex = 0;
  for (unsigned idx = 0, e = basis.size(); idx < e; ++idx)
    linearIndex += offsets[idx] * basis[idx];
  return linearIndex;
}

SmallVector<int64_t, 4> mlir::delinearize(ArrayRef<int64_t> sliceStrides,
                                          int64_t index) {
  int64_t rank = sliceStrides.size();
  SmallVector<int64_t, 4> vectorOffsets(rank);
  for (int64_t r = 0; r < rank; ++r) {
    assert(sliceStrides[r] > 0);
    vectorOffsets[r] = index / sliceStrides[r];
    index %= sliceStrides[r];
  }
  return vectorOffsets;
}

SmallVector<int64_t, 4> mlir::computeElementOffsetsFromVectorSliceOffsets(
    ArrayRef<int64_t> sizes, ArrayRef<int64_t> vectorOffsets) {
  SmallVector<int64_t, 4> result;
  for (auto it : llvm::zip(vectorOffsets, sizes))
    result.push_back(std::get<0>(it) * std::get<1>(it));
  return result;
}

SmallVector<int64_t, 4>
mlir::computeSliceSizes(ArrayRef<int64_t> shape, ArrayRef<int64_t> sizes,
                        ArrayRef<int64_t> elementOffsets) {
  int64_t rank = shape.size();
  SmallVector<int64_t, 4> sliceSizes(rank);
  for (unsigned r = 0; r < rank; ++r)
    sliceSizes[r] = std::min(sizes[r], shape[r] - elementOffsets[r]);
  return sliceSizes;
}

Optional<SmallVector<int64_t, 4>> mlir::shapeRatio(ArrayRef<int64_t> superShape,
                                                   ArrayRef<int64_t> subShape) {
  if (superShape.size() < subShape.size()) {
    return Optional<SmallVector<int64_t, 4>>();
  }

  // Starting from the end, compute the integer divisors.
  std::vector<int64_t> result;
  result.reserve(superShape.size());
  int64_t superSize = 0, subSize = 0;
  for (auto it :
       llvm::zip(llvm::reverse(superShape), llvm::reverse(subShape))) {
    std::tie(superSize, subSize) = it;
    assert(superSize > 0 && "superSize must be > 0");
    assert(subSize > 0 && "subSize must be > 0");

    // If integral division does not occur, return and let the caller decide.
    if (superSize % subSize != 0)
      return None;
    result.push_back(superSize / subSize);
  }

  // At this point we computed the ratio (in reverse) for the common
  // size. Fill with the remaining entries from the super-vector shape (still in
  // reverse).
  int commonSize = subShape.size();
  std::copy(superShape.rbegin() + commonSize, superShape.rend(),
            std::back_inserter(result));

  assert(result.size() == superShape.size() &&
         "super to sub shape ratio is not of the same size as the super rank");

  // Reverse again to get it back in the proper order and return.
  return SmallVector<int64_t, 4>{result.rbegin(), result.rend()};
}

Optional<SmallVector<int64_t, 4>> mlir::shapeRatio(VectorType superVectorType,
                                                   VectorType subVectorType) {
  assert(superVectorType.getElementType() == subVectorType.getElementType() &&
         "vector types must be of the same elemental type");
  return shapeRatio(superVectorType.getShape(), subVectorType.getShape());
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
  SmallVector<AffineExpr, 4> perm(enclosingLoopToVectorDim.size(),
                                  getAffineConstantExpr(0, context));

  for (auto kvp : enclosingLoopToVectorDim) {
    assert(kvp.second < perm.size());
    auto invariants = getInvariantAccesses(
        cast<AffineForOp>(kvp.first).getInductionVar(), indices);
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
  }
  return AffineMap::get(indices.size(), 0, perm, context);
}

/// Implementation detail that walks up the parents and records the ones with
/// the specified type.
/// TODO: could also be implemented as a collect parents followed by a
/// filter and made available outside this file.
template <typename T>
static SetVector<Operation *> getParentsOfType(Operation *op) {
  SetVector<Operation *> res;
  auto *current = op;
  while (auto *parent = current->getParentOp()) {
    if (auto typedParent = dyn_cast<T>(parent)) {
      assert(res.count(parent) == 0 && "Already inserted");
      res.insert(parent);
    }
    current = parent;
  }
  return res;
}

/// Returns the enclosing AffineForOp, from closest to farthest.
static SetVector<Operation *> getEnclosingforOps(Operation *op) {
  return getParentsOfType<AffineForOp>(op);
}

AffineMap mlir::makePermutationMap(
    Operation *op, ArrayRef<Value> indices,
    const DenseMap<Operation *, unsigned> &loopToVectorDim) {
  DenseMap<Operation *, unsigned> enclosingLoopToVectorDim;
  auto enclosingLoops = getEnclosingforOps(op);
  for (auto *forInst : enclosingLoops) {
    auto it = loopToVectorDim.find(forInst);
    if (it != loopToVectorDim.end()) {
      enclosingLoopToVectorDim.insert(*it);
    }
  }
  return ::makePermutationMap(indices, enclosingLoopToVectorDim);
}

AffineMap mlir::getTransferMinorIdentityMap(MemRefType memRefType,
                                            VectorType vectorType) {
  int64_t elementVectorRank = 0;
  VectorType elementVectorType =
      memRefType.getElementType().dyn_cast<VectorType>();
  if (elementVectorType)
    elementVectorRank += elementVectorType.getRank();
  return AffineMap::getMinorIdentityMap(
      memRefType.getRank(), vectorType.getRank() - elementVectorRank,
      memRefType.getContext());
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
    if (!isa<ReturnOp>(op)) {
      op.emitError("NYI: assuming only return operations can have 0 "
                   " results at this point");
    }
    return false;
  } else if (op.getNumResults() == 1) {
    if (auto v = op.getResult(0).getType().dyn_cast<VectorType>()) {
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
  auto ratio = shapeRatio(superVectorType, subVectorType);

  // Sanity check.
  assert((ratio.hasValue() || !mustDivide) &&
         "vector.transfer operation in which super-vector size is not an"
         " integer multiple of sub-vector size");

  // This catches cases that are not strictly necessary to have multiplicity but
  // still aren't divisible by the sub-vector shape.
  // This could be useful information if we wanted to reshape at the level of
  // the vector type (but we would have to look at the compute and distinguish
  // between parallel, reduction and possibly other cases.
  if (!ratio.hasValue()) {
    return false;
  }

  return true;
}

bool mlir::isDisjointTransferSet(VectorTransferOpInterface transferA,
                                 VectorTransferOpInterface transferB) {
  if (transferA.memref() != transferB.memref())
    return false;
  // For simplicity only look at transfer of same type.
  if (transferA.getVectorType() != transferB.getVectorType())
    return false;
  unsigned rankOffset = transferA.getLeadingMemRefRank();
  for (unsigned i = 0, e = transferA.indices().size(); i < e; i++) {
    auto indexA = transferA.indices()[i].getDefiningOp<ConstantOp>();
    auto indexB = transferB.indices()[i].getDefiningOp<ConstantOp>();
    // If any of the indices are dynamic we cannot prove anything.
    if (!indexA || !indexB)
      continue;

    if (i < rankOffset) {
      // For leading dimensions, if we can prove that index are different we
      // know we are accessing disjoint slices.
      if (indexA.getValue().cast<IntegerAttr>().getInt() !=
          indexB.getValue().cast<IntegerAttr>().getInt())
        return true;
    } else {
      // For this dimension, we slice a part of the memref we need to make sure
      // the intervals accessed don't overlap.
      int64_t distance =
          std::abs(indexA.getValue().cast<IntegerAttr>().getInt() -
                   indexB.getValue().cast<IntegerAttr>().getInt());
      if (distance >= transferA.getVectorType().getDimSize(i - rankOffset))
        return true;
    }
  }
  return false;
}
