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
#include "mlir/Support/MathExtras.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

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

  SmallVector<int64_t> transp;
  for (auto attr : op.getTransp())
    transp.push_back(cast<IntegerAttr>(attr).getInt());

  // Check whether the two source vector dimensions that are greater than one
  // must be transposed with each other so that we can apply one of the 2-D
  // transpose pattens. Otherwise, these patterns are not applicable.
  if (!areDimsTransposedIn2DSlice(srcGtOneDims[0], srcGtOneDims[1], transp))
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
