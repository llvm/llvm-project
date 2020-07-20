//===- VectorUtils.h - Vector Utilities -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_VECTORUTILS_H_
#define MLIR_DIALECT_VECTOR_VECTORUTILS_H_

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir {

// Forward declarations.
class AffineApplyOp;
class AffineForOp;
class AffineMap;
class Location;
class MemRefType;
class OpBuilder;
class Operation;
class Value;
class VectorType;

/// Return the number of elements of basis, `0` if empty.
int64_t computeMaxLinearIndex(ArrayRef<int64_t> basis);

/// Given a shape with sizes greater than 0 along all dimensions,
/// return the distance, in number of elements, between a slice in a dimension
/// and the next slice in the same dimension.
///   e.g. shape[3, 4, 5] -> linearization_basis[20, 5, 1]
SmallVector<int64_t, 8> computeStrides(ArrayRef<int64_t> shape);

/// Given the shape and sizes of a vector, returns the corresponding
/// strides for each dimension.
/// TODO: needs better doc of how it is used.
SmallVector<int64_t, 4> computeStrides(ArrayRef<int64_t> shape,
                                       ArrayRef<int64_t> sizes);

/// Computes and returns the linearized index of 'offsets' w.r.t. 'basis'.
int64_t linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis);

/// Given the strides together with a linear index in the dimension
/// space, returns the vector-space offsets in each dimension for a
/// de-linearized index.
SmallVector<int64_t, 4> delinearize(ArrayRef<int64_t> strides,
                                    int64_t linearIndex);

/// Given the target sizes of a vector, together with vector-space offsets,
/// returns the element-space offsets for each dimension.
SmallVector<int64_t, 4>
computeElementOffsetsFromVectorSliceOffsets(ArrayRef<int64_t> sizes,
                                            ArrayRef<int64_t> vectorOffsets);

/// Given the shape, sizes, and element-space offsets of a vector, returns
/// the slize sizes for each dimension.
SmallVector<int64_t, 4> computeSliceSizes(ArrayRef<int64_t> shape,
                                          ArrayRef<int64_t> sizes,
                                          ArrayRef<int64_t> elementOffsets);

/// Computes and returns the multi-dimensional ratio of `superShape` to
/// `subShape`. This is calculated by performing a traversal from minor to major
/// dimensions (i.e. in reverse shape order). If integral division is not
/// possible, returns None.
/// The ArrayRefs are assumed (and enforced) to only contain > 1 values.
/// This constraint comes from the fact that they are meant to be used with
/// VectorTypes, for which the property holds by construction.
///
/// Examples:
///   - shapeRatio({3, 4, 5, 8}, {2, 5, 2}) returns {3, 2, 1, 4}
///   - shapeRatio({3, 4, 4, 8}, {2, 5, 2}) returns None
///   - shapeRatio({1, 2, 10, 32}, {2, 5, 2}) returns {1, 1, 2, 16}
Optional<SmallVector<int64_t, 4>> shapeRatio(ArrayRef<int64_t> superShape,
                                             ArrayRef<int64_t> subShape);

/// Computes and returns the multi-dimensional ratio of the shapes of
/// `superVector` to `subVector`. If integral division is not possible, returns
/// None.
/// Assumes and enforces that the VectorTypes have the same elemental type.
Optional<SmallVector<int64_t, 4>> shapeRatio(VectorType superVectorType,
                                             VectorType subVectorType);

/// Constructs a permutation map of invariant memref indices to vector
/// dimension.
///
/// If no index is found to be invariant, 0 is added to the permutation_map and
/// corresponds to a vector broadcast along that dimension.
///
/// The implementation uses the knowledge of the mapping of loops to
/// vector dimension. `loopToVectorDim` carries this information as a map with:
///   - keys representing "vectorized enclosing loops";
///   - values representing the corresponding vector dimension.
/// Note that loopToVectorDim is a whole function map from which only enclosing
/// loop information is extracted.
///
/// Prerequisites: `opInst` is a vectorizable load or store operation (i.e. at
/// most one invariant index along each AffineForOp of `loopToVectorDim`).
///
/// Example 1:
/// The following MLIR snippet:
///
/// ```mlir
///    affine.for %i3 = 0 to %0 {
///      affine.for %i4 = 0 to %1 {
///        affine.for %i5 = 0 to %2 {
///          %a5 = load %arg0[%i4, %i5, %i3] : memref<?x?x?xf32>
///    }}}
/// ```
///
/// may vectorize with {permutation_map: (d0, d1, d2) -> (d2, d1)} into:
///
/// ```mlir
///    affine.for %i3 = 0 to %0 step 32 {
///      affine.for %i4 = 0 to %1 {
///        affine.for %i5 = 0 to %2 step 256 {
///          %4 = vector.transfer_read %arg0, %i4, %i5, %i3
///               {permutation_map: (d0, d1, d2) -> (d2, d1)} :
///               (memref<?x?x?xf32>, index, index) -> vector<32x256xf32>
///    }}}
/// ```
///
/// Meaning that vector.transfer_read will be responsible for reading the slice:
/// `%arg0[%i4, %i5:%15+256, %i3:%i3+32]` into vector<32x256xf32>.
///
/// Example 2:
/// The following MLIR snippet:
///
/// ```mlir
///    %cst0 = constant 0 : index
///    affine.for %i0 = 0 to %0 {
///      %a0 = load %arg0[%cst0, %cst0] : memref<?x?xf32>
///    }
/// ```
///
/// may vectorize with {permutation_map: (d0) -> (0)} into:
///
/// ```mlir
///    affine.for %i0 = 0 to %0 step 128 {
///      %3 = vector.transfer_read %arg0, %c0_0, %c0_0
///           {permutation_map: (d0, d1) -> (0)} :
///           (memref<?x?xf32>, index, index) -> vector<128xf32>
///    }
/// ````
///
/// Meaning that vector.transfer_read will be responsible of reading the slice
/// `%arg0[%c0, %c0]` into vector<128xf32> which needs a 1-D vector broadcast.
///
AffineMap
makePermutationMap(Operation *op, ArrayRef<Value> indices,
                   const DenseMap<Operation *, unsigned> &loopToVectorDim);

/// Build the default minor identity map suitable for a vector transfer. This
/// also handles the case memref<... x vector<...>> -> vector<...> in which the
/// rank of the identity map must take the vector element type into account.
AffineMap getTransferMinorIdentityMap(MemRefType memRefType,
                                      VectorType vectorType);

namespace matcher {

/// Matches vector.transfer_read, vector.transfer_write and ops that return a
/// vector type that is a multiple of the sub-vector type. This allows passing
/// over other smaller vector types in the function and avoids interfering with
/// operations on those.
/// This is a first approximation, it can easily be extended in the future.
/// TODO: this could all be much simpler if we added a bit that a vector type to
/// mark that a vector is a strict super-vector but it still does not warrant
/// adding even 1 extra bit in the IR for now.
bool operatesOnSuperVectorsOf(Operation &op, VectorType subVectorType);

} // end namespace matcher
} // end namespace mlir

#endif // MLIR_DIALECT_VECTOR_VECTORUTILS_H_
