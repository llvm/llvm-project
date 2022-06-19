//===- VectorTransforms.h - Vector transformations as patterns --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORTRANSFORMS_H
#define MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORTRANSFORMS_H

#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"

namespace mlir {
class MLIRContext;
class VectorTransferOpInterface;
class RewritePatternSet;
class RewriterBase;

namespace scf {
class IfOp;
} // namespace scf

namespace vector {

//===----------------------------------------------------------------------===//
// Standalone transformations and helpers.
//===----------------------------------------------------------------------===//
/// Split a vector.transfer operation into an in-bounds (i.e., no out-of-bounds
/// masking) fastpath and a slowpath.
/// If `ifOp` is not null and the result is `success, the `ifOp` points to the
/// newly created conditional upon function return.
/// To accomodate for the fact that the original vector.transfer indexing may be
/// arbitrary and the slow path indexes @[0...0] in the temporary buffer, the
/// scf.if op returns a view and values of type index.
/// At this time, only vector.transfer_read case is implemented.
///
/// Example (a 2-D vector.transfer_read):
/// ```
///    %1 = vector.transfer_read %0[...], %pad : memref<A...>, vector<...>
/// ```
/// is transformed into:
/// ```
///    %1:3 = scf.if (%inBounds) {
///      // fastpath, direct cast
///      memref.cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view : compatibleMemRefType, index, index
///    } else {
///      // slowpath, not in-bounds vector.transfer or linalg.copy.
///      memref.cast %alloc: memref<B...> to compatibleMemRefType
///      scf.yield %4 : compatibleMemRefType, index, index
//     }
///    %0 = vector.transfer_read %1#0[%1#1, %1#2] {in_bounds = [true ... true]}
/// ```
/// where `alloc` is a top of the function alloca'ed buffer of one vector.
///
/// Preconditions:
///  1. `xferOp.permutation_map()` must be a minor identity map
///  2. the rank of the `xferOp.memref()` and the rank of the `xferOp.vector()`
///  must be equal. This will be relaxed in the future but requires
///  rank-reducing subviews.
LogicalResult splitFullAndPartialTransfer(
    RewriterBase &b, VectorTransferOpInterface xferOp,
    VectorTransformsOptions options = VectorTransformsOptions(),
    scf::IfOp *ifOp = nullptr);

struct DistributeOps {
  ExtractMapOp extract;
  InsertMapOp insert;
};

/// Distribute a N-D vector pointwise operation over a range of given ids taking
/// *all* values in [0 .. multiplicity - 1] (e.g. loop induction variable or
/// SPMD id). This transformation only inserts
/// vector.extract_map/vector.insert_map. It is meant to be used with
/// canonicalizations pattern to propagate and fold the vector
/// insert_map/extract_map operations.
/// Transforms:
//  %v = arith.addf %a, %b : vector<32xf32>
/// to:
/// %v = arith.addf %a, %b : vector<32xf32>
/// %ev = vector.extract_map %v, %id, 32 : vector<32xf32> into vector<1xf32>
/// %nv = vector.insert_map %ev, %id, 32 : vector<1xf32> into vector<32xf32>
Optional<DistributeOps>
distributPointwiseVectorOp(OpBuilder &builder, Operation *op,
                           ArrayRef<Value> id, ArrayRef<int64_t> multiplicity,
                           const AffineMap &map);

/// Implements transfer op write to read forwarding and dead transfer write
/// optimizations.
void transferOpflowOpt(Operation *rootOp);

} // namespace vector
} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORTRANSFORMS_H
