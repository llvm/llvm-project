//===- VectorRewritePatterns.h - Vector rewrite patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_VECTOR_VECTORREWRITEPATTERNS_H_
#define DIALECT_VECTOR_VECTORREWRITEPATTERNS_H_

namespace mlir {
class RewritePatternSet;

namespace vector {

/// Populate `patterns` with the following patterns.
///
/// [VectorInsertStridedSliceOpDifferentRankRewritePattern]
/// =======================================================
/// RewritePattern for InsertStridedSliceOp where source and destination vectors
/// have different ranks.
///
/// When ranks are different, InsertStridedSlice needs to extract a properly
/// ranked vector from the destination vector into which to insert. This pattern
/// only takes care of this extraction part and forwards the rest to
/// [VectorInsertStridedSliceOpSameRankRewritePattern].
///
/// For a k-D source and n-D destination vector (k < n), we emit:
///   1. ExtractOp to extract the (unique) (n-1)-D subvector into which to
///      insert the k-D source.
///   2. k-D -> (n-1)-D InsertStridedSlice op
///   3. InsertOp that is the reverse of 1.
///
/// [VectorInsertStridedSliceOpSameRankRewritePattern]
/// ==================================================
/// RewritePattern for InsertStridedSliceOp where source and destination vectors
/// have the same rank. For each outermost index in the slice:
///   begin    end             stride
/// [offset : offset+size*stride : stride]
///   1. ExtractOp one (k-1)-D source subvector and one (n-1)-D dest subvector.
///   2. InsertStridedSlice (k-1)-D into (n-1)-D
///   3. the destination subvector is inserted back in the proper place
///   3. InsertOp that is the reverse of 1.
///
/// [VectorExtractStridedSliceOpRewritePattern]
/// ===========================================
/// Progressive lowering of ExtractStridedSliceOp to either:
///   1. single offset extract as a direct vector::ShuffleOp.
///   2. ExtractOp/ExtractElementOp + lower rank ExtractStridedSliceOp +
///      InsertOp/InsertElementOp for the n-D case.
void populateVectorInsertExtractStridedSliceTransforms(
    RewritePatternSet &patterns);

} // namespace vector
} // namespace mlir

#endif // DIALECT_VECTOR_VECTORREWRITEPATTERNS_H_
