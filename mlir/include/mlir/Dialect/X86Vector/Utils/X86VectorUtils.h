//===- X86VectorUtils.h - X86Vector Utilities -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_X86VECTOR_UTILS_X86VECTORUTILS_H_
#define MLIR_DIALECT_X86VECTOR_UTILS_X86VECTORUTILS_H_

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
class AffineMap;
class Operation;

namespace x86vector {

// Return true if the operation is in VNNI layout.
// Optionally, the check can be constrained to a specific VNNI blocking factor.
bool isInVnniLayout(Operation *op, llvm::ArrayRef<AffineMap> indexingMaps,
                    std::optional<unsigned> blockingFactor = std::nullopt);

// Returns true if two contraction ops form a valid pair for VNNI packing.
// It verifies that both contractions share the appropriate operand, read from
// the same source buffer, and use constant indices that differ by 8 or 16.
bool validatePairVectorContract(vector::ContractionOp contractOp,
                                vector::ContractionOp pairContOp,
                                bool rhsHasMultipleNonUnitDims,
                                int64_t nonUnitDimValue);

// Walks backward from a value to find its originating vector read-like op
// (vector.transfer_read or vector.load), following scf.for iter-args but
// stopping at layout-transforming ops; returns the read op or nullptr.
Operation *traceToVectorReadLikeParentOperation(Value v);

// Recursively traces a value to find a downstream vector write-like op
// (vector.transfer_write or vector.store), crossing scf.for/yield but
// stopping at layout-altering ops. Returns nullptr if no vector writer/store
// ops or there are multiple users.
Operation *traceToVectorWriteLikeUserOperation(Value v);

// Packs the accumulators of two flat BF16 vector.contraction ops into a
// VNNI-packed layout and replaces the original accumulators to enable post-read
// packing transformations.
LogicalResult shuffleAfterReadLikeOp(PatternRewriter &rewriter, Operation *opA,
                                     Operation *opB,
                                     vector::ContractionOp contractA,
                                     vector::ContractionOp contractB,
                                     int64_t nonUnitDimAcc, VectorType accTy);

// Shuffles vectors produced by vector.contraction ops into a flat layout
// before they are written to memory.
LogicalResult shuffleBeforeWriteLikeOp(PatternRewriter &rewriter,
                                       Operation *opA, Operation *opB,
                                       int64_t nonUnitDimAcc, VectorType accTy);

} // namespace x86vector
} // namespace mlir

#endif // MLIR_DIALECT_X86VECTOR_UTILS_X86VECTORUTILS_H_
