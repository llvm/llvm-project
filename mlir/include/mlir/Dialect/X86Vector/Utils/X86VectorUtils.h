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

bool validatePairVectorContract(vector::ContractionOp contractOp,
                                vector::ContractionOp pairContOp,
                                bool rhsHasMultipleNonUnitDims,
                                int64_t nonUnitDimValue);

Operation *traceToVectorReadLikeParentOperation(mlir::Value v);

Operation *traceToVectorWriteLikeUserOperation(mlir::Value v);

void shuffleBeforeWriteLikeOp(PatternRewriter &rewriter, Operation *op,
                              Operation *op1, int64_t nonUnitDimAcc,
                              VectorType accTy);

void shuffleAfterReadLikeOp(PatternRewriter &rewriter, Operation *op,
                            Operation *op1, vector::ContractionOp contractOp,
                            vector::ContractionOp pairContractOp,
                            int64_t nonUnitDimAcc, VectorType accTy);

void shuffleNonUnitDimOperand(PatternRewriter &rewriter, Operation *op,
                              Operation *op1, vector::ContractionOp contractOp,
                              vector::ContractionOp pairContractOp,
                              int64_t nonUnitDimAcc, VectorType Ty);

} // namespace x86vector
} // namespace mlir

#endif // MLIR_DIALECT_X86VECTOR_UTILS_X86VECTORUTILS_H_
