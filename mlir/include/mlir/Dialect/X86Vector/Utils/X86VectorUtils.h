//===- X86VectorUtils.h - X86Vector Utilities -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_X86VECTOR_UTILS_X86VECTORUTILS_H_
#define MLIR_DIALECT_X86VECTOR_UTILS_X86VECTORUTILS_H_

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
class Type;
class ShapedType;
class OpOperand;
class AffineDimExpr;
class AffineMap;
class VectorType;
class Operation;

namespace x86vector {
enum class VnniOperandRank {
  TRANSPOSE = 3,
  GEMM = 3,
  BRGEMM_INS = 4,
  BRGEMM_OUTS = 3
};

// Return true if the operation is in VNNI layout.
// Optionally, the check can be constrained to a specific VNNI blocking factor.
bool isInVnniLayout(Operation *op, llvm::ArrayRef<AffineMap> indexingMaps,
                    std::optional<unsigned> blockingFactor = std::nullopt);

} // namespace x86vector
} // namespace mlir

#endif // MLIR_DIALECT_X86VECTOR_UTILS_X86VECTORUTILS_H_
