//===- VerificationUtils.h - Common verification utilities ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines common verification utilities that can be shared
// across multiple MLIR dialects. These utilities help reduce code duplication
// for common verification patterns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UTILS_VERIFICATIONUTILS_H
#define MLIR_DIALECT_UTILS_VERIFICATIONUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

/// Verify that the number of dynamic size operands matches the number of
/// dynamic dimensions in the shaped type. Returns failure and emits an error
/// if the counts don't match.
LogicalResult verifyDynamicDimensionCount(Operation *op, ShapedType type,
                                          ValueRange dynamicSizes);

/// Verify that two shaped types have matching ranks. Returns failure and emits
/// an error if ranks don't match. Unranked types are considered compatible.
LogicalResult verifyRanksMatch(Operation *op, ShapedType lhs, ShapedType rhs,
                               StringRef lhsName, StringRef rhsName);

} // namespace mlir

#endif // MLIR_DIALECT_UTILS_VERIFICATIONUTILS_H
