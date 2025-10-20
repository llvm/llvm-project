//===- OpenACCUtils.h - OpenACC Utilities -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCUTILS_H_
#define MLIR_DIALECT_OPENACC_OPENACCUTILS_H_

#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace mlir {
namespace acc {

/// Used to obtain the enclosing compute construct operation that contains
/// the provided `region`. Returns nullptr if no compute construct operation
/// is found. The returned operation is one of types defined by
/// `ACC_COMPUTE_CONSTRUCT_OPS`.
mlir::Operation *getEnclosingComputeOp(mlir::Region &region);

/// Returns true if this value is only used by `acc.private` operations in the
/// `region`.
bool isOnlyUsedByPrivateClauses(mlir::Value val, mlir::Region &region);

/// Returns true if this value is only used by `acc.reduction` operations in
/// the `region`.
bool isOnlyUsedByReductionClauses(mlir::Value val, mlir::Region &region);

/// Looks for an OpenACC default attribute on the current operation `op` or in
/// a parent operation which encloses `op`. This is useful because OpenACC
/// specification notes that a visible default clause is the nearest default
/// clause appearing on the compute construct or a lexically containing data
/// construct.
std::optional<ClauseDefaultValue> getDefaultAttr(mlir::Operation *op);

/// Get the type category of an OpenACC variable.
mlir::acc::VariableTypeCategory getTypeCategory(mlir::Value var);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILS_H_
