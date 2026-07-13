//===- SMTTypes.h - SMT dialect types ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SMT_IR_SMTTYPES_H
#define MLIR_DIALECT_SMT_IR_SMTTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SMT/IR/SMTTypes.h.inc"

namespace mlir {
namespace smt {

/// Returns whether the given type is an SMT value type.
bool isAnySMTValueType(mlir::Type type);

/// Returns whether the given type is an SMT value type (excluding functions).
bool isAnyNonFuncSMTValueType(mlir::Type type);

} // namespace smt
} // namespace mlir

#endif // MLIR_DIALECT_SMT_IR_SMTTYPES_H
