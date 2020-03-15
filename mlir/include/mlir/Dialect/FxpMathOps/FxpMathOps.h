//===- FxpMathOps.h - Fixed point ops ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FXPMATHOPS_FXPMATHOPS_H_
#define MLIR_DIALECT_FXPMATHOPS_FXPMATHOPS_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffects.h"

namespace mlir {
namespace fxpmath {

#include "mlir/Dialect/FxpMathOps/FxpMathOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/FxpMathOps/FxpMathOps.h.inc"

} // namespace fxpmath
} // namespace mlir

#endif // MLIR_DIALECT_FXPMATHOPS_FXPMATHOPS_H_
