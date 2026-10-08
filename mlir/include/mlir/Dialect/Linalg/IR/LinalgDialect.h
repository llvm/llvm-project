//===- LinalgDialect.h - Linalg dialect declaration ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_IR_LINALGDIALECT_H
#define MLIR_DIALECT_LINALG_IR_LINALGDIALECT_H

#include "mlir/IR/Dialect.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
class ImplicitLocOpBuilder;
} // namespace mlir

#include "mlir/Dialect/Linalg/IR/LinalgOpsDialect.h.inc"

#endif // MLIR_DIALECT_LINALG_IR_LINALGDIALECT_H
