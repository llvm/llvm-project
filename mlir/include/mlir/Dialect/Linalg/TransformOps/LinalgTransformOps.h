//===- LinalgTransformOps.h - Linalg transform ops --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H
#define MLIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace linalg {
class LinalgOp;
} // namespace linalg
} // namespace mlir

//===----------------------------------------------------------------------===//
// Linalg Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace linalg {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H
