//===- MathTransformOps.h - Math transform ops ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MATH_TRANSFORMOPS_MATHTRANSFORMOPS_H
#define MLIR_DIALECT_MATH_TRANSFORMOPS_MATHTRANSFORMOPS_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// Math Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Math/TransformOps/MathTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace math {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace math
} // namespace mlir

#endif // MLIR_DIALECT_MATH_TRANSFORMOPS_MATHTRANSFORMOPS_H
