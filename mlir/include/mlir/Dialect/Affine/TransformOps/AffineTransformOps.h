//===- AffineTransformOps.h - Affine transformation ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_TRANSFORMOPS_AFFINETRANSFORMOPS_H
#define MLIR_DIALECT_AFFINE_TRANSFORMOPS_AFFINETRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
class AffineForOp;
namespace func {
class FuncOp;
} // namespace func
namespace affine {
class ForOp;
} // namespace affine
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace affine {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_TRANSFORMOPS_AFFINETRANSFORMOPS_H
