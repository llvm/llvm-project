//===- XeGPUTransformOps.h - XeGPU transformation ops -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_TRANSFORMOPS_XEGPUTRANSFORMOPS_H
#define MLIR_DIALECT_XEGPU_TRANSFORMOPS_XEGPUTRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace xegpu {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_TRANSFORMOPS_XEGPUTRANSFORMOPS_H
