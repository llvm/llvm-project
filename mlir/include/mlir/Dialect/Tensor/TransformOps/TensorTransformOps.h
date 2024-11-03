//===- TensorTransformOps.h - Tensor transformation ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H
#define MLIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class DialectRegistry;

namespace tensor {
void registerTransformDialectExtension(DialectRegistry &registry);
void registerFindPayloadReplacementOpInterfaceExternalModels(
    DialectRegistry &registry);
} // namespace tensor
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h.inc"

#endif // MLIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H
