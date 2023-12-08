//===- BufferizationTransformOps.h - Buff. transf. ops ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMOPS_BUFFERIZATIONTRANSFORMOPS_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMOPS_BUFFERIZATIONTRANSFORMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace tensor {
class EmptyOp;
} // namespace tensor
} // namespace mlir

//===----------------------------------------------------------------------===//
// Bufferization Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace bufferization {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMOPS_BUFFERIZATIONTRANSFORMOPS_H
