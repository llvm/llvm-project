//===- BufferizationTypeInterfaces.h - Type Interfaces ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONTYPEINTERFACES_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONTYPEINTERFACES_H_

//===----------------------------------------------------------------------===//
// Bufferization Type Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"

namespace mlir::bufferization {
struct BufferizationOptions;
class BufferLikeType;
} // namespace mlir::bufferization

#include "mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h.inc"

#endif // MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONTYPEINTERFACES_H_
