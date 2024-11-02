//===- TransformTypes.h - Transform dialect types ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMTYPES_H
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMTYPES_H

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class DiagnosedSilenceableFailure;
class Operation;
class Type;
} // namespace mlir

#include "mlir/Dialect/Transform/IR/TransformTypeInterfaces.h.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Transform/IR/TransformTypes.h.inc"

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMTYPES_H
