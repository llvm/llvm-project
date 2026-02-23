//===- AMXTransformOps.h - AMX transform ops --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMX_TRANSFORMOPS_AMXTRANSFORMOPS_H
#define MLIR_DIALECT_AMX_TRANSFORMOPS_AMXTRANSFORMOPS_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// AMX Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/AMX/TransformOps/AMXTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace amx {
void registerTransformDialectExtension(DialectRegistry &registry);

} // namespace amx
} // namespace mlir

#endif // MLIR_DIALECT_AMX_TRANSFORMOPS_AMXTRANSFORMOPS_H
