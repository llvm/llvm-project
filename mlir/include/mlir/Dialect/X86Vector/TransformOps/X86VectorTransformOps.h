//===- VectorTransformOps.h - Vector transform ops --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_X86VECTOR_TRANSFORMOPS_X86VECTORTRANSFORMOPS_H
#define MLIR_DIALECT_X86VECTOR_TRANSFORMOPS_X86VECTORTRANSFORMOPS_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace x86vector {
} // namespace vector
} // namespace mlir

//===----------------------------------------------------------------------===//
// Vector Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/X86Vector/TransformOps/X86VectorTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace x86vector {
void registerTransformDialectExtension(DialectRegistry &registry);

} // namespace vector
} // namespace mlir

#endif // MLIR_DIALECT_X86VECTOR_TRANSFORMOPS_X86VECTORTRANSFORMOPS_H
