//===- ArmSVEVectorTransformOps.h - Vector transform ops --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARM_SVE_VECTOR_TRANSFORMOPS_H
#define MLIR_DIALECT_ARM_SVE_VECTOR_TRANSFORMOPS_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// ArmSVE Vector Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace arm_sve {
void registerTransformDialectExtension(DialectRegistry &registry);

} // namespace arm_sve
} // namespace mlir

#endif // MLIR_DIALECT_ARM_SVE_VECTOR_TRANSFORMOPS_H
