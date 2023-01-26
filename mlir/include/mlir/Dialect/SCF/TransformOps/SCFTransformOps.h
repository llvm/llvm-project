//===- SCFTransformOps.h - SCF transformation ops ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H
#define MLIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace scf {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H
