//===- AMDGPUTransformOps.h - AMDGPU transform ops ---------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDGPU_TRANSFORMOPS_AMDGPUTRANSFORMOPS_H
#define MLIR_DIALECT_AMDGPU_TRANSFORMOPS_AMDGPUTRANSFORMOPS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"

namespace mlir {
namespace transform {
class TransformHandleTypeInterface;
} // namespace transform
} // namespace mlir

namespace mlir {
class DialectRegistry;

namespace linalg {
class LinalgOp;
} // namespace linalg

namespace scf {
class ForOp;
} // namespace scf

namespace amdgpu {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace amdgpu
} // namespace mlir

//===----------------------------------------------------------------------===//
// AMDGPU Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/AMDGPU/TransformOps/AMDGPUTransformOps.h.inc"

#endif // MLIR_DIALECT_AMDGPU_TRANSFORMOPS_AMDGPUTRANSFORMOPS_H
