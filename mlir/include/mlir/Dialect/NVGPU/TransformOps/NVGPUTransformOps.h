//===- NVGPUTransformOps.h - NVGPU transform ops ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_NVGPU_TRANSFORMOPS_NVGPUTRANSFORMOPS_H
#define MLIR_DIALECT_NVGPU_TRANSFORMOPS_NVGPUTRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
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

namespace nvgpu {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace nvgpu
} // namespace mlir

//===----------------------------------------------------------------------===//
// NVGPU Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h.inc"

#endif // MLIR_DIALECT_NVGPU_TRANSFORMOPS_NVGPUTRANSFORMOPS_H
