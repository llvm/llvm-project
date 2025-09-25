//===- TransformGPU.cpp - C Interface for Transform GPU extension ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/GPU.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"

void mlirGPURegisterTransformDialectExtension(MlirDialectRegistry registry) {
  mlir::gpu::registerTransformDialectExtension(*unwrap(registry));
}
