//===- TransformTensor.cpp - C Interface for Transform tensor extensio ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Tensor.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"

void mlirTensorRegisterTransformDialectExtension(MlirDialectRegistry registry) {
  mlir::tensor::registerTransformDialectExtension(*unwrap(registry));
}
