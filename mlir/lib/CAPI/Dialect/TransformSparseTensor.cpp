//===- TransformSparseTensor.cpp - C Interface for SparseTensor extension -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/SparseTensor.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.h"

void mlirSparseTensorRegisterTransformDialectExtension(
    MlirDialectRegistry registry) {
  mlir::sparse_tensor::registerTransformDialectExtension(*unwrap(registry));
}
