//===- TransformVector.cpp - C Interface for Transform Vector extension ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Vector.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"

void mlirVectorRegisterTransformDialectExtension(MlirDialectRegistry registry) {
  mlir::vector::registerTransformDialectExtension(*unwrap(registry));
}
