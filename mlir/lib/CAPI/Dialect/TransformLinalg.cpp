//===- TransformLinalg.cpp - C Interface for Transform Linalg extension ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Linalg.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"

void mlirLinalgRegisterTransformDialectExtension(MlirDialectRegistry registry) {
  mlir::linalg::registerTransformDialectExtension(*unwrap(registry));
}
