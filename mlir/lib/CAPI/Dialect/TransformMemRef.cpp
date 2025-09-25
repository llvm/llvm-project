//===- TransformMemRef.cpp - C Interface for Transform MemRef extension ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/MemRef.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"

void mlirMemRefRegisterTransformDialectExtension(MlirDialectRegistry registry) {
  mlir::memref::registerTransformDialectExtension(*unwrap(registry));
}
