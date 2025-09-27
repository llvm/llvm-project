//===- RegisterAllExtensions.cpp - Register all MLIR extensions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/RegisterAllExtensions.h"
#include "mlir/CAPI/IR.h"
#include "mlir/InitAllExtensions.h"

using namespace mlir;

void mlirRegisterAllExtensions(MlirDialectRegistry mlirRegistry) {
  mlir::DialectRegistry *registry = unwrap(mlirRegistry);
  mlir::registerAllExtensions(*registry);
}
