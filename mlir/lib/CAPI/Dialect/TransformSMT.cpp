//===- TransformSMT.cpp - C Interface for Transform SMT extension ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/SMT.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Transform/SMTExtension/SMTExtension.h"

void mlirSMTRegisterTransformDialectExtension(MlirDialectRegistry registry) {
  mlir::transform::registerSMTExtension(*unwrap(registry));
}
