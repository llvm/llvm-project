//===- TransformAffine.cpp - C Interface for Transform affine extension ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Affine.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"

MLIR_CAPI_EXPORTED void
mlirAffineRegisterTransformDialectExtension(MlirDialectRegistry registry) {
  mlir::affine::registerTransformDialectExtension(*unwrap(registry));
}
