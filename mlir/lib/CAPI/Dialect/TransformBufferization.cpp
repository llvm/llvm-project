//==- TransformBufferization.cpp - C Interface for bufferization extension -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Bufferization.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"

void mlirBufferizationRegisterTransformDialectExtension(
    MlirDialectRegistry registry) {
  mlir::bufferization::registerTransformDialectExtension(*unwrap(registry));
}
