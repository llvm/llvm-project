//===-- mlir-c/Dialect/Bufferization.h - C API for Bufferization dialect --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Bufferization dialect. A dialect should be registered with a context to make
// it available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_BUFFERIZATION_H
#define MLIR_C_DIALECT_BUFFERIZATION_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Bufferization, bufferization);

MLIR_CAPI_EXPORTED void mlirBufferizationRegisterTransformDialectExtension(
    MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_BUFFERIZATION_H
