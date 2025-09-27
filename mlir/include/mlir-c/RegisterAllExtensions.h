//===-- mlir-c/RegisterAllExtensions.h - Register all MLIR Extensions ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_REGISTER_Extensions_H
#define MLIR_C_REGISTER_Extensions_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Register all compiler Extensions of MLIR.
MLIR_CAPI_EXPORTED void mlirRegisterAllExtensions(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REGISTER_Extensions_H
