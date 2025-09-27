//===-- mlir-c/RegisterAllExternalModels.h - Register all MLIR model ----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_REGISTER_EXTERNAL_MODELS_H
#define MLIR_C_REGISTER_EXTERNAL_MODELS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Register all compiler External models of MLIR.
MLIR_CAPI_EXPORTED void
mlirRegisterAllExternalModels(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REGISTER_EXTERNAL_MODELS_H
