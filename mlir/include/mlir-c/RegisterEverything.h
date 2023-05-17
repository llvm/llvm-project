//===-- mlir-c/RegisterEverything.h - Register all MLIR entities --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header contains registration entry points for MLIR upstream dialects
// and passes. Downstream projects typically will not want to use this unless
// if they don't care about binary size or build bloat and just wish access
// to the entire set of upstream facilities. For those that do care, they
// should use registration functions specific to their project.
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_REGISTER_EVERYTHING_H
#define MLIR_C_REGISTER_EVERYTHING_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Appends all upstream dialects and extensions to the dialect registry.
MLIR_CAPI_EXPORTED void mlirRegisterAllDialects(MlirDialectRegistry registry);

/// Register all translations to LLVM IR for dialects that can support it.
MLIR_CAPI_EXPORTED void mlirRegisterAllLLVMTranslations(MlirContext context);

/// Register all compiler passes of MLIR.
MLIR_CAPI_EXPORTED void mlirRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REGISTER_EVERYTHING_H
