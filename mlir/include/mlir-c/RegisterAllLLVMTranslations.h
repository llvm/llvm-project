//===-- mlir-c/RegisterAllLLVMTranslations.h - Register all LLVM translations //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_REGISTER_LLVM_TRANSLATIONS_H
#define MLIR_C_REGISTER_LLVM_TRANSLATIONS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Register all LLVM translations of MLIR.
MLIR_CAPI_EXPORTED void mlirRegisterAllLLVMTranslations(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REGISTER_LLVM_TRANSLATIONS_H
