//===-- mlir-c/Dialect/SCF.h - C API for SCF dialect --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_SCF_H
#define MLIR_C_DIALECT_SCF_H

#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SCF, scf);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_SCF_H
