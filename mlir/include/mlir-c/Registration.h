/*===-- mlir-c/Registration.h - Registration functions for MLIR ---*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef MLIR_C_REGISTRATION_H
#define MLIR_C_REGISTRATION_H

#ifdef __cplusplus
extern "C" {
#endif

/** Registers all dialects known to core MLIR with the system. This must be
 * called before creating an MlirContext if it needs access to the registered
 * dialects. */
void mlirRegisterAllDialects();

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REGISTRATION_H
