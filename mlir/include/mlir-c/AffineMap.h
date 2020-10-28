/*===-- mlir-c/AffineMap.h - C API for MLIR Affine maps -----------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef MLIR_C_AFFINEMAP_H
#define MLIR_C_AFFINEMAP_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

DEFINE_C_API_STRUCT(MlirAffineMap, const void);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_AFFINEMAP_H
