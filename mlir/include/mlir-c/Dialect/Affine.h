//===-- mlir-c/Dialect/Affine.h - C API for Affine Dialect --------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_AFFINE_H
#define MLIR_C_DIALECT_AFFINE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Affine, affine);

#ifdef __cplusplus
}
#endif

// Currently these are opt-in
#define GET_ENUM_CAPI_DECLS
#define GET_ENUM_ATTR_CAPI_DECLS
#include "mlir/Dialect/Affine/IR/AffineOpsCAPIEnumAttrs.h.inc"
#define GET_TYPE_CAPI_DECLS
#include "mlir/Dialect/Affine/IR/AffineOpsCAPITypes.h.inc"
#define GET_ATTR_CAPI_DECLS
#include "mlir/Dialect/Affine/IR/AffineOpsCAPIAttrs.h.inc"

#include "mlir/Dialect/Affine/Transforms/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_AFFINE_H
