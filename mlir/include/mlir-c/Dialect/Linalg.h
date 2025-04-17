//===-- mlir-c/Dialect/Linalg.h - C API for Linalg dialect -------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_LINALG_H
#define MLIR_C_DIALECT_LINALG_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Apply the special region builder for the builtin named Linalg op.
/// Assert that `mlirOp` is a builtin named Linalg op.
MLIR_CAPI_EXPORTED void
mlirLinalgFillBuiltinNamedOpRegion(MlirOperation mlirOp);

MLIR_CAPI_EXPORTED bool mlirLinalgIsAContractionOp(MlirOperation op);

typedef struct MlirLinalgContractionDimensions {
  MlirAttribute batch;
  MlirAttribute m;
  MlirAttribute n;
  MlirAttribute k;
} MlirLinalgContractionDimensions;

MLIR_CAPI_EXPORTED MlirLinalgContractionDimensions
mlirLinalgInferContractionDimensions(MlirOperation op);

MLIR_CAPI_EXPORTED bool mlirLinalgIsAConvolutionOp(MlirOperation op);

typedef struct MlirLinalgConvolutionDimensions {
  MlirAttribute batch;
  MlirAttribute outputImage;
  MlirAttribute outputChannel;
  MlirAttribute filterLoop;
  MlirAttribute inputChannel;
  MlirAttribute depth;
  MlirAttribute strides;
  MlirAttribute dilations;
} MlirLinalgConvolutionDimensions;

MLIR_CAPI_EXPORTED MlirLinalgConvolutionDimensions
mlirLinalgInferConvolutionDimensions(MlirOperation op);

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Linalg, linalg);

#ifdef __cplusplus
}
#endif

#include "mlir/Dialect/Linalg/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_LINALG_H
