//===- Transform.h - C API Utils for Transform dialect ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// the Transform dialect. This file should not be included from C++ code other
// than C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_DIALECT_TRANSFORM_H
#define MLIR_CAPI_DIALECT_TRANSFORM_H

#include "mlir-c/Dialect/Transform.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

DEFINE_C_API_PTR_METHODS(MlirTransformRewriter,
                         mlir::transform::TransformRewriter)
DEFINE_C_API_PTR_METHODS(MlirTransformResults,
                         mlir::transform::TransformResults)
DEFINE_C_API_PTR_METHODS(MlirTransformState, mlir::transform::TransformState)

#endif // MLIR_CAPI_DIALECT_TRANSFORM_H
