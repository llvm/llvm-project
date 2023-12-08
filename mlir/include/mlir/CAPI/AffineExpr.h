//===- AffineExpr.h - C API Utils for Affine Expressions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// MLIR Affine Expression. This file should not be included from C++ code other
// than C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_AFFINEEXPR_H
#define MLIR_CAPI_AFFINEEXPR_H

#include "mlir-c/AffineExpr.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/AffineExpr.h"

DEFINE_C_API_METHODS(MlirAffineExpr, mlir::AffineExpr)

#endif // MLIR_CAPI_AFFINEEXPR_H
