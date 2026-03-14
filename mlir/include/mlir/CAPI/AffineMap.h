//===- AffineMap.h - C API Utils for Affine Maps ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// MLIR Affine maps. This file should not be included from C++ code other than
// C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_AFFINEMAP_H
#define MLIR_CAPI_AFFINEMAP_H

#include "mlir-c/AffineMap.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/AffineMap.h"

DEFINE_C_API_METHODS(MlirAffineMap, mlir::AffineMap)

#endif // MLIR_CAPI_AFFINEMAP_H
