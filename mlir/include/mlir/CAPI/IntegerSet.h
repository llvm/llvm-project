//===- IntegerSet.h - C API Utils for Integer Sets --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// MLIR IntegerSets. This file should not be included from C++ code other than C
// API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_INTEGERSET_H
#define MLIR_CAPI_INTEGERSET_H

#include "mlir-c/IntegerSet.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/IntegerSet.h"

DEFINE_C_API_METHODS(MlirIntegerSet, mlir::IntegerSet)

#endif // MLIR_CAPI_INTEGERSET_H
