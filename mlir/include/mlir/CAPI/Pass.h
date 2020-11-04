//===- IR.h - C API Utils for Core MLIR classes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// core MLIR classes. This file should not be included from C++ code other than
// C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_PASS_H
#define MLIR_CAPI_PASS_H

#include "mlir-c/Pass.h"

#include "mlir/CAPI/Wrap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

DEFINE_C_API_PTR_METHODS(MlirPass, mlir::Pass)
DEFINE_C_API_PTR_METHODS(MlirPassManager, mlir::PassManager)
DEFINE_C_API_PTR_METHODS(MlirOpPassManager, mlir::OpPassManager)

#endif // MLIR_CAPI_PASS_H
