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

#ifndef MLIR_CAPI_EXECUTIONENGINE_H
#define MLIR_CAPI_EXECUTIONENGINE_H

#include "mlir-c/ExecutionEngine.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

DEFINE_C_API_PTR_METHODS(MlirExecutionEngine, mlir::ExecutionEngine)

#endif // MLIR_CAPI_EXECUTIONENGINE_H
