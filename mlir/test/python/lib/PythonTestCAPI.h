//===- PythonTestCAPI.h - C API for the PythonTest dialect ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_PYTHON_LIB_PYTHONTESTCAPI_H
#define MLIR_TEST_PYTHON_LIB_PYTHONTESTCAPI_H

#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PythonTest, python_test);

#ifdef __cplusplus
}
#endif

#endif // MLIR_TEST_PYTHON_LIB_PYTHONTESTCAPI_H
