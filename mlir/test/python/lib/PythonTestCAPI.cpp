//===- PythonTestCAPI.cpp - C API for the PythonTest dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonTestCAPI.h"
#include "PythonTestDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PythonTest, python_test,
                                      python_test::PythonTestDialect)
