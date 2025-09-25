//===- Complex.cpp - C Interface for Complex dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Complex.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Complex/IR/Complex.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Complex, complex,
                                      mlir::complex::ComplexDialect)
