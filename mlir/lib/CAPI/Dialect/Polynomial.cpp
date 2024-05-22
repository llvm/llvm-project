//===- Polynomial.cpp - C Interface for Polynomial dialect
//--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Polynomial.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Polynomial/IR/PolynomialDialect.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(polynomial, polynomial,
                                      polynomial::PolynomialDialect)
