//===- Polynomial.cpp - C Interface for Polynomial dialect ----------------===//
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

MlirIntMonomial mlirPolynomialGetIntMonomial(int64_t coeff, uint64_t expo) {
  return wrap(new mlir::polynomial::IntMonomial(coeff, expo));
}

int64_t mlirPolynomialIntMonomialGetCoefficient(MlirIntMonomial intMonomial) {
  return unwrap(intMonomial)
      ->getCoefficient()
      .getLimitedValue(/*Limit = UINT64_MAX*/);
}

uint64_t mlirPolynomialIntMonomialGetExponent(MlirIntMonomial intMonomial) {
  return unwrap(intMonomial)
      ->getExponent(/*Limit = UINT64_MAX*/)
      .getLimitedValue();
}