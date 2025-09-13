//===-- mlir-c/Dialect/Polynomial.h - C API for Polynomial --------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_POLYNOMIAL_H
#define MLIR_C_DIALECT_POLYNOMIAL_H

#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Polynomial/IR/Polynomial.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Polynomial, polynomial);

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirIntMonomial, void);

#undef DEFINE_C_API_STRUCT

DEFINE_C_API_PTR_METHODS(MlirIntMonomial, mlir::polynomial::IntMonomial);

MLIR_CAPI_EXPORTED MlirIntMonomial mlirPolynomialGetIntMonomial(int64_t coeff,
                                                                uint64_t expo);

MLIR_CAPI_EXPORTED int64_t
mlirPolynomialIntMonomialGetCoefficient(MlirIntMonomial intMonomial);

MLIR_CAPI_EXPORTED uint64_t
mlirPolynomialIntMonomialGetExponent(MlirIntMonomial intMonomial);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_POLYNOMIAL_H
