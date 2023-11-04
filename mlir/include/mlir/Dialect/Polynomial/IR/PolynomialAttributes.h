//===- PolynomialAttributes.h - Attributes for the Polynomial dialect -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_H_
#define INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_H_

#include "Polynomial.h"
#include "PolynomialDialect.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.h.inc"

#endif // INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_H_
