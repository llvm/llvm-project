//===- PolynomialOps.h - Ops for the Polynomial dialect ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_
#define MLIR_INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_

#include "PolynomialDialect.h"
#include "PolynomialTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Polynomial/IR/Polynomial.h.inc"

#endif // MLIR_INCLUDE_MLIR_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_
