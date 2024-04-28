//===- PolynomialDialect.cpp - Polynomial dialect ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polynomial/IR/Polynomial.h"

#include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "mlir/Dialect/Polynomial/IR/PolynomialOps.h"
#include "mlir/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::polynomial;

#include "mlir/Dialect/Polynomial/IR/PolynomialDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Polynomial/IR/PolynomialTypes.cpp.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/Polynomial/IR/Polynomial.cpp.inc"

void PolynomialDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Polynomial/IR/PolynomialTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Polynomial/IR/Polynomial.cpp.inc"
      >();
}
