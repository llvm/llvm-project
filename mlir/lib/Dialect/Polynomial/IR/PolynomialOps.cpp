//===- PolynomialOps.cpp - Polynomial dialect ops ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polynomial/IR/Polynomial.h"

using namespace mlir;
using namespace mlir::polynomial;

#define GET_OP_CLASSES
#include "mlir/Dialect/Polynomial/IR/Polynomial.cpp.inc"
