//===- Math.h - Math dialect --------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MATH_IR_MATH_H_
#define MLIR_DIALECT_MATH_IR_MATH_H_

#include "mlir/Dialect/Arith/IR/ArithOpInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

//===----------------------------------------------------------------------===//
// Math Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Math/IR/MathOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Math Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Math/IR/MathOps.h.inc"

#endif // MLIR_DIALECT_MATH_IR_MATH_H_
