//===- Arithmetic.h - Arithmetic dialect --------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_ARITHMETIC_IR_ARITHMETIC_H_
#define MLIR_DIALECT_ARITHMETIC_IR_ARITHMETIC_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// ArithmeticDialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/ArithmeticOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Arithmetic Dialect Enum Attributes
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/ArithmeticOpsEnums.h.inc"

//===----------------------------------------------------------------------===//
// Arithmetic Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.h.inc"

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace arith {

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known integer
/// comparison predicates.
bool applyCmpPredicate(arith::CmpIPredicate predicate, const APInt &lhs,
                       const APInt &rhs);

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known floating point
/// comparison predicates.
bool applyCmpPredicate(arith::CmpFPredicate predicate, const APFloat &lhs,
                       const APFloat &rhs);

} // end namespace arith
} // end namespace mlir

#endif // MLIR_DIALECT_ARITHMETIC_IR_ARITHMETIC_H_
