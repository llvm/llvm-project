//===- ArithAttributes.h - Arith dialect attributes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_IR_ARITHATTRIBUTES_H
#define MLIR_DIALECT_ARITH_IR_ARITHATTRIBUTES_H

#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringExtras.h"

#include "mlir/Dialect/Arith/IR/ArithOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Arith/IR/ArithOpsAttributes.h.inc"

#endif // MLIR_DIALECT_ARITH_IR_ARITHATTRIBUTES_H
