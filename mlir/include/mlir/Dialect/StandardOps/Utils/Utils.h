//===- Utils.h - General transformation utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// the StandardOps dialect. These are not passes by themselves but are used
// either by passes, optimization sequences, or in turn by other transformation
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARDOPS_UTILS_UTILS_H
#define MLIR_DIALECT_STANDARDOPS_UTILS_UTILS_H

#include "mlir/IR/Value.h"

namespace mlir {

class Location;
class OpBuilder;

/// Given an operation, retrieves the value of each dynamic dimension through
/// constructing the necessary DimOp operators.
SmallVector<Value, 4> getDynOperands(Location loc, Value val, OpBuilder &b);

} // end namespace mlir

#endif // MLIR_DIALECT_STANDARDOPS_UTILS_UTILS_H
