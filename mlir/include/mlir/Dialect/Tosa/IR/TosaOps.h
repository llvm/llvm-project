//===-- TosaOps.h - TOSA dialect operation definitions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the TOSA Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_IR_TOSAOPS_H
#define MLIR_DIALECT_TOSA_IR_TOSAOPS_H

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// TOSA dialect and structs includes.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOpsDialect.h.inc"

namespace mlir {
class PatternRewriter;

namespace tosa {

#include "mlir/Dialect/Tosa/IR/TosaInterfaces.h.inc"

} // namespace tosa
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOps.h.inc"

#endif // MLIR_DIALECT_TOSA_IR_TOSAOPS_H
