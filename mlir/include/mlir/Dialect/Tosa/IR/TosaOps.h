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

#ifndef DIALECT_TOSA_IR_TOSA_OPS_H
#define DIALECT_TOSA_IR_TOSA_OPS_H

#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// TOSA dialect and structs includes.
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Tosa/IR/TosaOpsDialect.h.inc"
#include "mlir/Dialect/Tosa/IR/TosaStructs.h.inc"

namespace mlir {
namespace tosa {

#include "mlir/Dialect/Tosa/IR/TosaInterfaces.h.inc"

} // end namespace tosa
} // end namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOps.h.inc"

#endif // TOSA_OPS_H
