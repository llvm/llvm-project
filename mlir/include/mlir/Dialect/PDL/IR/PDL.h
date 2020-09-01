//===- PDL.h - Pattern Descriptor Language Dialect --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dialect for the Pattern Descriptor Language.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDL_IR_PDL_H_
#define MLIR_DIALECT_PDL_IR_PDL_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace pdl {
//===----------------------------------------------------------------------===//
// PDL Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDL/IR/PDLOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// PDL Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PDL/IR/PDLOps.h.inc"

} // end namespace pdl
} // end namespace mlir

#endif // MLIR_DIALECT_PDL_IR_PDL_H_
