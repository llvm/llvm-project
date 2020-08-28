//===- PDLInterp.h - PDL Interpreter dialect --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interpreter dialect for the PDL pattern descriptor
// language.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDLINTERP_IR_PDLINTERP_H_
#define MLIR_DIALECT_PDLINTERP_IR_PDLINTERP_H_

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace pdl_interp {
//===----------------------------------------------------------------------===//
// PDLInterp Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDLInterp/IR/PDLInterpOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// PDLInterp Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.h.inc"

} // end namespace pdl_interp
} // end namespace mlir

#endif // MLIR_DIALECT_PDLINTERP_IR_PDLINTERP_H_
