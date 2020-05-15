//===- SCFOps.h - Structured Control Flow -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines structured control flow operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_H_
#define MLIR_DIALECT_SCF_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace scf {

#include "mlir/Dialect/SCF/SCFOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/SCF/SCFOps.h.inc"

// Insert `loop.yield` at the end of the only region's only block if it
// does not have a terminator already.  If a new `loop.yield` is inserted,
// the location is specified by `loc`. If the region is empty, insert a new
// block first.
void ensureLoopTerminator(Region &region, Builder &builder, Location loc);

/// Returns the loop parent of an induction variable. If the provided value is
/// not an induction variable, then return nullptr.
ForOp getForInductionVarOwner(Value val);

/// Returns the parallel loop parent of an induction variable. If the provided
// value is not an induction variable, then return nullptr.
ParallelOp getParallelForInductionVarOwner(Value val);

} // end namespace scf
} // end namespace mlir
#endif // MLIR_DIALECT_SCF_H_
