//===- Intrinsics.h - MLIR EDSC Intrinsics for Linalg -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_LOOPOPS_EDSC_INTRINSICS_H_
#define MLIR_DIALECT_LOOPOPS_EDSC_INTRINSICS_H_

#include "mlir/Dialect/LoopOps/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"

namespace mlir {
namespace edsc {
namespace intrinsics {
using loop_yield = OperationBuilder<loop::YieldOp>;

} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_LOOPOPS_EDSC_INTRINSICS_H_
