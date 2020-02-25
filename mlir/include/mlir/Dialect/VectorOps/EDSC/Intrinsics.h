//===- Intrinsics.h - MLIR EDSC Intrinsics for VectorOps --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_VECTOROPS_EDSC_INTRINSICS_H_
#define MLIR_DIALECT_VECTOROPS_EDSC_INTRINSICS_H_

#include "mlir/Dialect/VectorOps/EDSC/Builders.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

using vector_contract = ValueBuilder<vector::ContractionOp>;
using vector_print = OperationBuilder<vector::PrintOp>;

} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_VECTOROPS_EDSC_INTRINSICS_H_
