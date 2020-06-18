//===- Intrinsics.h - MLIR EDSC Intrinsics for AffineOps --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_AFFINE_EDSC_INTRINSICS_H_
#define MLIR_DIALECT_AFFINE_EDSC_INTRINSICS_H_

#include "mlir/Dialect/Affine/EDSC/Builders.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

using affine_apply = ValueBuilder<AffineApplyOp>;
using affine_if = OperationBuilder<AffineIfOp>;
using affine_load = ValueBuilder<AffineLoadOp>;
using affine_min = ValueBuilder<AffineMinOp>;
using affine_max = ValueBuilder<AffineMaxOp>;
using affine_store = OperationBuilder<AffineStoreOp>;

/// Provide an index notation around affine_load and affine_store.
using AffineIndexedValue = TemplatedIndexedValue<affine_load, affine_store>;

} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_STANDARDOPS_EDSC_INTRINSICS_H_
