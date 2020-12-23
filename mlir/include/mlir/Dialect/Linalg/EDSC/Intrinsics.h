//===- Intrinsics.h - MLIR EDSC Intrinsics for Linalg -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_LINALG_EDSC_INTRINSICS_H_
#define MLIR_DIALECT_LINALG_EDSC_INTRINSICS_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

using linalg_copy = OperationBuilder<linalg::CopyOp>;
using linalg_dot = OperationBuilder<linalg::DotOp>;
using linalg_fill = OperationBuilder<linalg::FillOp>;
using linalg_init_tensor = ValueBuilder<linalg::InitTensorOp>;
using linalg_matmul = OperationBuilder<linalg::MatmulOp>;
using linalg_matvec = OperationBuilder<linalg::MatvecOp>;
using linalg_vecmat = OperationBuilder<linalg::VecmatOp>;
using linalg_range = ValueBuilder<linalg::RangeOp>;
using linalg_reshape = ValueBuilder<linalg::ReshapeOp>;
using linalg_slice = ValueBuilder<linalg::SliceOp>;
using linalg_yield = OperationBuilder<linalg::YieldOp>;

} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_EDSC_INTRINSICS_H_
