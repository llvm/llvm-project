//===- Intrinsics.h - MLIR EDSC Intrinsics for Vector -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_VECTOR_EDSC_INTRINSICS_H_
#define MLIR_DIALECT_VECTOR_EDSC_INTRINSICS_H_

#include "mlir/Dialect/Vector/EDSC/Builders.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

using vector_broadcast = ValueBuilder<vector::BroadcastOp>;
using vector_contract = ValueBuilder<vector::ContractionOp>;
using vector_extract = ValueBuilder<vector::ExtractOp>;
using vector_extract_element = ValueBuilder<vector::ExtractElementOp>;
using vector_extract_slices = ValueBuilder<vector::ExtractSlicesOp>;
using vector_extract_strided_slice =
    ValueBuilder<vector::ExtractStridedSliceOp>;
using vector_fma = ValueBuilder<vector::FMAOp>;
using vector_insert = ValueBuilder<vector::InsertOp>;
using vector_insert_element = ValueBuilder<vector::InsertElementOp>;
using vector_insert_slices = ValueBuilder<vector::InsertSlicesOp>;
using vector_insert_strided_slice = ValueBuilder<vector::InsertStridedSliceOp>;
using vector_matmul = ValueBuilder<vector::MatmulOp>;
using vector_outerproduct = ValueBuilder<vector::OuterProductOp>;
using vector_print = OperationBuilder<vector::PrintOp>;
using vector_transfer_read = ValueBuilder<vector::TransferReadOp>;
using vector_transfer_write = OperationBuilder<vector::TransferWriteOp>;
using vector_transpose = ValueBuilder<vector::TransposeOp>;
using vector_type_cast = ValueBuilder<vector::TypeCastOp>;

} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_EDSC_INTRINSICS_H_
