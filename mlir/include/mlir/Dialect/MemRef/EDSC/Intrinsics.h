//===- Intrinsics.h - MLIR EDSC Intrinsics for MemRefOps --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_MEMREF_EDSC_INTRINSICS_H_
#define MLIR_DIALECT_MEMREF_EDSC_INTRINSICS_H_

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/EDSC/Builders.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

using memref_alloc = ValueBuilder<memref::AllocOp>;
using memref_alloca = ValueBuilder<memref::AllocaOp>;
using memref_cast = ValueBuilder<memref::CastOp>;
using memref_dealloc = OperationBuilder<memref::DeallocOp>;
using memref_dim = ValueBuilder<memref::DimOp>;
using memref_load = ValueBuilder<memref::LoadOp>;
using memref_store = OperationBuilder<memref::StoreOp>;
using memref_sub_view = ValueBuilder<memref::SubViewOp>;
using memref_tensor_load = ValueBuilder<memref::TensorLoadOp>;
using memref_tensor_store = OperationBuilder<memref::TensorStoreOp>;
using memref_view = ValueBuilder<memref::ViewOp>;

/// Provide an index notation around memref_load and memref_store.
using MemRefIndexedValue =
    TemplatedIndexedValue<intrinsics::memref_load, intrinsics::memref_store>;
} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_EDSC_INTRINSICS_H_
