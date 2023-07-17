//===- Utils.cpp - Transform utilities ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/NVGPU/Transforms/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::nvgpu;

Operation::operand_range nvgpu::getIndices(Operation *op) {
  if (auto ldmatrixOp = dyn_cast<LdMatrixOp>(op))
    return ldmatrixOp.getIndices();
  if (auto copyOp = dyn_cast<DeviceAsyncCopyOp>(op))
    return copyOp.getDstIndices();
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getIndices();
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getIndices();
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndices();
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndices();
  llvm_unreachable("unsupported op type");
}

void nvgpu::setIndices(Operation *op, ArrayRef<Value> indices) {
  if (auto ldmatrixOp = dyn_cast<LdMatrixOp>(op))
    return ldmatrixOp.getIndicesMutable().assign(indices);
  if (auto copyOp = dyn_cast<DeviceAsyncCopyOp>(op))
    return copyOp.getDstIndicesMutable().assign(indices);
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getIndicesMutable().assign(indices);
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getIndicesMutable().assign(indices);
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndicesMutable().assign(indices);
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndicesMutable().assign(indices);
  llvm_unreachable("unsupported op type");
}
