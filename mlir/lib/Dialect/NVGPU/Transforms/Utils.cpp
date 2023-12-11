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

Value nvgpu::getValueStored(Operation *op) {
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getValueToStore();
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op))
    return transferWrite.getValue();
  if (auto storeOp = dyn_cast<vector::StoreOp>(op))
    return storeOp.getValueToStore();
  llvm_unreachable("unsupported op type");
}

Value nvgpu::getMemrefOperand(Operation *op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getMemref();
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getMemref();
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op))
    return transferWrite.getSource();
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op))
    return transferRead.getSource();
  if (auto storeOp = dyn_cast<vector::StoreOp>(op))
    return storeOp.getBase();
  if (auto loadOp = dyn_cast<vector::LoadOp>(op))
    return loadOp.getBase();
  llvm_unreachable("unsupported op type");
}
