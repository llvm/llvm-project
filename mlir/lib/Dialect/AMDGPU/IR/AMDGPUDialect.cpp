//===- AMDGPUDialect.cpp - MLIR AMDGPU dialect implementation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMDGPU dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.cpp.inc"

void amdgpu::AMDGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AMDGPU/AMDGPU.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// RawBuffer*Op
//===----------------------------------------------------------------------===//
template <typename T>
static LogicalResult verifyRawBufferOp(T &op) {
  MemRefType bufferType = op.memref().getType().template cast<MemRefType>();
  if (bufferType.getMemorySpaceAsInt() != 0)
    return op.emitOpError(
        "Buffer ops must operate on a memref in global memory");
  if (!bufferType.hasRank())
    return op.emitOpError(
        "Cannot meaningfully buffer_store to an unranked memref");
  if (static_cast<int64_t>(op.indices().size()) != bufferType.getRank())
    return op.emitOpError("Expected " + Twine(bufferType.getRank()) +
                          " indices to memref");
  return success();
}

LogicalResult amdgpu::RawBufferLoadOp::verify() {
  return verifyRawBufferOp(*this);
}

LogicalResult amdgpu::RawBufferStoreOp::verify() {
  return verifyRawBufferOp(*this);
}

LogicalResult amdgpu::RawBufferAtomicFaddOp::verify() {
  return verifyRawBufferOp(*this);
}

#define GET_OP_CLASSES
#include "mlir/Dialect/AMDGPU/AMDGPU.cpp.inc"
