//===- NVGPUDialect.cpp - MLIR NVGPU ops implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the NVGPU dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/NVGPU/NVGPUDialect.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::nvgpu;

#include "mlir/Dialect/NVGPU/NVGPUDialect.cpp.inc"

void nvgpu::NVGPUDialect::initialize() {
  addTypes<DeviceAsyncTokenType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/NVGPU/NVGPU.cpp.inc"
      >();
}

Type NVGPUDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();
  // Handle 'device async token' types.
  if (keyword == "device.async.token")
    return DeviceAsyncTokenType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown nvgpu type: " + keyword);
  return Type();
}

void NVGPUDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<DeviceAsyncTokenType>([&](Type) { os << "device.async.token"; })
      .Default([](Type) { llvm_unreachable("unexpected 'nvgpu' type kind"); });
}
//===----------------------------------------------------------------------===//
// NVGPU_DeviceAsyncCopyOp
//===----------------------------------------------------------------------===//

/// Return true if the last dimension of the MemRefType has unit stride. Also
/// return true for memrefs with no strides.
static bool isLastMemrefDimUnitStride(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(getStridesAndOffset(type, strides, offset))) {
    return false;
  }
  return strides.back() == 1;
}

LogicalResult DeviceAsyncCopyOp::verify() {
  auto srcMemref = src().getType().cast<MemRefType>();
  auto dstMemref = dst().getType().cast<MemRefType>();
  unsigned workgroupAddressSpace = gpu::GPUDialect::getWorkgroupAddressSpace();
  if (!isLastMemrefDimUnitStride(srcMemref))
    return emitError("source memref most minor dim must have unit stride");
  if (!isLastMemrefDimUnitStride(dstMemref))
    return emitError("destination memref most minor dim must have unit stride");
  if (dstMemref.getMemorySpaceAsInt() != workgroupAddressSpace)
    return emitError("destination memref must have memory space ")
           << workgroupAddressSpace;
  if (dstMemref.getElementType() != srcMemref.getElementType())
    return emitError("source and destination must have the same element type");
  if (size_t(srcMemref.getRank()) != srcIndices().size())
    return emitOpError() << "expected " << srcMemref.getRank()
                         << " source indices, got " << srcIndices().size();
  if (size_t(dstMemref.getRank()) != dstIndices().size())
    return emitOpError() << "expected " << dstMemref.getRank()
                         << " destination indices, got " << dstIndices().size();
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/NVGPU/NVGPU.cpp.inc"
