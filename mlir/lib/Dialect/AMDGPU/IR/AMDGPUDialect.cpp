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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::amdgpu;

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.cpp.inc"

void AMDGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AMDGPU/AMDGPU.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/AMDGPU/AMDGPUAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// RawBuffer*Op
//===----------------------------------------------------------------------===//
template <typename T>
static LogicalResult verifyRawBufferOp(T &op) {
  MemRefType bufferType = op.getMemref().getType().template cast<MemRefType>();
  if (bufferType.getMemorySpaceAsInt() != 0)
    return op.emitOpError(
        "Buffer ops must operate on a memref in global memory");
  if (!bufferType.hasRank())
    return op.emitOpError(
        "Cannot meaningfully buffer_store to an unranked memref");
  if (static_cast<int64_t>(op.getIndices().size()) != bufferType.getRank())
    return op.emitOpError("Expected " + Twine(bufferType.getRank()) +
                          " indices to memref");
  return success();
}

LogicalResult RawBufferLoadOp::verify() { return verifyRawBufferOp(*this); }

LogicalResult RawBufferStoreOp::verify() { return verifyRawBufferOp(*this); }

LogicalResult RawBufferAtomicFaddOp::verify() {
  return verifyRawBufferOp(*this);
}

//===----------------------------------------------------------------------===//
// MFMAOp
//===----------------------------------------------------------------------===//
LogicalResult MFMAOp::verify() {
  constexpr uint32_t waveSize = 64;
  Builder b(getContext());

  Type sourceType = getSourceA().getType();
  Type destType = getDestC().getType();

  Type sourceElem = sourceType, destElem = destType;
  uint32_t sourceLen = 1, destLen = 1;
  if (auto sourceVector = sourceType.dyn_cast<VectorType>()) {
    sourceLen = sourceVector.getNumElements();
    sourceElem = sourceVector.getElementType();
  }
  if (auto destVector = destType.dyn_cast<VectorType>()) {
    destLen = destVector.getNumElements();
    destElem = destVector.getElementType();
  }

  // Normalize the wider integer types the compiler expects to i8
  if (sourceElem.isInteger(32)) {
    sourceLen *= 4;
    sourceElem = b.getI8Type();
  }
  if (sourceElem.isInteger(64)) {
    sourceLen *= 8;
    sourceElem = b.getI8Type();
  }

  int64_t numSourceElems = (getM() * getK() * getBlocks()) / waveSize;
  if (sourceLen != numSourceElems)
    return emitOpError("expected " + Twine(numSourceElems) +
                       " source values for this operation but got " +
                       Twine(sourceLen));

  int64_t numDestElems = (getM() * getN() * getBlocks()) / waveSize;
  if (destLen != numDestElems)
    return emitOpError("expected " + Twine(numDestElems) +
                       " result values for this operation but got " +
                       Twine(destLen));

  if (destElem.isF64() && getBlgp() != MFMAPermB::none)
    return emitOpError(
        "double-precision ops do not support permuting lanes of B");
  if (destElem.isF64() && getCbsz() != 0)
    return emitOpError(
        "double-precision ops do not support permuting lanes of A");
  if (getAbid() >= (1u << getCbsz()))
    return emitOpError(
        "block ID for permuting A (abid) must be below 2 ** cbsz");

  if ((getNegateA() || getNegateB() || getNegateC()) && !destElem.isF64())
    return emitOpError(
        "negation flags only available for double-precision operations");

  return success();
}

#include "mlir/Dialect/AMDGPU/AMDGPUEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AMDGPU/AMDGPUAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/AMDGPU/AMDGPU.cpp.inc"
