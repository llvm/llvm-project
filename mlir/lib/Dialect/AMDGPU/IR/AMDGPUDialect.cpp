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

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::amdgpu;

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.cpp.inc"

void AMDGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AMDGPU/IR/AMDGPU.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/AMDGPU/IR/AMDGPUAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// 8-bit float ops
//===----------------------------------------------------------------------===//
LogicalResult PackedTrunc2xFp8Op::verify() {
  if (getExisting() && getExisting().getType() != getResult().getType())
    return emitOpError("existing values must have same type as result");
  return success();
}

LogicalResult PackedStochRoundFp8Op::verify() {
  if (getExisting() && getExisting().getType() != getResult().getType())
    return emitOpError("existing values must have same type as result");
  return success();
}

//===----------------------------------------------------------------------===//
// RawBuffer*Op
//===----------------------------------------------------------------------===//
template <typename T>
static LogicalResult verifyRawBufferOp(T &op) {
  MemRefType bufferType = llvm::cast<MemRefType>(op.getMemref().getType());
  Attribute memorySpace = bufferType.getMemorySpace();
  bool isGlobal = false;
  if (!memorySpace)
    isGlobal = true;
  else if (auto intMemorySpace = llvm::dyn_cast<IntegerAttr>(memorySpace))
    isGlobal = intMemorySpace.getInt() == 0 || intMemorySpace.getInt() == 1;
  else if (auto gpuMemorySpace =
               llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    isGlobal = gpuMemorySpace.getValue() == gpu::AddressSpace::Global;

  if (!isGlobal)
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

LogicalResult RawBufferAtomicFmaxOp::verify() {
  return verifyRawBufferOp(*this);
}

LogicalResult RawBufferAtomicSmaxOp::verify() {
  return verifyRawBufferOp(*this);
}

LogicalResult RawBufferAtomicUminOp::verify() {
  return verifyRawBufferOp(*this);
}

LogicalResult RawBufferAtomicCmpswapOp::verify() {
  return verifyRawBufferOp(*this);
}

static std::optional<uint32_t> getConstantUint32(Value v) {
  APInt cst;
  if (!v.getType().isInteger(32))
    return std::nullopt;
  if (matchPattern(v, m_ConstantInt(&cst)))
    return cst.getZExtValue();
  return std::nullopt;
}

template <typename OpType>
static bool staticallyOutOfBounds(OpType op) {
  if (!op.getBoundsCheck())
    return false;
  MemRefType bufferType = op.getMemref().getType();
  if (!bufferType.hasStaticShape())
    return false;
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(getStridesAndOffset(bufferType, strides, offset)))
    return false;
  int64_t result = offset + op.getIndexOffset().value_or(0);
  if (op.getSgprOffset()) {
    std::optional<uint32_t> sgprOffset = getConstantUint32(op.getSgprOffset());
    if (!sgprOffset)
      return false;
    result += *sgprOffset;
  }
  if (strides.size() != op.getIndices().size())
    return false;
  int64_t indexVal = 0;
  for (auto pair : llvm::zip(strides, op.getIndices())) {
    int64_t stride = std::get<0>(pair);
    Value idx = std::get<1>(pair);
    std::optional<uint32_t> idxVal = getConstantUint32(idx);
    if (!idxVal)
      return false;
    indexVal += stride * *idxVal;
  }
  result += indexVal;
  if (result > std::numeric_limits<uint32_t>::max())
    // Overflow means don't drop
    return false;
  return result >= bufferType.getNumElements();
}

namespace {
template <typename OpType>
struct RemoveStaticallyOobBufferLoads final : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op, PatternRewriter &rw) const override {
    if (!staticallyOutOfBounds(op))
      return failure();
    Type loadType = op.getResult().getType();
    rw.replaceOpWithNewOp<arith::ConstantOp>(op, loadType,
                                             rw.getZeroAttr(loadType));
    return success();
  }
};

template <typename OpType>
struct RemoveStaticallyOobBufferWrites final : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op, PatternRewriter &rw) const override {
    if (!staticallyOutOfBounds(op))
      return failure();

    rw.eraseOp(op);
    return success();
  }
};
} // end namespace

void RawBufferLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<RemoveStaticallyOobBufferLoads<RawBufferLoadOp>>(context);
}

void RawBufferStoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<RemoveStaticallyOobBufferWrites<RawBufferStoreOp>>(context);
}

void RawBufferAtomicFaddOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<RemoveStaticallyOobBufferWrites<RawBufferAtomicFaddOp>>(context);
}

void RawBufferAtomicFmaxOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<RemoveStaticallyOobBufferWrites<RawBufferAtomicFmaxOp>>(context);
}

void RawBufferAtomicSmaxOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<RemoveStaticallyOobBufferWrites<RawBufferAtomicSmaxOp>>(context);
}

void RawBufferAtomicUminOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<RemoveStaticallyOobBufferWrites<RawBufferAtomicUminOp>>(context);
}

void RawBufferAtomicCmpswapOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<RemoveStaticallyOobBufferLoads<RawBufferAtomicCmpswapOp>>(
      context);
}

//===----------------------------------------------------------------------===//
// WMMAOp
//===----------------------------------------------------------------------===//
LogicalResult WMMAOp::verify() {
  Type sourceAType = getSourceA().getType();
  Type destType = getDestC().getType();

  VectorType sourceVectorAType = sourceAType.dyn_cast<VectorType>();
  VectorType destVectorType = destType.dyn_cast<VectorType>();

  Type sourceAElemType = sourceVectorAType.getElementType();
  Type destElemType = destVectorType.getElementType();

  bool isDestFloat =
      (destElemType.isF32() || destElemType.isF16() || destElemType.isBF16());
  bool isSrcFloat = (sourceAElemType.isF16() || sourceAElemType.isBF16());

  if (isDestFloat && !isSrcFloat) {
    return emitOpError("Expected float sources with float destination");
  }

  if (!isDestFloat && isSrcFloat) {
    return emitOpError("Expected int sources with int destination");
  }

  return success();
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
  if (auto sourceVector = llvm::dyn_cast<VectorType>(sourceType)) {
    sourceLen = sourceVector.getNumElements();
    sourceElem = sourceVector.getElementType();
  }
  if (auto destVector = llvm::dyn_cast<VectorType>(destType)) {
    destLen = destVector.getNumElements();
    destElem = destVector.getElementType();
  }

  Type sourceBType = getSourceB().getType();
  if (sourceElem.isFloat8E5M2FNUZ() || sourceElem.isFloat8E4M3FNUZ()) {
    int64_t sourceBLen = 1;
    Type sourceBElem = sourceBType;
    if (auto sourceBVector = llvm::dyn_cast<VectorType>(sourceBType)) {
      sourceBLen = sourceBVector.getNumElements();
      sourceBElem = sourceBVector.getElementType();
    }
    if (!sourceBElem.isFloat8E5M2FNUZ() && !sourceBElem.isFloat8E4M3FNUZ())
      return emitOpError("expected both source operands to have f8 elements");
    if (sourceLen != sourceBLen)
      return emitOpError(
          "expected both f8 source vectors to have the same length");
  } else {
    if (sourceType != sourceBType)
      return emitOpError(
          "expected both non-f8 source operand types to match exactly");
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

#include "mlir/Dialect/AMDGPU/IR/AMDGPUEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AMDGPU/IR/AMDGPUAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/AMDGPU/IR/AMDGPU.cpp.inc"
