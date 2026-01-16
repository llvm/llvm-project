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
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::amdgpu;

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.cpp.inc"

namespace {
struct AMDGPUInlinerInterface final : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void AMDGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AMDGPU/IR/AMDGPU.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/AMDGPU/IR/AMDGPUTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/AMDGPU/IR/AMDGPUAttributes.cpp.inc"
      >();
  addInterfaces<AMDGPUInlinerInterface>();
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
// mxfp float ops
//===----------------------------------------------------------------------===//
LogicalResult PackedScaledTruncOp::verify() {
  if (getExisting() && getExisting().getType() != getResult().getType())
    return emitOpError("existing values must have same type as result");
  return success();
}

//===----------------------------------------------------------------------===//
// FatRawBufferCastOp
//===----------------------------------------------------------------------===//

/// Convert the type `source` to one with the same sizes and strides - and
/// offset, unless `stripOffset` is true, in which case the offset is reset to
/// 0, if the offset should be reset but the layout of `source` isn't either the
/// identity layout or a strided layout, this function fails.
static FailureOr<MemRefType> getFatRawBufferTypeLike(MemRefType source,
                                                     bool resetOffset) {
  MLIRContext *ctx = source.getContext();
  MemRefType::Builder mb(source);
  mb.setMemorySpace(
      amdgpu::AddressSpaceAttr::get(ctx, amdgpu::AddressSpace::FatRawBuffer));
  MemRefLayoutAttrInterface layout = source.getLayout();
  if (resetOffset && !layout.isIdentity()) {
    auto stridedLayout = dyn_cast<StridedLayoutAttr>(layout);
    if (!stridedLayout)
      return failure();
    MemRefLayoutAttrInterface newLayout =
        StridedLayoutAttr::get(ctx, 0, stridedLayout.getStrides());
    // Special case: if resetting the offset causes the strided layout to become
    // the identity layout, then reset to the identity layout.
    // TODO: this'll get a lot simpler when we have the contiguous layout.
    SmallVector<int64_t> stridesIfIdentity;
    if (source.hasStaticShape()) {
      stridesIfIdentity = computeSuffixProduct(source.getShape());
    } else if (source.getRank() <= 1) {
      stridesIfIdentity = SmallVector<int64_t>(source.getRank(), 1);
    }
    if (stridesIfIdentity == stridedLayout.getStrides()) {
      newLayout = AffineMapAttr::get(
          AffineMap::getMultiDimIdentityMap(source.getRank(), ctx));
    }
    mb.setLayout(newLayout);
  }
  return (MemRefType)(mb);
}

LogicalResult FatRawBufferCastOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  auto sourceType =
      dyn_cast_if_present<MemRefType>(adaptor.getSource().getType());
  if (!sourceType)
    return failure();
  FailureOr<MemRefType> resultType =
      getFatRawBufferTypeLike(sourceType, adaptor.getResetOffset());
  if (failed(resultType))
    return failure();
  inferredReturnTypes = SmallVector<Type>{*resultType};
  return success();
}

FailureOr<OpFoldResult> FatRawBufferCastOp::reifyDimOfResult(OpBuilder &builder,
                                                             int resultIndex,
                                                             int dim) {
  assert(resultIndex == 0 && "FatRawBufferCastOp has a single result");
  return memref::getMixedSize(builder, getLoc(), getSource(), dim);
}

LogicalResult FatRawBufferCastOp::verify() {
  FailureOr<MemRefType> expectedResultType =
      getFatRawBufferTypeLike(getSource().getType(), getResetOffset());
  if (failed(expectedResultType))
    return emitOpError("source type ")
           << getSource().getType() << " can't have its offset reset";
  if (getResult().getType() != *expectedResultType)
    return emitOpError("expected result type to be ")
           << *expectedResultType << " but got " << getResult().getType();
  return success();
}

static bool hasGlobalMemorySpace(Attribute memorySpace) {
  if (!memorySpace)
    return true;
  if (auto intMemorySpace = dyn_cast<IntegerAttr>(memorySpace))
    return intMemorySpace.getInt() == 0 || intMemorySpace.getInt() == 1;
  if (auto gpuMemorySpace = dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuMemorySpace.getValue() == gpu::AddressSpace::Global;
  return false;
}

static bool hasWorkgroupMemorySpace(Attribute memorySpace) {
  if (!memorySpace)
    return false;
  if (auto intMemorySpace = dyn_cast<IntegerAttr>(memorySpace))
    return intMemorySpace.getInt() == 3;
  if (auto gpuMemorySpace = dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuMemorySpace.getValue() == gpu::AddressSpace::Workgroup;
  return false;
}

static bool hasFatRawBufferMemorySpace(Attribute memorySpace) {
  if (!memorySpace)
    return false;
  if (auto intMemorySpace = dyn_cast<IntegerAttr>(memorySpace))
    return intMemorySpace.getInt() == 7;
  if (auto gpuMemorySpace = dyn_cast<amdgpu::AddressSpaceAttr>(memorySpace))
    return gpuMemorySpace.getValue() == amdgpu::AddressSpace::FatRawBuffer;
  return false;
}

//===----------------------------------------------------------------------===//
// RawBuffer*Op
//===----------------------------------------------------------------------===//
template <typename T>
static LogicalResult verifyRawBufferOp(T &op) {
  MemRefType bufferType = llvm::cast<MemRefType>(op.getMemref().getType());
  bool isGlobal = hasGlobalMemorySpace(bufferType.getMemorySpace());

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
  if (failed(bufferType.getStridesAndOffset(strides, offset)))
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
// ScaledExtPackedMatrixOp
//===----------------------------------------------------------------------===//
LogicalResult ScaledExtPackedMatrixOp::verify() {
  int blockSize = getBlockSize();
  assert(llvm::is_contained({16, 32}, blockSize) && "invalid block size");

  int firstScaleByte = getFirstScaleByte();
  int firstScaleLane = getFirstScaleLane();
  auto sourceType = cast<VectorType>(getSource().getType());
  Type elementType = sourceType.getElementType();
  auto floatType = cast<FloatType>(elementType);
  unsigned bitWidth = floatType.getWidth();

  assert(llvm::is_contained(llvm::ArrayRef<unsigned>{4, 6, 8}, bitWidth));

  const bool is_fp8 = bitWidth == 8;
  const bool is_block_16 = blockSize == 16;

  if (!is_fp8) {
    if (is_block_16) {
      if (!llvm::is_contained({0, 1}, firstScaleByte)) {
        return emitOpError("blockSize of 16 can only have firstScaleByte be 0 "
                           "or 1 for f4 and f6.");
      }
    } else {
      if (!llvm::is_contained({0, 2}, firstScaleByte)) {
        return emitOpError("blockSize of 32 can only have firstScaleByte be 0 "
                           "or 2 for f4 and f6.");
      }
    }
  } else {
    if (is_block_16) {
      bool is_valid = ((firstScaleLane == 0) && (firstScaleByte == 0)) ||
                      ((firstScaleLane == 16) && (firstScaleByte == 2));
      if (!is_valid) {
        return emitOpError("blockSize of 16 can only have (firstScaleLane, "
                           "firstScaleByte) be (0, 0) or (16, 2) for f8.");
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// WMMAOp
//===----------------------------------------------------------------------===//

ParseResult mlir::amdgpu::parseMNKDimensionList(OpAsmParser &parser,
                                                IntegerAttr &m, IntegerAttr &n,
                                                IntegerAttr &k) {
  SmallVector<int64_t, 3> dimensions;
  if (parser.parseDimensionList(dimensions, false, false))
    return failure();
  if (dimensions.size() != 3)
    return parser.emitError(parser.getCurrentLocation())
           << "expected 3 dimensions in MNK dimension list";

  m = parser.getBuilder().getI32IntegerAttr(dimensions[0]);
  n = parser.getBuilder().getI32IntegerAttr(dimensions[1]);
  k = parser.getBuilder().getI32IntegerAttr(dimensions[2]);
  return success();
}

LogicalResult WMMAOp::verify() {
  auto sourceAType = cast<VectorType>(getSourceA().getType());
  auto sourceBType = cast<VectorType>(getSourceB().getType());
  auto destType = cast<VectorType>(getDestC().getType());

  Type sourceAElemType = sourceAType.getElementType();
  Type sourceBElemType = sourceBType.getElementType();
  if (sourceAType.getNumElements() != sourceBType.getNumElements()) {
    return emitOpError("source vectors have different lengths: ")
           << sourceAType << " vs. " << sourceBType;
  }

  bool isDestFloat = destType.getElementType().isFloat();
  bool isSrcFloat = sourceAElemType.isFloat();

  if (isDestFloat && !isSrcFloat)
    return emitOpError("expected float sources with float destination");
  if (!isDestFloat && isSrcFloat)
    return emitOpError("expected int sources with int destination");

  if (!sourceAElemType.isFloat(8) && sourceAElemType != sourceBElemType) {
    return emitOpError(
               "source element types must match (except for fp8/bf8) but have ")
           << sourceAType << " and " << sourceBType;
  }

  if (isSrcFloat) {
    if (getClamp())
      return emitOpError("clamp flag is not supported for float types");
    if (getUnsignedA() || getUnsignedB())
      return emitOpError("unsigned flags are not supported for float types");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ScaledWMMAOp
//===----------------------------------------------------------------------===//

LogicalResult ScaledWMMAOp::verify() {
  // Helper functions for type classification.
  auto isF8 = llvm::IsaPred<Float8E4M3FNType, Float8E5M2Type>;
  auto isF6 = llvm::IsaPred<Float6E2M3FNType, Float6E3M2FNType>;
  auto isF4 = llvm::IsaPred<Float4E2M1FNType>;
  auto isScaleF8 = llvm::IsaPred<Float8E8M0FNUType, Float8E4M3FNType>;
  auto isE8M0 = llvm::IsaPred<Float8E8M0FNUType>;
  auto isE4M3 = llvm::IsaPred<Float8E4M3FNType>;

  auto sourceAType = cast<VectorType>(getSourceA().getType());
  auto sourceBType = cast<VectorType>(getSourceB().getType());
  auto destType = cast<VectorType>(getDestC().getType());

  // Validate source element types are small floats (fp4/fp6/fp8).
  Type aElemType = sourceAType.getElementType();
  Type bElemType = sourceBType.getElementType();

  // Validate vector lengths based on dimensions.
  int64_t m = getM();
  int64_t aLen = sourceAType.getNumElements();
  int64_t bLen = sourceBType.getNumElements();
  int64_t expectedOutLen = (m == 16) ? 8 : 16;

  if (destType.getNumElements() != expectedOutLen)
    return emitOpError("expected output vector of length ")
           << expectedOutLen << " but got " << destType.getNumElements();

  if (m == 16) {
    // For 16×16×128: both A and B must be 64 elements.
    if (aLen != 64)
      return emitOpError(
                 "for 16x16x128, sourceA must have 64 elements but got ")
             << aLen;
    if (bLen != 64)
      return emitOpError(
                 "for 16x16x128, sourceB must have 64 elements but got ")
             << bLen;
  } else { // m == 32
    // For 32×16×128: only fp4 is supported, A is 128, B is 64.
    if (!isF4(aElemType) && !isF4(bElemType))
      return emitOpError("32x16x128 only supports fp4 element types");

    if (aLen != 128)
      return emitOpError(
                 "for 32x16x128, sourceA must have 128 elements but got ")
             << aLen;
    if (bLen != 64)
      return emitOpError(
                 "for 32x16x128, sourceB must have 64 elements but got ")
             << bLen;

    // For 32x16x128, matrix A uses all 32 lanes so a_first_scale_lane must be
    // 0.
    if (getAFirstScaleLane() != 0)
      return emitOpError("for 32x16x128, a_first_scale_lane must be 0");
  }

  // Validate scale types and their compatibility with matrix element types.
  auto scaleAType = cast<VectorType>(getScaleA().getType());
  auto scaleBType = cast<VectorType>(getScaleB().getType());
  Type scaleAElemType = scaleAType.getElementType();
  Type scaleBElemType = scaleBType.getElementType();

  // Validate scale element types are valid scale f8 types (E8M0FNU or E4M3FN).
  if (!isScaleF8(scaleAElemType) || !isScaleF8(scaleBElemType))
    return emitOpError(
        "scale operands must have f8 element types (E8M0FNU or E4M3FN)");

  // Any matrices A/B (fp8|fp6|fp4) with E8M0 scales for matrix A/B are valid.
  if (isE8M0(scaleAElemType) && isE8M0(scaleBElemType))
    return success();

  // Matrix A (F8|F6) x Matrix B (F4) with Scale A (E8M0), Scale B (E5M3|E4M3).
  if ((isF8(aElemType) || isF6(aElemType)) && isE8M0(scaleAElemType) &&
      isF4(bElemType) && isE4M3(scaleBElemType))
    return success();

  // Matrix A (F4) x Matrix B (F8|F6) with Scale A (E5M3|E4M3), Scale B (E8M0).
  if (isF4(aElemType) && isE4M3(scaleAElemType) &&
      (isF8(bElemType) || isF6(bElemType)) && isE8M0(scaleBElemType))
    return success();

  // Matrix A (F4) x Matrix B (F4) with Scale A (E4M3), Scale B (E4M3).
  if (isF4(aElemType) && isF4(bElemType) && isE4M3(scaleAElemType) &&
      isE4M3(scaleBElemType))
    return success();

  // No valid combination matched.
  return emitOpError("invalid combination of matrix and scale types: ")
         << "sourceA=" << aElemType << ", scaleA=" << scaleAElemType
         << ", sourceB=" << bElemType << ", scaleB=" << scaleBElemType;
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
  if (auto sourceVector = dyn_cast<VectorType>(sourceType)) {
    sourceLen = sourceVector.getNumElements();
    sourceElem = sourceVector.getElementType();
  }
  if (auto destVector = dyn_cast<VectorType>(destType)) {
    destLen = destVector.getNumElements();
    destElem = destVector.getElementType();
  }

  Type sourceBType = getSourceB().getType();
  if (sourceElem.isFloat(8) || sourceElem.isFloat(6) || sourceElem.isFloat(4)) {
    int64_t sourceBLen = 1;
    Type sourceBElem = sourceBType;
    if (auto sourceBVector = llvm::dyn_cast<VectorType>(sourceBType)) {
      sourceBLen = sourceBVector.getNumElements();
      sourceBElem = sourceBVector.getElementType();
    }
    if (!sourceBElem.isFloat(8) && !sourceBElem.isFloat(6) &&
        !sourceBElem.isFloat(4))
      return emitOpError("expected both source operands to have small-float "
                         "elements if one does");
    if (sourceLen != sourceBLen)
      return emitOpError(
          "expected both small-float source vectors to have the same length");
  } else {
    if (sourceType != sourceBType)
      return emitOpError("expected both non-small-float source operand types "
                         "to match exactly");
  }
  // Normalize the wider integer types the compiler expects to i8.
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

//===----------------------------------------------------------------------===//
// SparseMFMAOp
//===----------------------------------------------------------------------===//

LogicalResult SparseMFMAOp::verify() {
  constexpr uint32_t waveSize = 64;

  auto sparseType = cast<VectorType>(getSourceA().getType());
  auto denseType = cast<VectorType>(getSourceB().getType());
  auto destType = cast<VectorType>(getDestC().getType());

  Type sparseElem = sparseType.getElementType();
  Type denseElem = denseType.getElementType();
  int64_t sparseLen = sparseType.getNumElements();
  int64_t denseLen = denseType.getNumElements();
  int64_t destLen = destType.getNumElements();

  if (denseLen != 2 * sparseLen)
    return emitOpError("expected dense source operand to have exactly double "
                       "the number of elements of the sparse source operand");

  // Check that source element types are compatible.
  // For fp8/bf8 mixed operations, element types can differ (e.g., fp8 * bf8).
  // For other types, element types must match exactly.
  bool bothFloat8 = sparseElem.isFloat(8) && denseElem.isFloat(8);
  if (!bothFloat8 && sparseElem != denseElem)
    return emitOpError(
        "expected source operands to have the same element type");

  // When CBSZ == 0, ABID selects the index set within the sparse index VGPR.
  // When CBSZ != 0, the first index set is always used (ABID ignored).
  bool is8BitSource = sparseElem.isFloat(8) || sparseElem.isInteger(8);
  // 8-bit source: ABID selects one of two 16-bit index sets.
  if (getCbsz() == 0 && is8BitSource && getAbid() > 1)
    return emitOpError("ABID must be 0 or 1 for 8-bit source data");
  // 16-bit source: ABID selects one of four 8-bit index sets (0-3 all valid).
  if (getCbsz() == 0 && !is8BitSource && getAbid() > 3)
    return emitOpError("ABID must be between 0 and 3 for 16-bit source data");

  // Validate sparseIdx type matches source element type.
  auto sparseIdxType = cast<VectorType>(getSparseIdx().getType());
  if (is8BitSource) {
    // 8-bit source data requires vector<2xi16> sparse indices.
    if (sparseIdxType.getNumElements() != 2 ||
        !sparseIdxType.getElementType().isInteger(16))
      return emitOpError("expected vector<2xi16> sparse indices for 8-bit "
                         "source data, but got ")
             << getSparseIdx().getType();
  } else {
    // 16-bit source data requires vector<4xi8> sparse indices.
    if (sparseIdxType.getNumElements() != 4 ||
        !sparseIdxType.getElementType().isInteger(8))
      return emitOpError("expected vector<4xi8> sparse indices for 16-bit "
                         "source data, but got ")
             << getSparseIdx().getType();
  }

  int64_t expectedSourceElems = (getM() * getK()) / waveSize;
  if (denseLen != expectedSourceElems)
    return emitOpError("expected " + Twine(expectedSourceElems) +
                       " source values for this operation but got " +
                       Twine(denseLen));

  int64_t expectedDestElems = (getM() * getN()) / waveSize;
  if (destLen != expectedDestElems)
    return emitOpError("expected " + Twine(expectedDestElems) +
                       " result values for this operation but got " +
                       Twine(destLen));

  return success();
}

//===----------------------------------------------------------------------===//
// DPPOp
//===----------------------------------------------------------------------===//
LogicalResult DPPOp::verify() {
  Type srcType = getSrc().getType();
  if (srcType.getIntOrFloatBitWidth() > 64) {
    return emitOpError("integer and floating point types larger than 64 bits "
                       "are not supported");
  }

  DPPPerm kind = getKind();
  Attribute permArgument = getPermArgument().value_or(Attribute{});

  switch (kind) {

  case DPPPerm::quad_perm: {
    auto quadPermAttr = dyn_cast_or_null<ArrayAttr>(permArgument);
    if (!quadPermAttr || quadPermAttr.size() != 4) {
      return emitOpError("quad_perm attribute must have exactly 4 elements");
    }
    for (auto elem : quadPermAttr.getAsRange<IntegerAttr>()) {
      int32_t num = elem.getInt();
      if (num < 0 || num > 3) {
        return emitOpError(
            "Each element of quad_perm must be in the range [0, 3]");
      }
    }
  } break;

  case DPPPerm::row_shl:
  case DPPPerm::row_shr:
  case DPPPerm::row_ror: {
    if (!permArgument) {
      return emitOpError("Attribute '" + Twine(stringifyDPPPerm(kind)) +
                         "' value not specified");
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(permArgument)) {
      uint32_t attrValue = intAttr.getInt();
      if (attrValue < 1 || attrValue > 15) {
        return emitOpError("Attribute value must be between 1 and 15");
      }
    }
  } break;

  case DPPPerm::wave_shl:
  case DPPPerm::wave_shr:
  case DPPPerm::wave_rol:
  case DPPPerm::wave_ror:
  case DPPPerm::row_mirror:
  case DPPPerm::row_half_mirror:
  case DPPPerm::row_bcast_15:
  case DPPPerm::row_bcast_31: {
    if (permArgument && !isa<UnitAttr>(permArgument)) {
      return emitOpError("Expected unit attribute for permArgument, but found "
                         "non-trivial argument");
    }
    break;
  }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PermlaneSwapOp
//===----------------------------------------------------------------------===//
LogicalResult PermlaneSwapOp::verify() {
  unsigned rowLength = getRowLength();

  if (rowLength != 16 && rowLength != 32)
    return emitOpError("row_length attribute must either be 16 or 32.");

  return success();
}

/// Remove amdgpu.lds_barrier after amdgpu.lds_barrier.
static LogicalResult eraseRedundantLDSBarrierOps(LDSBarrierOp op,
                                                 PatternRewriter &rewriter) {
  if (isa_and_nonnull<LDSBarrierOp>(op->getNextNode())) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

void LDSBarrierOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add(eraseRedundantLDSBarrierOps);
}

//===----------------------------------------------------------------------===//
// MemoryCounterWaitOp
//===----------------------------------------------------------------------===//

namespace {
/// Fuse adjacent memory counter wait ops, taking the minimum value of the
/// counters.
struct FuseMemoryCounterWaitOp final : OpRewritePattern<MemoryCounterWaitOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(MemoryCounterWaitOp op,
                                PatternRewriter &rewriter) const override {
    auto next = dyn_cast<MemoryCounterWaitOp>(op->getNextNode());
    if (!next)
      return failure();

    auto setters = {&MemoryCounterWaitOp::setLoad,
                    &MemoryCounterWaitOp::setStore, &MemoryCounterWaitOp::setDs,
                    &MemoryCounterWaitOp::setExp,
                    &MemoryCounterWaitOp::setTensor};
    auto lhsVals = {op.getLoad(), op.getStore(), op.getDs(), op.getExp(),
                    op.getTensor()};
    auto rhsVals = {next.getLoad(), next.getStore(), next.getDs(),
                    next.getExp(), next.getTensor()};
    rewriter.modifyOpInPlace(op, [&] {
      for (auto [setter, lhs, rhs] :
           llvm::zip_equal(setters, lhsVals, rhsVals)) {
        if (lhs && rhs) {
          (op.*setter)(std::min(*lhs, *rhs));
        } else if (lhs) {
          (op.*setter)(*lhs);
        } else if (rhs) {
          (op.*setter)(*rhs);
        }
      }
    });
    rewriter.eraseOp(next);
    return success();
  }
};
} // namespace

void MemoryCounterWaitOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<FuseMemoryCounterWaitOp>(context);
}

//===----------------------------------------------------------------------===//
// GatherToLDSOp
//===----------------------------------------------------------------------===//

LogicalResult GatherToLDSOp::verify() {
  MemRefType srcType = cast<MemRefType>(getSrc().getType());
  MemRefType dstType = cast<MemRefType>(getDst().getType());

  if (dstType.getRank() > 0 && !dstType.areTrailingDimsContiguous(1))
    return emitOpError("destination type inner most dim must be contiguous");

  auto elemType = srcType.getElementType();
  // Check $src and $dst element types are the same.
  if (elemType != dstType.getElementType())
    return emitOpError("source and destination element types must match");

  // copy type sizes should be 1, 2, 4, 12 or 16 bytes.
  auto transferType = getTransferType();
  int transferSize;
  if (auto vectorTransfer = dyn_cast<VectorType>(transferType)) {
    transferSize = vectorTransfer.getNumElements() *
                   vectorTransfer.getElementTypeBitWidth();
  } else {
    transferSize = transferType.getIntOrFloatBitWidth();
  }
  if (!llvm::is_contained({8, 16, 32, 96, 128}, transferSize))
    return emitOpError(
        "Transfering type size must be 8, 16, 32, 96 or 128 bits");

  if (!hasGlobalMemorySpace(srcType.getMemorySpace()) &&
      !hasFatRawBufferMemorySpace(srcType.getMemorySpace()))
    return emitOpError(
        "source memory address space must be global or fat raw buffer");

  if (!hasWorkgroupMemorySpace(dstType.getMemorySpace()))
    return emitOpError("destination memory address space must be Workgroup");

  return success();
}

namespace {
/// If the source/target of a GatherToLDSOp is a CastOp that only removes static
/// information or changes layout, the cast can be skipped.
struct FoldGatherToLDSOfCast final : OpRewritePattern<GatherToLDSOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherToLDSOp gatherOp,
                                PatternRewriter &rewriter) const override {
    bool modified = false;
    auto foldCast = [&](OpOperand &operand) {
      if (auto castOp = operand.get().getDefiningOp<memref::CastOp>()) {
        if (memref::CastOp::canFoldIntoConsumerOp(castOp)) {
          rewriter.modifyOpInPlace(gatherOp,
                                   [&] { operand.assign(castOp.getSource()); });
          modified = true;
        }
      }
    };

    foldCast(gatherOp.getSrcMutable());
    foldCast(gatherOp.getDstMutable());

    return success(modified);
  }
};
} // namespace

void GatherToLDSOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<FoldGatherToLDSOfCast>(context);
}

//===----------------------------------------------------------------------===//
// TransposeLoadOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeLoadOp::verify() {
  MemRefType srcType = cast<MemRefType>(getSrc().getType());

  if (!hasWorkgroupMemorySpace(srcType.getMemorySpace()))
    return emitOpError("source memory address space must be Workgroup");

  auto transferType = cast<VectorType>(getType());
  size_t numElements = transferType.getNumElements();
  size_t elementTypeSize =
      transferType.getElementType().getIntOrFloatBitWidth();

  // ElementSize -> NumElements
  const llvm::SmallDenseMap<size_t, size_t> kValidLoadSizeMap = {
      {4, 16},
      {6, 16},
      {8, 8},
      {16, 4},
  };

  auto validNumElems = kValidLoadSizeMap.find(elementTypeSize);
  if (validNumElems == kValidLoadSizeMap.end())
    return emitOpError("Unsupported element type size for transpose load: ")
           << elementTypeSize << " bits";

  if (numElements != validNumElems->second)
    return emitOpError(
               "Transferring type size mismatch: expected num of elements: ")
           << validNumElems->second;

  return success();
}

//===----------------------------------------------------------------------===//
// MakeDmaBaseOp
//===----------------------------------------------------------------------===//

template <typename BaseOp>
static LogicalResult verifyBase(BaseOp op) {
  auto ldsType = cast<MemRefType>(op.getLds().getType());
  auto globalType = cast<MemRefType>(op.getGlobal().getType());
  if (!hasWorkgroupMemorySpace(ldsType.getMemorySpace()))
    return op.emitOpError(
        "lds memref must have workgroup address space attribute.");
  if (!hasGlobalMemorySpace(globalType.getMemorySpace()))
    return op.emitOpError(
        "global memref must have global address space attribute.");

  Type elementType = ldsType.getElementType();
  unsigned width = elementType.getIntOrFloatBitWidth();

  if (!llvm::is_contained({8u, 16u, 32u, 64u}, width))
    return op.emitOpError(
               "element type must be 1, 2, 4, or 8 bytes long but type was ")
           << width << " bits long.";
  return success();
}

LogicalResult MakeDmaBaseOp::verify() { return verifyBase(*this); }

//===----------------------------------------------------------------------===//
// MakeGatherDmaBaseOp
//===----------------------------------------------------------------------===//

LogicalResult
TDMGatherBaseType::verify(function_ref<InFlightDiagnostic()> emitError,
                          Type elementType, Type indexType) {
  unsigned width = elementType.getIntOrFloatBitWidth();
  if (!llvm::is_contained({8u, 16u, 32u, 64u}, width))
    return emitError()
           << "element type must be 1, 2, 4, or 8 bytes wide but type "
           << elementType << " is " << width / 8 << " bytes wide.";
  MLIRContext *ctx = elementType.getContext();
  Type i16 = IntegerType::get(ctx, 32);
  Type i32 = IntegerType::get(ctx, 16);
  if (!llvm::is_contained({i16, i32}, indexType))
    return emitError() << "index type must be i16 or i32 but index type is "
                       << indexType << ".";
  return success();
}

LogicalResult MakeGatherDmaBaseOp::verify() { return verifyBase(*this); }

//===----------------------------------------------------------------------===//
// MakeDmaDescriptorOp
//===----------------------------------------------------------------------===//

template <typename DescriptorOp>
static LogicalResult verifyDescriptorOp(DescriptorOp op) {
  ArrayRef<int64_t> globalStaticStrides = op.getGlobalStaticStrides();

  if (globalStaticStrides.empty())
    return op.emitOpError("strides must not be empty.");
  if (globalStaticStrides.back() != 1)
    return op.emitOpError("strides for the innermost dimension must be 1.");

  ArrayRef<int64_t> globalStaticSizes = op.getGlobalStaticSizes();
  size_t rank = globalStaticSizes.size();
  if (rank > 5)
    return op.emitOpError("tensor and tile must be at most of rank 5.");
  if (rank != globalStaticStrides.size())
    return op.emitOpError("strides and sizes must have same rank.");

  ArrayRef<int64_t> sharedStaticSizes = op.getSharedStaticSizes();
  if (rank != sharedStaticSizes.size())
    return op.emitOpError("tensor must have same rank as tile.");

  unsigned elementTypeWidth = op.getElementTypeWidth();
  if (!llvm::is_contained({8u, 16u, 32u, 64u}, elementTypeWidth))
    return op.emitOpError(
               "element type width must be 1, 2, 4 or 8 bytes, but was ")
           << elementTypeWidth << " bits long";

  if (Value atomicBarrierAddress = op.getAtomicBarrierAddress()) {
    auto atomicBarrierAddressType =
        cast<MemRefType>(atomicBarrierAddress.getType());
    bool barrierInLDS =
        hasWorkgroupMemorySpace(atomicBarrierAddressType.getMemorySpace());
    if (!barrierInLDS)
      return op.emitOpError("atomic barrier address must be in LDS.");
  }

  if (op.getEarlyTimeout() && !op.getWorkgroupMask())
    return op.emitOpError(
        "early timeout does not apply when workgroup_mask is not set.");
  return success();
}

template <typename DescriptorOp, typename FoldAdaptor>
static OpFoldResult foldDescriptorOp(DescriptorOp op, FoldAdaptor adaptor) {
  SmallVector<OpFoldResult> mixedGlobalSizes(op.getMixedGlobalSizes());
  SmallVector<OpFoldResult> mixedGlobalStrides(op.getMixedGlobalStrides());
  SmallVector<OpFoldResult> mixedSharedSizes(op.getMixedSharedSizes());

  if (failed(foldDynamicIndexList(mixedGlobalSizes, /*onlyNonNegative=*/true,
                                  /*onlyNonZero=*/true)) &&
      failed(foldDynamicIndexList(mixedGlobalStrides, /*onlyNonNegative=*/true,
                                  /*onlyNonZero=*/true)) &&
      failed(foldDynamicIndexList(mixedSharedSizes, /*onlyNonNegative=*/true,
                                  /*onlyNonZero=*/true)))
    return nullptr;

  SmallVector<Value> dynamicGlobalSizes, dynamicGlobalStrides,
      dynamicSharedSizes;
  SmallVector<int64_t> staticGlobalSizes, staticGlobalStrides,
      staticSharedSizes;

  dispatchIndexOpFoldResults(mixedGlobalSizes, dynamicGlobalSizes,
                             staticGlobalSizes);
  op.setGlobalStaticSizes(staticGlobalSizes);
  op.getGlobalDynamicSizesMutable().assign(dynamicGlobalSizes);

  dispatchIndexOpFoldResults(mixedGlobalStrides, dynamicGlobalStrides,
                             staticGlobalStrides);
  op.setGlobalStaticStrides(staticGlobalStrides);
  op.getGlobalDynamicStridesMutable().assign(dynamicGlobalStrides);

  dispatchIndexOpFoldResults(mixedSharedSizes, dynamicSharedSizes,
                             staticSharedSizes);
  op.setSharedStaticSizes(staticSharedSizes);
  op.getSharedDynamicSizesMutable().assign(dynamicSharedSizes);
  return op.getResult();
}

LogicalResult MakeDmaDescriptorOp::verify() {
  return verifyDescriptorOp(*this);
}

OpFoldResult MakeDmaDescriptorOp::fold(FoldAdaptor adaptor) {
  return foldDescriptorOp(*this, adaptor);
}

//===----------------------------------------------------------------------===//
// MakeGatherDmaDescriptorOp
//===----------------------------------------------------------------------===//

LogicalResult MakeGatherDmaDescriptorOp::verify() {
  ArrayRef<int64_t> globalStaticSizes = getGlobalStaticSizes();
  size_t rank = globalStaticSizes.size();
  if (rank > 2)
    return emitOpError(
        "tensor and tile must be at most of rank two in gather mode.");
  Value indices = getIndices();
  Type elementType = cast<VectorType>(indices.getType()).getElementType();
  if (elementType != getBase().getType().getIndexType())
    return emitOpError("indices' element type must match base's element type.");

  return verifyDescriptorOp(*this);
}

OpFoldResult MakeGatherDmaDescriptorOp::fold(FoldAdaptor adaptor) {
  return foldDescriptorOp(*this, adaptor);
}

//===----------------------------------------------------------------------===//
// ScaledMFMAOp
//===----------------------------------------------------------------------===//

namespace {
/// Check if the scales input is used in other scaled mfma's while they exist.
/// If theyre unused then pack the scales.
struct PackScales final : OpRewritePattern<ScaledMFMAOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ScaledMFMAOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto setOpsel = [&op](unsigned idx, int64_t val) {
      switch (idx) {
      case 3:
        op.setScalesIdxA(val);
        break;
      case 4:
        op.setScalesIdxB(val);
        break;
      default:
        break;
      }
    };

    // For every scale operand of this ScaledMFMAOp, if the scale is produced by
    // the extraction of a single scale from some vector, then attempt to
    // extract 4 values from that vector instead.
    //
    // Example: (f8 here means f8E8M0FNU)
    // %unit = vector.extract %ScaleSrc[offsets] : f8 from vector<...>
    // %scale = vector.insert %unit, ... : f8 into vector<4xf8>
    // amdgpu.scaled_mfma(%scale[0] * ...
    //
    // rewrite to:
    //
    // %reshaped = vector.shape_cast %ScaleSrc : vector<...> to vector<?xf8>
    // %scale = vector.extract %reshaped[?] : vector<4xf8> from vector<?xf8>
    // amdgpu.scaled_mfma(%scale[0-3] * ...
    //
    // This creates duplicate shape_casts for every use but these will be
    // removed in CSE.
    for (auto opIdx : std::array<int64_t, 2>({3, 4})) {
      auto insertOp = op.getOperand(opIdx).getDefiningOp<vector::InsertOp>();
      if (!insertOp) {
        return rewriter.notifyMatchFailure(op,
                                           "defining op not a vector.insert");
      }
      // If the extracted value is not a single scalar, then it has been packed.
      if (isa<VectorType>(insertOp.getValueToStore().getType())) {
        return rewriter.notifyMatchFailure(
            op, "scaled mfma operand already packed");
      }

      auto extractOp =
          insertOp.getValueToStore().getDefiningOp<vector::ExtractOp>();
      if (!extractOp) {
        return rewriter.notifyMatchFailure(op,
                                           "defining op not a vector.extract");
      }

      Value scaleSrc = extractOp.getOperand(0);
      auto scaleSrcType = dyn_cast<VectorType>(scaleSrc.getType());
      if (!scaleSrcType) {
        return rewriter.notifyMatchFailure(op, "not a vector type");
      }

      // We do not handle dynamic dims yet, assume that the input is padded to
      // a static shape now.
      if (!scaleSrcType.hasStaticShape()) {
        return rewriter.notifyMatchFailure(op,
                                           "dynamic dims not yet supported");
      }

      int64_t numElements = scaleSrcType.getNumElements();
      if (numElements <= 4) {
        return rewriter.notifyMatchFailure(
            op, "no packing if # of scales less than four");
      }

      // Find a linearized idx using the size and offsets of the extract op.
      auto extractedPos = llvm::to_vector_of<int64_t>(
          llvm::reverse(extractOp.getStaticPosition()));
      ArrayRef<int64_t> scaleSrcShape = scaleSrcType.getShape();
      int64_t scaleSrcRank = scaleSrcType.getRank();
      SmallVector<int64_t> extractSizes(scaleSrcRank, 1);
      for (int64_t i = 1; i < scaleSrcRank; ++i) {
        extractSizes[i] = extractSizes[i - 1] * scaleSrcShape[scaleSrcRank - i];
      }
      int64_t idx = linearize(extractedPos, extractSizes);

      // All n scales (where n is the total number of scales) must now be
      // extracted in chunks of 4 elements. This is done by dividing the
      // original vector of scales into groups of 4 elements
      // at offsets 0, 4, ..., m (where m = n/4). All extractions of a
      // scale at a particular index are now replaced with an extraction
      // of the entire group of 4 elements to which that index belongs.
      //
      // If the number of scales happens to be indivisible by 4, extract
      // the remaining n - m scales in a chunk of 4 elements starting at
      // offset n - 4.
      int64_t offset = idx - (idx % 4);
      int64_t opsel = idx - offset;
      int64_t size = 4l;
      // Accomdate remaining elements in the case of non-4-divisible vectors.
      if (numElements - offset < size) {
        opsel = size - (numElements - idx);
        offset = numElements - 4l;
      }
      Type scaleSrcElemType = scaleSrcType.getElementType();
      auto newSrcType =
          VectorType::get(ArrayRef{numElements}, scaleSrcElemType);
      Value newScaleSrc =
          vector::ShapeCastOp::create(rewriter, loc, newSrcType, scaleSrc);
      auto extract = vector::ExtractStridedSliceOp::create(
          rewriter, loc, newScaleSrc, ArrayRef{offset}, ArrayRef{size},
          ArrayRef{int64_t(1)});
      rewriter.modifyOpInPlace(op, [&] {
        op->setOperand(opIdx, extract);
        setOpsel(opIdx, opsel);
      });
    }
    return success();
  }
};
} // namespace

void ScaledMFMAOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<PackScales>(context);
}

#include "mlir/Dialect/AMDGPU/IR/AMDGPUEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AMDGPU/IR/AMDGPUAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/AMDGPU/IR/AMDGPUTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/AMDGPU/IR/AMDGPU.cpp.inc"
