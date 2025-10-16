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
// WMMAOp
//===----------------------------------------------------------------------===//
LogicalResult WMMAOp::verify() {
  Type sourceAType = getSourceA().getType();
  Type sourceBType = getSourceB().getType();
  Type destType = getDestC().getType();

  VectorType sourceVectorAType = dyn_cast<VectorType>(sourceAType);
  VectorType sourceVectorBType = dyn_cast<VectorType>(sourceBType);
  VectorType destVectorType = dyn_cast<VectorType>(destType);

  Type sourceAElemType = sourceVectorAType.getElementType();
  Type sourceBElemType = sourceVectorBType.getElementType();
  Type destElemType = destVectorType.getElementType();

  if (sourceVectorAType.getNumElements() !=
      sourceVectorBType.getNumElements()) {
    return emitOpError("source vectors have different lengths: ")
           << sourceVectorAType << " vs. " << sourceVectorBType;
  }

  bool isDestFloat = isa<Float32Type, Float16Type, BFloat16Type>(destElemType);
  bool isSrcFloat =
      isa<Float16Type, BFloat16Type, Float8E4M3FNType, Float8E5M2Type>(
          sourceAElemType);

  if (isDestFloat && !isSrcFloat) {
    return emitOpError("Expected float sources with float destination");
  }

  if (!isDestFloat && isSrcFloat) {
    return emitOpError("Expected int sources with int destination");
  }

  if (sourceAElemType != sourceBElemType &&
      !(isa<Float8E5M2Type, Float8E4M3FNType>(sourceAElemType) &&
        isa<Float8E5M2Type, Float8E4M3FNType>(sourceBElemType))) {
    return emitOpError(
               "source element types much match (except for fp8) but have ")
           << sourceAType << " and " << sourceBType;
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

//===----------------------------------------------------------------------===//
// GatherToLDSOp
//===----------------------------------------------------------------------===//

LogicalResult GatherToLDSOp::verify() {
  MemRefType srcType = cast<MemRefType>(getSrc().getType());
  MemRefType dstType = cast<MemRefType>(getDst().getType());

  if (!dstType.areTrailingDimsContiguous(1))
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
  if (validNumElems == kValidLoadSizeMap.end()) {
    return emitOpError("Unsupported element type size for transpose load: ")
           << elementTypeSize << " bits";
  }
  if (numElements != validNumElems->second) {
    return emitOpError(
               "Transferring type size mismatch: expected num of elements: ")
           << validNumElems->second;
  }

  return success();
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
      auto newSrcType = VectorType::get(SmallVector<int64_t>({numElements}),
                                        scaleSrcElemType);
      Value newScaleSrc =
          vector::ShapeCastOp::create(rewriter, loc, newSrcType, scaleSrc);
      auto extract = vector::ExtractStridedSliceOp::create(
          rewriter, loc, newScaleSrc, ArrayRef<int64_t>{offset},
          ArrayRef<int64_t>{size}, ArrayRef<int64_t>{1});
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

#define GET_OP_CLASSES
#include "mlir/Dialect/AMDGPU/IR/AMDGPU.cpp.inc"
