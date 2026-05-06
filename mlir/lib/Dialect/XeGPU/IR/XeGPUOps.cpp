//===- XeGPUOps.cpp - MLIR XeGPU ops implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "xegpu"

using namespace mlir;
using namespace mlir::xegpu;

template <typename T>
static std::string makeString(T array, bool breakline = false) {
  std::string buf;
  buf.clear();
  llvm::raw_string_ostream os(buf);
  os << "[";
  for (size_t i = 1; i < array.size(); i++) {
    os << array[i - 1] << ", ";
    if (breakline)
      os << "\n\t\t";
  }
  os << array.back() << "]";
  return buf;
}

static SmallVector<int64_t> getShapeOf(Type type) {
  SmallVector<int64_t> shape;
  if (auto ty = llvm::dyn_cast<ShapedType>(type))
    shape = SmallVector<int64_t>(ty.getShape());
  else
    shape.push_back(1);
  return shape;
}

static bool isReadHintOrNone(const CachePolicyAttr &attr) {
  if (!attr)
    return true;
  auto kind = attr.getValue();
  return kind == CachePolicy::CACHED || kind == CachePolicy::UNCACHED ||
         kind == CachePolicy::STREAMING || kind == CachePolicy::READ_INVALIDATE;
}

static bool isWriteHintOrNone(const CachePolicyAttr &attr) {
  if (!attr)
    return true;
  auto kind = attr.getValue();
  return kind == CachePolicy::CACHED || kind == CachePolicy::UNCACHED ||
         kind == CachePolicy::WRITE_BACK || kind == CachePolicy::WRITE_THROUGH;
}

static LogicalResult
isValidGatherScatterBufferParams(Type offsetsTy, Type maskTy,
                                 VectorType valueTy, int64_t chunkSize,
                                 function_ref<InFlightDiagnostic()> emitError) {

  auto maskVecTy = dyn_cast<VectorType>(maskTy);
  auto offsetsVecTy = dyn_cast<VectorType>(offsetsTy);
  if (!valueTy) {
    if (chunkSize > 1)
      return emitError() << "Expecting chunk size == 1 for scalar result";
    if (maskVecTy || offsetsVecTy)
      return emitError() << "Expecting scalar mask and offsets.";
    else if (maskVecTy && offsetsVecTy)
      return emitError() << "Expecting a vector type result.";
    return success();
  }

  auto valueSize = valueTy.getNumElements();
  // SIMT mode with scalar mask and offsets.
  if (!maskVecTy && !offsetsVecTy) {
    if (valueSize != chunkSize)
      return emitError() << "value elements must match chunk size "
                         << chunkSize;
    return success();
  }
  auto maskShape = getShapeOf(maskTy);
  auto valueShape = getShapeOf(valueTy);

  if (!maskVecTy)
    return emitError() << "Expecting a vector type mask.";
  int64_t maskSize = maskVecTy.getNumElements();

  if (chunkSize > 1) {
    if ((valueTy.getRank() == 1) && (valueSize != chunkSize))
      return emitError() << "value elements must match chunk size "
                         << chunkSize;
  } else {
    if (valueSize != maskSize)
      return emitError()
             << "Mask should match value except the chunk size dim.";
  }
  llvm::SmallVector<int64_t> expectedMaskShape(valueShape);
  if (maskSize == 1)
    return success();
  if (chunkSize > 1)
    expectedMaskShape.pop_back();
  if (expectedMaskShape != maskShape)
    return emitError() << "Mask should match value except the chunk size dim.";

  return success();
}

LogicalResult
IsValidMatrixOpParams(VectorType dataTy, MemDescType mdescTy,
                      UnitAttr subgroup_block_io, DistributeLayoutAttr layout,
                      function_ref<InFlightDiagnostic()> emitError) {

  if (!dataTy) {
    if (subgroup_block_io)
      return emitError() << "subgroup_block_io "
                            "are only allowed when result is a VectorType.";
    else
      return success();
  }

  if (mdescTy.getRank() < 2)
    return emitError() << "mem_desc must be 2D or greater.";

  ArrayRef<int64_t> dataShape = dataTy.getShape();
  ArrayRef<int64_t> mdescShape = mdescTy.getShape();

  SmallVector<int64_t> blockShape = mdescTy.getBlockShape();
  ArrayAttr strideAttr = mdescTy.getStrideAttr();
  SmallVector<int64_t> strides;
  for (Attribute attr : strideAttr.getValue()) {
    strides.push_back(cast<IntegerAttr>(attr).getInt());
  }
  if (subgroup_block_io && layout) {
    auto laneData = layout.getEffectiveLaneDataAsInt();
    auto laneLayout = layout.getEffectiveLaneLayoutAsInt();
    if (!laneData.empty()) {
      bool isLaneDataContiguous =
          std::all_of(laneData.begin(), std::prev(laneData.end()),
                      [](int x) { return x == 1; });
      if (!isLaneDataContiguous)
        return emitError() << "With subgroup_block_io, accessed data must be "
                              "contiguous and coalesced.";
      for (size_t i = 0; i < laneData.size(); ++i) {
        if (laneLayout[i] != blockShape[i])
          return emitError() << "With subgroup_block_io, the block shape must "
                                "match the lane layout.";
        if (laneLayout[i] != 1 && strides[i] != 1)
          return emitError() << "With subgroup_block_io, the distributed "
                                "dimensions must be contiguous.";
      }
    }
  }

  if (layout && !layout.isDistributable(
                    SmallVector<int64_t>(dataShape.begin(), dataShape.end())))
    return emitError() << "Value shape is not distributable with the layout";

  if (dataShape.size() == 2) {
    if (llvm::any_of(llvm::zip_equal(dataShape, mdescShape),
                     [](auto p) { return std::get<0>(p) > std::get<1>(p); }))
      return emitError() << "data shape must not exceed mem_desc shape.";
  } else {
    // if the subgroup_block_io attribute is set,  mdescTy must have block
    // attribute
    if (subgroup_block_io && !blockShape.size())
      return emitError() << "mem_desc must have block attribute when "
                            "subgroup_block_io is set.";
    // if the subgroup_block_io attribute is set, the memdesc should be row
    // major
    if (subgroup_block_io && mdescTy.isColMajor())
      return emitError() << "mem_desc should be row major when "
                            "subgroup_block_io is set.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_CreateNdDescOp
//===----------------------------------------------------------------------===//

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, TypedValue<MemRefType> source) {
  [[maybe_unused]] auto ty = source.getType();
  assert(ty.hasStaticShape() && "expecting a memref with static shape");

  build(builder, state, tdesc, source, ValueRange({}) /* empty dynamic shape */,
        ValueRange({}) /* empty dynamic strides */,
        DenseI64ArrayAttr({}) /* empty const shape*/,
        DenseI64ArrayAttr({}) /* empty const strides*/);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, Value source,
                           llvm::ArrayRef<OpFoldResult> shape,
                           llvm::ArrayRef<OpFoldResult> strides) {
  Type srcTy = source.getType();
  assert((isa<IntegerType, MemRefType>(srcTy)) &&
         "Source has to be either int or memref.");

  llvm::SmallVector<Value> dynamicShape;
  llvm::SmallVector<Value> dynamicStrides;

  llvm::SmallVector<int64_t> staticShape;
  llvm::SmallVector<int64_t> staticStrides;

  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  auto staticShapeAttr = builder.getDenseI64ArrayAttr(staticShape);
  auto staticStridesAttr = builder.getDenseI64ArrayAttr(staticStrides);

  if (auto memrefTy = dyn_cast<MemRefType>(srcTy)) {
    auto memrefShape = memrefTy.getShape();
    auto [memrefStrides, _] = memrefTy.getStridesAndOffset();

    // if shape and strides are from Memref, we don't need attributes for them
    // to keep the IR print clean (only do so for full-static case, otherwise
    // printer would fail trying to print empty array-attr).
    if (staticShape == memrefShape && staticStrides == memrefStrides &&
        dynamicShape.empty() && dynamicStrides.empty()) {
      staticShapeAttr = DenseI64ArrayAttr();
      staticStridesAttr = DenseI64ArrayAttr();
    }
  }

  build(builder, state, tdesc, source, dynamicShape, dynamicStrides,
        staticShapeAttr, staticStridesAttr);
}

LogicalResult CreateNdDescOp::verify() {
  size_t rank = getMixedSizes().size();
  bool invalidRank = rank != getMixedStrides().size();
  bool invalidElemTy = false;

  // Memory space of created TensorDesc should match with the source.
  // Both source and TensorDesc are considered for global memory by default,
  // if the memory scope attr is not specified. If source is an integer,
  // it is considered as ptr to global memory.
  auto srcMemorySpace = getSourceMemorySpace();
  auto tdescMemorySpace = static_cast<unsigned>(getType().getMemorySpace());
  if (srcMemorySpace != tdescMemorySpace)
    return emitOpError("Memory space mismatch.")
           << " Source: " << srcMemorySpace
           << ", TensorDesc: " << tdescMemorySpace;

  // check source type matches the rank if it is a memref.
  // It also should have the same ElementType as TensorDesc.
  if (auto memrefTy = dyn_cast<MemRefType>(getSourceType()))
    invalidElemTy |= memrefTy.getElementType() != getElementType();

  if (llvm::isa<IntegerType>(getSourceType())) {
    // strides and shape must present for integer source.
    if (getMixedStrides().empty() || getMixedSizes().empty())
      return emitOpError("expecting strides and shape to be present for "
                         "integer source.");
  }

  if (invalidRank)
    return emitOpError(
        "Expecting the rank of shape, strides, and source (if source "
        "is a memref) should match with each other.");

  // check result TensorDesc rank
  if (getType().getRank() > (int64_t)rank)
    return emitOpError("Expecting the TensorDesc rank is not greater than the "
                       "ranks of shape, strides or the memref source.");

  if (invalidElemTy)
    return emitOpError("TensorDesc should have the same element "
                       "type with the source if it is a memref.\n");

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_PrefetchNdOp
//===----------------------------------------------------------------------===//

void PrefetchNdOp::build(OpBuilder &builder, OperationState &state,
                         Value tensorDesc, ArrayRef<OpFoldResult> offsets,
                         xegpu::CachePolicyAttr l1_hint,
                         xegpu::CachePolicyAttr l2_hint,
                         xegpu::CachePolicyAttr l3_hint,
                         xegpu::DistributeLayoutAttr layout) {
  SmallVector<Value> dynamicOffsets;
  SmallVector<int64_t> staticOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  auto staticOffsetsAttr = builder.getDenseI64ArrayAttr(staticOffsets);

  build(builder, state, tensorDesc, dynamicOffsets, staticOffsetsAttr, l1_hint,
        l2_hint, l3_hint, /*anchor_layout=*/layout);
}

LogicalResult PrefetchNdOp::verify() {
  auto tdescTy = getTensorDescType();

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  int64_t tDescRank = tdescTy.getRank();
  int64_t offsetSize = getMixedOffsets().size();
  if (offsetSize != tDescRank)
    return emitOpError(
        "Mismatched ranks between offsets and tensor descriptor");

  if (auto layout = getAnchorLayout()) {
    if (!layout.isDistributable(getShapeOf(tdescTy)))
      return emitOpError(
          "TensorDesc shape is not distributable with the layout");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadNdOp
//===----------------------------------------------------------------------===//

void LoadNdOp::build(OpBuilder &builder, OperationState &state, Type retType,
                     Value tensorDesc, ArrayRef<OpFoldResult> offsets,
                     UnitAttr packed, DenseI64ArrayAttr transpose,
                     xegpu::CachePolicyAttr l1_hint,
                     xegpu::CachePolicyAttr l2_hint,
                     xegpu::CachePolicyAttr l3_hint,
                     xegpu::DistributeLayoutAttr layout) {
  SmallVector<Value> dynamicOffsets;
  SmallVector<int64_t> staticOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  auto staticOffsetsAttr = builder.getDenseI64ArrayAttr(staticOffsets);

  build(builder, state, retType, tensorDesc, dynamicOffsets, staticOffsetsAttr,
        packed, transpose, l1_hint, l2_hint, l3_hint,
        /*anchor_layout=*/layout);
}

LogicalResult LoadNdOp::verify() {
  auto tdescTy = getTensorDescType();
  auto valueTy = getType();

  if (tdescTy.getRank() > 2)
    return emitOpError("Expects a 1D or 2D TensorDesc.\n");

  if (!valueTy)
    return emitOpError("Invalid result, it should be a VectorType.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  int tdescElems = tdescTy.getNumElements() * tdescTy.getArrayLength();
  int valueElems = valueTy.getNumElements();

  // If the result vector is 1D and has less elements than the tensor
  // descriptor, it is supposed to be a SIMT op. The layout attribute in
  // tensor_desc is not needed.
  if (valueElems < tdescElems && valueTy.getRank() == 1) {
    // SIMT mode doesn't need LayoutAttr.
    if (tdescTy.getLayoutAttr())
      return emitOpError()
             << "TensorDesc doesn't need LayoutAttr for SIMT code";

    // For SIMT code, the load is evenly distributed across all lanes in a
    // subgroup. Since subgroup size is arch dependent, we only check even
    // distribution here.
    if (tdescElems % valueElems)
      return emitOpError()
             << "Result shape " << makeString(getShapeOf(valueTy))
             << " is not a valid distribution for tensor descriptor "
             << tdescTy;

    return success();
  }

  // Check SIMD mode.
  auto tdescShape = getShapeOf(tdescTy);
  auto valueShape = getShapeOf(valueTy);

  if (getTranspose()) {
    auto trans = getTranspose().value();
    // Make sure the transpose value is valid, and apply it
    if (llvm::all_of(trans, [&](size_t s) { return s < tdescShape.size(); }))
      tdescShape = applyPermutation(tdescShape, trans);
    else
      mlir::emitWarning(getLoc()) << "Invalid transpose attr. It is ignored.";
  }

  if (getPacked()) {
    if (tdescTy.getRank() == 2) {
      const int axis = 0;
      auto vnni_factor = valueShape.back();
      tdescShape[axis] /= vnni_factor;
      tdescShape.push_back(vnni_factor);
    } else {
      mlir::emitWarning(getLoc())
          << "Invalid Packed Attr. It is ignored (available for 2D "
             "TensorDesc only).";
    }
  }

  // Handle array_length. Two result shape conventions are accepted:
  //   * 3D shape: leading array_length dimension prepended, e.g. descriptor
  //     16x16 with array_length=2 -> [2, 16, 16].
  //   * Stacked 2D shape: array blocks stacked along the non-FCD (first)
  //     dimension, e.g. descriptor 16x16 with array_length=2 -> [32, 16].
  auto array_len = tdescTy.getArrayLength();
  SmallVector<int64_t> stacked2DShape(tdescShape);
  SmallVector<int64_t> threeDShape(tdescShape);
  if (array_len > 1 && !tdescShape.empty()) {
    stacked2DShape[0] *= array_len;
    threeDShape.insert(threeDShape.begin(), array_len);
  }

  if (valueShape != stacked2DShape && valueShape != threeDShape)
    return emitOpError() << "Result shape " << makeString(valueShape)
                         << " is not consistent with tensor descriptor "
                         << tdescTy;

  int64_t tDescRank = tdescTy.getRank();
  int64_t offsetSize = getMixedOffsets().size();
  if (offsetSize != tDescRank)
    return emitOpError(
        "Mismatched ranks between offsets and tensor descriptor");

  if (auto layout = getAnchorLayout()) {
    if (!layout.isDistributable(getShapeOf(tdescTy)))
      return emitOpError(
          "TensorDesc shape is not distributable with the layout");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreNdOp
//===----------------------------------------------------------------------===//

void StoreNdOp::build(OpBuilder &builder, OperationState &state, Value value,
                      Value tensorDesc, ArrayRef<OpFoldResult> offsets,
                      xegpu::CachePolicyAttr l1_hint,
                      xegpu::CachePolicyAttr l2_hint,
                      xegpu::CachePolicyAttr l3_hint,
                      xegpu::DistributeLayoutAttr layout) {
  SmallVector<Value> dynamicOffsets;
  SmallVector<int64_t> staticOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  auto staticOffsetsAttr = builder.getDenseI64ArrayAttr(staticOffsets);

  build(builder, state, value, tensorDesc, dynamicOffsets, staticOffsetsAttr,
        l1_hint, l2_hint, l3_hint, /*anchor_layout=*/layout);
}

LogicalResult StoreNdOp::verify() {
  auto dstTy = getTensorDescType(); // Tile
  auto valTy = getValueType();      // Vector

  if (dstTy.getRank() > 2)
    return emitOpError("Expects a 1D or 2D TensorDesc.\n");

  if (!valTy)
    return emitOpError("Expecting a VectorType result.\n");

  if (!isWriteHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isWriteHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isWriteHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  auto array_len = dstTy.getArrayLength();
  if (array_len > 1)
    return emitOpError("array length is not supported by store_nd.\n");

  auto tdescElems = dstTy.getNumElements();
  auto valueElems = valTy.getNumElements();

  // Similar to LoadNdOp, if the value vector is 1D and has less elements than
  // the tensor descriptor, it is supposed to be a SIMT op. The layout attribute
  // in tensor_desc is not needed.
  if (valTy.getRank() == 1 && valueElems < tdescElems) {
    // SIMT mode doesn't need LayoutAttr.
    if (dstTy.getLayoutAttr())
      return emitOpError()
             << "TensorDesc doesn't need LayoutAttr for SIMT code";

    if (tdescElems % valueElems)
      return emitOpError()
             << "Value shape " << makeString(getShapeOf(valTy))
             << " is not a valid distribution for tensor descriptor " << dstTy;

    return success();
  }

  // SIMD code should have the same shape as the tensor descriptor.
  auto tdescShape = getShapeOf(dstTy);
  auto valueShape = getShapeOf(valTy);
  if (tdescShape != valueShape)
    return emitOpError() << "Value shape " << makeString(valueShape)
                         << " is not consistent with tensor descriptor "
                         << dstTy;

  int64_t tDescRank = dstTy.getRank();
  int64_t offsetSize = getMixedOffsets().size();
  if (offsetSize != tDescRank)
    return emitOpError(
        "Mismatched ranks between offsets and tensor descriptor");

  if (auto layout = getAnchorLayout()) {
    if (!layout.isDistributable(tdescShape))
      return emitOpError(
          "TensorDesc shape is not distributable with the layout");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_PrefetchOp
//===----------------------------------------------------------------------===//
LogicalResult PrefetchOp::verify() {
  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  auto srcTy = getSourceType();
  if (srcTy.isInteger() && !getOffsetAlignByteAttr())
    return emitOpError("offset_align_byte is required with integer source.");

  if (getOffsetAlignByteAttr() && !srcTy.isInteger())
    return emitOpError("offset_align_byte only allowed with integer source.");

  if (auto layout = getAnchorLayout()) {
    // get the offset operand and its shape
    auto offsetsTy = getOffsets().getType();
    if (llvm::isa<VectorType>(offsetsTy) &&
        !layout.isDistributable(getShapeOf(offsetsTy)))
      return emitOpError("offset shape is not distributable with the layout");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadGatherOp
//===----------------------------------------------------------------------===//
LogicalResult LoadGatherOp::verify() {
  auto maskTy = getMaskType();
  auto valueTy = getValueType();

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  auto srcTy = getSourceType();
  uint64_t chunkSize = static_cast<int64_t>(getChunkSize().value_or(1));
  auto memTy = dyn_cast<MemRefType>(srcTy);

  if (memTy && (getElementType() != memTy.getElementType()))
    return emitError() << "Value should have the same element type as MemRef.";

  if (auto layout = getAnchorLayout()) {
    if (!layout.isDistributable(getShapeOf(valueTy)))
      return emitOpError("Value shape is not distributable with the layout");
  }

  auto offsetsTy = getOffsets().getType();
  return isValidGatherScatterBufferParams(offsetsTy, maskTy, valueTy, chunkSize,
                                          [&]() { return emitOpError(); });
}

void LoadGatherOp::build(OpBuilder &builder, OperationState &state,
                         Type valueType, Value source,
                         ArrayRef<OpFoldResult> offsets, Value mask,
                         IntegerAttr chunk_size, xegpu::CachePolicyAttr l1_hint,
                         xegpu::CachePolicyAttr l2_hint,
                         xegpu::CachePolicyAttr l3_hint) {
  auto loc = source.getLoc();
  int64_t size = static_cast<int64_t>(offsets.size());
  auto type = VectorType::get(size, builder.getIndexType());
  auto values = getValueOrCreateConstantIndexOp(builder, loc, offsets);
  auto offset = vector::FromElementsOp::create(builder, loc, type, values);

  build(builder, state, valueType, source, offset, mask, chunk_size, l1_hint,
        l2_hint, l3_hint, /*anchor_layout=*/nullptr);
}

void LoadGatherOp::build(OpBuilder &builder, OperationState &state,
                         Type valueType, Value source,
                         ArrayRef<OpFoldResult> offsets, Value mask,
                         IntegerAttr chunk_size, xegpu::CachePolicyAttr l1_hint,
                         xegpu::CachePolicyAttr l2_hint,
                         xegpu::CachePolicyAttr l3_hint,
                         DistributeLayoutAttr layout) {
  auto loc = source.getLoc();
  int64_t size = static_cast<int64_t>(offsets.size());
  auto type = VectorType::get(size, builder.getIndexType());
  auto values = getValueOrCreateConstantIndexOp(builder, loc, offsets);
  auto offset = vector::FromElementsOp::create(builder, loc, type, values);

  build(builder, state, valueType, source, offset, mask, chunk_size, l1_hint,
        l2_hint, l3_hint, layout);
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreScatterOp
//===----------------------------------------------------------------------===//
LogicalResult StoreScatterOp::verify() {
  auto maskTy = getMaskType();
  auto valueTy = getValueType();

  if (!isWriteHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isWriteHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isWriteHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  auto destTy = getDestType();
  uint64_t chunkSize = static_cast<int64_t>(getChunkSize().value_or(1));
  auto memTy = dyn_cast<MemRefType>(destTy);

  if (memTy && (getElementType() != memTy.getElementType()))
    return emitError() << "Value should have the same element type as MemRef.";

  if (auto layout = getAnchorLayout()) {
    if (!layout.isDistributable(getShapeOf(valueTy)))
      return emitOpError("Value shape is not distributable with the layout");
  }

  auto offsetsTy = getOffsets().getType();
  return isValidGatherScatterBufferParams(offsetsTy, maskTy, valueTy, chunkSize,
                                          [&]() { return emitOpError(); });
}

void StoreScatterOp::build(OpBuilder &builder, OperationState &state,
                           Value value, Value dest,
                           ArrayRef<OpFoldResult> offsets, Value mask,
                           IntegerAttr chunk_size,
                           xegpu::CachePolicyAttr l1_hint,
                           xegpu::CachePolicyAttr l2_hint,
                           xegpu::CachePolicyAttr l3_hint) {
  auto loc = dest.getLoc();
  int64_t size = static_cast<int64_t>(offsets.size());
  auto type = VectorType::get(size, builder.getIndexType());
  auto values = getValueOrCreateConstantIndexOp(builder, loc, offsets);
  auto offset = vector::FromElementsOp::create(builder, loc, type, values);

  // Call the correct builder overload that does not expect result types.
  build(builder, state, value, dest, offset, mask, chunk_size, l1_hint, l2_hint,
        l3_hint, /*anchor_layout=*/nullptr);
}

void StoreScatterOp::build(
    OpBuilder &builder, OperationState &state, Value value, Value dest,
    ArrayRef<OpFoldResult> offsets, Value mask, IntegerAttr chunk_size,
    xegpu::CachePolicyAttr l1_hint, xegpu::CachePolicyAttr l2_hint,
    xegpu::CachePolicyAttr l3_hint, DistributeLayoutAttr layout) {
  auto loc = dest.getLoc();
  int64_t size = static_cast<int64_t>(offsets.size());
  auto type = VectorType::get(size, builder.getIndexType());
  auto values = getValueOrCreateConstantIndexOp(builder, loc, offsets);
  auto offset = vector::FromElementsOp::create(builder, loc, type, values);

  // Call the correct builder overload that does not expect result types.
  build(builder, state, value, dest, offset, mask, chunk_size, l1_hint, l2_hint,
        l3_hint, layout);
}

//===----------------------------------------------------------------------===//
// DPAS Common Verification Helpers
//===----------------------------------------------------------------------===//

// Helper to verify layout distributability for a value
static LogicalResult
verifyLayoutDistributable(Operation *op,
                          std::optional<DistributeLayoutAttr> layout,
                          ArrayRef<int64_t> shape, StringRef operandName) {
  if (layout && !layout->isDistributable(
                    SmallVector<int64_t>(shape.begin(), shape.end())))
    return op->emitOpError(operandName)
           << " shape is not distributable with the layout";
  return success();
}

// Helper to verify M, N, K dimensions match between A, B, and result matrices
static LogicalResult verifyDpasDimensions(Operation *op,
                                          ArrayRef<int64_t> aShape,
                                          ArrayRef<int64_t> bShape,
                                          ArrayRef<int64_t> resShape) {

  auto aRank = aShape.size();
  auto bRank = bShape.size();
  auto resRank = resShape.size();
  if (aRank == 1 && bRank == 1 && resRank == 1)
    return success();

  // Validate A and B are 2D
  if (aRank != 2)
    return op->emitOpError("A operand must be a 2D vector.");
  if (bRank < 2 || bRank > 3)
    return op->emitOpError("B operand must be a 2D or 3D vector.");
  if (resRank != 2)
    return op->emitOpError("Result must be a 2D vector.");

  // Calculate effective K dimension for B (handle 3D packed case)
  int64_t bK = bRank == 3 ? bShape[0] * bShape[2] : bShape[0];

  // Verify K dimension match between A and B
  if (bK != aShape[1])
    return op->emitOpError("K-dimension mismatch: A has K=")
           << aShape[1] << " but B has K=" << bK << ".";

  // Verify M dimension match between A and result
  if (aShape[0] != resShape[0])
    return op->emitOpError("M-dimension mismatch: A has M=")
           << aShape[0] << " but result has M=" << resShape[0] << ".";

  // Verify N dimension match between B and result
  if (bShape[1] != resShape[1])
    return op->emitOpError("N-dimension mismatch: B has N=")
           << bShape[1] << " but result has N=" << resShape[1] << ".";

  return success();
}

// Helper to verify accumulator matches result type
static LogicalResult verifyDpasAccumulator(Operation *op, Type accType,
                                           Type resultType) {
  if (accType != resultType)
    return op->emitOpError("Accumulator type must match result type.");
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_DpasOp
//===----------------------------------------------------------------------===//
LogicalResult DpasOp::verify() {
  auto lhsShape = getLhsType().getShape();
  auto rhsShape = getRhsType().getShape();
  auto resShape = getResultType().getShape();

  // Verify layout distributability
  if (failed(
          verifyLayoutDistributable(*this, getLayoutCd(), resShape, "Result")))
    return failure();
  if (failed(verifyLayoutDistributable(*this, getLayoutA(), lhsShape, "A")))
    return failure();
  if (failed(verifyLayoutDistributable(*this, getLayoutB(), rhsShape, "B")))
    return failure();

  // Verify accumulator if present
  if (getAcc() &&
      failed(verifyDpasAccumulator(*this, getAcc().getType(), getResultType())))
    return failure();

  return verifyDpasDimensions(*this, lhsShape, rhsShape, resShape);
}

//===----------------------------------------------------------------------===//
// XeGPU_ConvertLayoutOp
//===----------------------------------------------------------------------===//
LogicalResult ConvertLayoutOp::verify() {
  auto srcLayout = getInputLayout();
  auto resLayout = getTargetLayout();
  if (!srcLayout)
    return emitOpError("expected input layout.");
  if (!resLayout)
    return emitOpError("expected target layout.");

  // both input and target layouts should be WgLayout or SgLayout at the same
  // time.
  if ((!srcLayout.isForWorkgroup() || !resLayout.isForWorkgroup()) &&
      (!srcLayout.isForSubgroup() || !resLayout.isForSubgroup()))
    return emitOpError("expected input layout and target layout be WgLayout or "
                       "SgLayout at the same time.");

  Type srcType = getSource().getType();
  if (llvm::isa<VectorType>(srcType)) {
    SmallVector<int64_t> shape(llvm::cast<VectorType>(srcType).getShape());
    if (!srcLayout.isDistributable(shape))
      return emitOpError(
          "invalid input layout, data cannot be evenly distributed.");

    if (!resLayout.isDistributable(shape))
      return emitOpError(
          "invalid target layout, data cannot be evenly distributed.");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadMatrixOp
//===----------------------------------------------------------------------===//
void LoadMatrixOp::build(OpBuilder &builder, OperationState &state, Type res,
                         TypedValue<MemDescType> memDesc,
                         llvm::ArrayRef<OpFoldResult> offsets,
                         DistributeLayoutAttr layout) {
  llvm::SmallVector<Value> dynamicOffsets;
  llvm::SmallVector<int64_t> staticOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  auto staticOffsetsAttr = builder.getDenseI64ArrayAttr(staticOffsets);
  // Call the generated builder with all parameters (including optional ones as
  // nullptr/empty)
  build(builder, state, res, memDesc, dynamicOffsets, staticOffsetsAttr,
        /*subgroup_block_io=*/nullptr, layout);
}

LogicalResult LoadMatrixOp::verify() {

  auto resTy = dyn_cast<VectorType>(getRes().getType());
  UnitAttr subgroup_block_io = getSubgroupBlockIoAttr();
  MemDescType mdescTy = getMemDesc().getType();

  return IsValidMatrixOpParams(resTy, mdescTy, subgroup_block_io,
                               getLayoutAttr(), [&]() { return emitError(); });
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreMatrixOp
//===----------------------------------------------------------------------===//
void StoreMatrixOp::build(OpBuilder &builder, OperationState &state, Value data,
                          TypedValue<MemDescType> memDesc,
                          llvm::ArrayRef<OpFoldResult> offsets,
                          DistributeLayoutAttr layout) {
  llvm::SmallVector<Value> dynamicOffsets;
  llvm::SmallVector<int64_t> staticOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  auto staticOffsetsAttr = builder.getDenseI64ArrayAttr(staticOffsets);
  build(builder, state, data, memDesc, dynamicOffsets, staticOffsetsAttr,
        /*subgroup_block_io=*/nullptr, layout);
}

LogicalResult StoreMatrixOp::verify() {

  auto dataTy = dyn_cast<VectorType>(getData().getType());
  UnitAttr subgroup_block_io = getSubgroupBlockIoAttr();
  MemDescType mdescTy = getMemDesc().getType();
  return IsValidMatrixOpParams(dataTy, mdescTy, subgroup_block_io,
                               getLayoutAttr(), [&]() { return emitError(); });
}

//===----------------------------------------------------------------------===//
// XeGPU_TruncfOp
//===----------------------------------------------------------------------===//

LogicalResult TruncfOp::verify() {
  auto sourceVecType = dyn_cast<VectorType>(getSource().getType());
  auto resultVecType = dyn_cast<VectorType>(getResult().getType());

  if (sourceVecType.getElementTypeBitWidth() <=
      resultVecType.getElementTypeBitWidth())
    return emitOpError("input type must be wider than result type.");

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_DpasMxOp
//===----------------------------------------------------------------------===//

LogicalResult DpasMxOp::verify() {
  auto aShape = getAType().getShape();
  auto bShape = getBType().getShape();
  auto resShape = getResultType().getShape();

  // Verify layout distributability for A, B, and result
  if (failed(
          verifyLayoutDistributable(*this, getLayoutCd(), resShape, "Result")))
    return failure();
  if (failed(verifyLayoutDistributable(*this, getLayoutA(), aShape, "A")))
    return failure();
  if (failed(verifyLayoutDistributable(*this, getLayoutB(), bShape, "B")))
    return failure();

  // Verify accumulator if present
  if (getAcc() &&
      failed(verifyDpasAccumulator(*this, getAcc().getType(), getResultType())))
    return failure();

  // Verify M, N, K dimensions
  if (failed(verifyDpasDimensions(*this, aShape, bShape, resShape)))
    return failure();

  // Validate scale_a if present
  if (getScaleA()) {
    auto scaleAVecType = dyn_cast<VectorType>(getScaleAType());
    // Only validate if scale is a vector (scalars are always valid)
    if (scaleAVecType) {
      auto scaleAShape = scaleAVecType.getShape();

      if (scaleAVecType.getRank() != 2)
        return emitOpError("Scale A must be a 2D vector when not a scalar.");

      // Verify layout distributability for scale_a
      if (failed(verifyLayoutDistributable(*this, getLayoutAScale(),
                                           scaleAShape, "ScaleA")))
        return failure();

      // Validate M dimension: scale_a[0] must match a[0]
      if (scaleAShape[0] != aShape[0])
        return emitOpError("Scale A M dimension [")
               << scaleAShape[0] << "] must match A M dimension [" << aShape[0]
               << "].";
    }
  }

  // Validate scale_b if present
  if (getScaleB()) {
    auto scaleBVecType = dyn_cast<VectorType>(getScaleBType());
    // Only validate if scale is a vector (scalars are always valid)
    if (scaleBVecType) {
      auto scaleBShape = scaleBVecType.getShape();

      if (scaleBVecType.getRank() != 2)
        return emitOpError("Scale B must be a 2D vector when not a scalar.");

      // Verify layout distributability for scale_b
      if (failed(verifyLayoutDistributable(*this, getLayoutBScale(),
                                           scaleBShape, "ScaleB")))
        return failure();

      // Validate N dimension: scale_b[1] must match b[1]
      if (scaleBShape[1] != bShape[1])
        return emitOpError("Scale B N dimension [")
               << scaleBShape[1] << "] must match B N dimension [" << bShape[1]
               << "].";
    }
  }

  // Validate scale K dimension compatibility if both scales are present and
  // vectors
  if (getScaleA() && getScaleB()) {
    auto scaleAVecType = dyn_cast<VectorType>(getScaleAType());
    auto scaleBVecType = dyn_cast<VectorType>(getScaleBType());

    if (scaleAVecType && scaleBVecType) {
      auto scaleAShape = scaleAVecType.getShape();
      auto scaleBShape = scaleBVecType.getShape();

      // Validate scale K dimension compatibility: scale_a[1] must match
      // scale_b[0]
      if (scaleAShape[1] != scaleBShape[0])
        return emitOpError("Scale K dimension mismatch: scale_a has K=")
               << scaleAShape[1] << " but scale_b has K=" << scaleBShape[0]
               << ".";
    }
  }

  return success();
}

namespace mlir {
#include <mlir/Dialect/XeGPU/IR/XeGPUAttrInterface.cpp.inc>
} // namespace mlir
#include <mlir/Dialect/XeGPU/IR/XeGPUEnums.cpp.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPU.cpp.inc>
