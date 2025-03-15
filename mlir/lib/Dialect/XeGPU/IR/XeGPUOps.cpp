//===- XeGPUOps.cpp - MLIR XeGPU ops implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "xegpu"

namespace mlir {
namespace xegpu {

static void transpose(llvm::ArrayRef<int64_t> trans,
                      SmallVector<int64_t> &shape) {
  SmallVector<int64_t> old = shape;
  for (size_t i = 0; i < trans.size(); i++)
    shape[i] = old[trans[i]];
}

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

static int64_t getRankOf(Value val) {
  auto type = val.getType();
  if (auto ty = llvm::dyn_cast<ShapedType>(type))
    return ty.getRank();
  return 0;
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

// Helper to validate value shape of LoadNd and StoreNd ops.
static LogicalResult
isArgShapesValid(TensorDescType tdescTy, VectorType valueTy,
                 ArrayRef<int64_t> adjustedTdescShape,
                 function_ref<InFlightDiagnostic()> emitError) {
  auto sgMap = tdescTy.getSGMapAttr();
  auto valueShape = valueTy.getShape();
  // sg_map not present means IR is in SIMD mode. In this case value shape must
  // match adjusted tensor descriptor shape.
  if (!sgMap)
    return valueShape == adjustedTdescShape
               ? success()
               : emitError()
                     << "Value shape " << makeString(valueShape)
                     << " is not consistent with tensor descriptor " << tdescTy;

  // sg_map present means IR is in SIMT mode. In this case sg_map determines the
  // value shape.
  auto expectedValueShapeOrFailure = tdescTy.getDistributedVectorType();
  assert(succeeded(expectedValueShapeOrFailure) &&
         "Failed to compute distributed vector shape for "
         "tensor descriptor ");

  return valueTy == expectedValueShapeOrFailure.value()
             ? success()
             : emitError()
                   << "Result shape " << makeString(valueShape)
                   << " is not consistent with distributed vector shape "
                   << makeString(expectedValueShapeOrFailure.value().getShape())
                   << " for tensor descriptor " << tdescTy;
}

//===----------------------------------------------------------------------===//
// XeGPU_CreateNdDescOp
//===----------------------------------------------------------------------===//
void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, TypedValue<MemRefType> source,
                           llvm::ArrayRef<OpFoldResult> offsets) {
  [[maybe_unused]] auto ty = source.getType();
  assert(ty.hasStaticShape() && offsets.size() == (size_t)ty.getRank());

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, tdesc, source, dynamicOffsets /* dynamic offsets */,
        ValueRange({}) /* empty dynamic shape */,
        ValueRange({}) /* empty dynamic strides */,
        staticOffsets /* const offsets */, {} /* empty const shape*/,
        {} /* empty const strides*/);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, TypedValue<MemRefType> source,
                           llvm::ArrayRef<OpFoldResult> offsets,
                           llvm::ArrayRef<OpFoldResult> shape,
                           llvm::ArrayRef<OpFoldResult> strides) {
  assert(shape.size() && offsets.size() && strides.size() &&
         shape.size() == strides.size() && shape.size() == offsets.size());

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<int64_t> staticShape;
  llvm::SmallVector<int64_t> staticStrides;
  llvm::SmallVector<Value> dynamicOffsets;
  llvm::SmallVector<Value> dynamicShape;
  llvm::SmallVector<Value> dynamicStrides;

  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  auto staticOffsetsAttr = builder.getDenseI64ArrayAttr(staticOffsets);
  auto staticShapeAttr = builder.getDenseI64ArrayAttr(staticShape);
  auto staticStridesAttr = builder.getDenseI64ArrayAttr(staticStrides);

  build(builder, state, tdesc, source, dynamicOffsets, dynamicShape,
        dynamicStrides, staticOffsetsAttr, staticShapeAttr, staticStridesAttr);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, TypedValue<IntegerType> source,
                           llvm::ArrayRef<OpFoldResult> offsets,
                           llvm::ArrayRef<OpFoldResult> shape,
                           llvm::ArrayRef<OpFoldResult> strides) {
  assert(shape.size() && offsets.size() && strides.size() &&
         shape.size() == strides.size() && shape.size() == offsets.size());

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<int64_t> staticShape;
  llvm::SmallVector<int64_t> staticStrides;
  llvm::SmallVector<Value> dynamicOffsets;
  llvm::SmallVector<Value> dynamicShape;
  llvm::SmallVector<Value> dynamicStrides;

  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  auto staticOffsetsAttr = builder.getDenseI64ArrayAttr(staticOffsets);
  auto staticShapeAttr = builder.getDenseI64ArrayAttr(staticShape);
  auto staticStridesAttr = builder.getDenseI64ArrayAttr(staticStrides);

  build(builder, state, tdesc, source, dynamicOffsets, dynamicShape,
        dynamicStrides, staticOffsetsAttr, staticShapeAttr, staticStridesAttr);
}

LogicalResult CreateNdDescOp::verify() {
  auto rank = (int64_t)getMixedOffsets().size();
  bool invalidRank = false;
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
  auto memrefTy = dyn_cast<MemRefType>(getSourceType());
  if (memrefTy) {
    invalidRank |= (memrefTy.getRank() != rank);
    invalidElemTy |= memrefTy.getElementType() != getElementType();
  }

  // mismatches among shape, strides, and offsets are
  // already handeled by OffsetSizeAndStrideOpInterface.
  // So they are not check here.
  if (invalidRank)
    return emitOpError(
        "Expecting the rank of shape, strides, offsets, and source (if source "
        "is a memref) should match with each other.");

  // check result TensorDesc rank
  invalidRank = (getType().getRank() > 2 || getType().getRank() > rank);

  if (invalidRank)
    return emitOpError(
        "Expecting the TensorDesc rank is up to 2 and not greater than the "
        "ranks of shape, strides, offsets or the memref source.");

  if (invalidElemTy)
    return emitOpError("TensorDesc should have the same element "
                       "type with the source if it is a memref.\n");

  if (getType().isScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_PrefetchNdOp
//===----------------------------------------------------------------------===//
LogicalResult PrefetchNdOp::verify() {
  auto tdescTy = getTensorDescType();
  if (tdescTy.isScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadNdOp
//===----------------------------------------------------------------------===//
LogicalResult LoadNdOp::verify() {
  auto tdescTy = getTensorDescType();
  auto valueTy = getType();

  if (tdescTy.getRank() > 2)
    return emitOpError("Expecting a 1D/2D TensorDesc.\n");

  if (tdescTy.isScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  if (!valueTy)
    return emitOpError("Invalid result, it should be a VectorType.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  auto array_len = tdescTy.getArrayLength();
  // adjusted tensor descriptor shape tracks the expected shape of the result.
  auto adjustedTdescShape = getShapeOf(tdescTy);
  auto valueShape = getShapeOf(valueTy);

  if (getTranspose()) {
    auto trans = getTranspose().value();

    // Make sure the transpose value is valid.
    bool valid = std::all_of(trans.begin(), trans.end(), [&](int t) {
      return t >= 0 && t < tdescTy.getRank();
    });

    if (valid)
      transpose(trans, adjustedTdescShape);
    else
      mlir::emitWarning(getLoc()) << "Invalid transpose attr. It is ignored.";
  }

  if (getPacked()) {
    if (tdescTy.getRank() == 2) {
      const int axis = 0;
      auto vnni_factor = valueShape.back();
      adjustedTdescShape[axis] /= vnni_factor;
      adjustedTdescShape.push_back(vnni_factor);
    } else {
      mlir::emitWarning(getLoc())
          << "Invalid Packed Attr. It is ignored (available for 2D "
             "TensorDesc only).";
    }
  }

  if (array_len > 1) {
    auto it = adjustedTdescShape.begin();
    adjustedTdescShape.insert(it, array_len);
  }

  return isArgShapesValid(tdescTy, valueTy, adjustedTdescShape,
                          [&]() { return emitOpError(); });
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreNdOp
//===----------------------------------------------------------------------===//
LogicalResult StoreNdOp::verify() {
  auto dstTy = getTensorDescType(); // Tile
  auto valTy = getValueType();      // Vector

  if (dstTy.getRank() > 2)
    return emitOpError("Expecting a 1D/2D TensorDesc.\n");

  if (dstTy.isScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  if (!valTy)
    return emitOpError("Expecting a VectorType result.\n");

  if (!isWriteHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isWriteHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isWriteHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  auto tdescShape = getShapeOf(dstTy);
  auto valueShape = getShapeOf(valTy);

  return isArgShapesValid(dstTy, valTy, tdescShape,
                          [&]() { return emitOpError(); });
}

//===----------------------------------------------------------------------===//
// XeGPU_UpdateNDOffsetOp
//===----------------------------------------------------------------------===//
LogicalResult UpdateNdOffsetOp::verify() {
  auto ty = getTensorDescType();
  if (ty.isScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  // number of offsets specified must match the rank of the tensor descriptor
  if (ty.getRank() != (int64_t)getNumOffsets()) {
    return emitOpError("Invalid number of offsets.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_CreateDescOp
//===----------------------------------------------------------------------===//

void CreateDescOp::build(OpBuilder &builder, OperationState &state,
                         TensorDescType TensorDesc, Value source,
                         llvm::ArrayRef<OpFoldResult> offsets) {
  auto loc = source.getLoc();
  int64_t size = static_cast<int64_t>(offsets.size());
  auto type = VectorType::get(size, builder.getIndexType());
  auto values = getValueOrCreateConstantIndexOp(builder, loc, offsets);
  auto offset = builder.create<vector::FromElementsOp>(loc, type, values);
  build(builder, state, TensorDesc, source, offset);
}

void CreateDescOp::build(OpBuilder &builder, OperationState &state,
                         TensorDescType TensorDesc, Value source,
                         llvm::ArrayRef<int64_t> offsets) {
  auto ofrs = getAsIndexOpFoldResult(builder.getContext(), offsets);
  build(builder, state, TensorDesc, source, ofrs);
}

LogicalResult CreateDescOp::verify() {
  auto tdescTy = getTensorDescType();

  if (getRankOf(getSource()) > 1)
    return emitOpError(
        "Expecting the source is a 1D memref or pointer (uint64_t).");

  if (!tdescTy.isScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  // Memory space of created TensorDesc should match with the source.
  // Both source and TensorDesc are considered for global memory by default,
  // if the memory scope attr is not specified. If source is an integer,
  // it is considered as ptr to global memory.
  auto srcMemorySpace = getSourceMemorySpace();
  auto tdescMemorySpace = static_cast<unsigned>(tdescTy.getMemorySpace());
  if (srcMemorySpace != tdescMemorySpace)
    return emitOpError("Memory space mismatch.")
           << " Source: " << srcMemorySpace
           << ", TensorDesc: " << tdescMemorySpace;

  // check total size
  auto chunkSize = tdescTy.getChunkSize();
  auto elemBits = tdescTy.getElementType().getIntOrFloatBitWidth();
  auto bitsPerLane = elemBits * chunkSize;
  if (chunkSize > 1 && bitsPerLane % 32) {
    // For 8-bit and 16-bit data, the hardware only supports chunk size of 1.
    // For 32-bit data, the hardware can support larger larger chunk size. So
    // we can bitcast 8-bit/16-bit data to 32-bit data for better performance.
    // But this requires the total size is 32 bit aligned to make the
    // optimization work.
    return emitOpError(
        "access size (chunk_size * sizeof(elemTy)) should be 32-bit aligned.");
  }

  auto lscConstraints = 512 * 8; // each access is upto 512 bytes.
  if (elemBits * tdescTy.getNumElements() > lscConstraints)
    return emitOpError("total access size (simd_lanes * chunk_size * "
                       "sizeof(elemTy)) is upto 512 bytes.");

  SmallVector<int64_t> shape({(int64_t)getNumOffsets()});
  if (chunkSize != 1)
    shape.push_back(chunkSize);

  auto tdescShape = getShapeOf(tdescTy);
  if (shape != tdescShape)
    return emitOpError("Incorrect TensorDesc shape. ")
           << "Expected is " << makeString(shape) << "\n";

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_PrefetchOp
//===----------------------------------------------------------------------===//
LogicalResult PrefetchOp::verify() {
  auto tdescTy = getTensorDescType();
  if (!tdescTy.isScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadGatherOp
//===----------------------------------------------------------------------===//
LogicalResult LoadGatherOp::verify() {
  auto tdescTy = getTensorDescType();
  auto maskTy = getMaskType();
  auto valueTy = getValueType();

  if (!valueTy)
    return emitOpError("Expecting a vector type result.\n");

  if (!tdescTy.isScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  auto tdescElemTy = tdescTy.getElementType();
  auto valueElemTy = getElementType();
  if (tdescElemTy != valueElemTy)
    return emitOpError(
        "Value should have the same element type as TensorDesc.");

  auto maskShape = getShapeOf(maskTy);
  auto valueShape = getShapeOf(valueTy);
  auto tdescShape = getShapeOf(tdescTy);

  if (tdescShape[0] != maskShape[0])
    return emitOpError("dim-0 of the Mask and TensorDesc should be the same.");

  if (tdescTy.getRank() == 2) {
    if (!getTransposeAttr())
      return emitOpError("load of rank-2 tensor has to be transposed.");
    transpose({1, 0}, tdescShape);
  }

  return isArgShapesValid(tdescTy, valueTy, tdescShape,
                          [&]() { return emitOpError(); });
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreScatterOp
//===----------------------------------------------------------------------===//
LogicalResult StoreScatterOp::verify() {
  auto tdescTy = getTensorDescType();
  if (!tdescTy.isScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  if (!isWriteHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isWriteHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isWriteHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  auto maskTy = getMaskType();
  auto valueTy = getValueType();

  if (!valueTy)
    return emitOpError("Expecting a vector type for the value.\n");

  auto maskShape = getShapeOf(maskTy);
  auto tdescShape = getShapeOf(tdescTy);
  auto valueShape = getShapeOf(valueTy);
  if (tdescShape[0] != maskShape[0])
    return emitOpError("dim-0 of the Mask and TensorDesc should be the same.");

  if (tdescTy.getRank() == 2) {
    if (!getTransposeAttr())
      return emitOpError("Store of a rank-2 tensor has to be transposed.");
    transpose({1, 0}, tdescShape);
  }

  return isArgShapesValid(tdescTy, valueTy, tdescShape,
                          [&]() { return emitOpError(); });
}

//===----------------------------------------------------------------------===//
// XeGPU_UpdateOffsetOp
//===----------------------------------------------------------------------===//
void UpdateOffsetOp::build(OpBuilder &builder, OperationState &state,
                           mlir::Value tensorDesc,
                           llvm::ArrayRef<OpFoldResult> offsets) {
  auto tdescTy = mlir::dyn_cast<TensorDescType>(tensorDesc.getType());
  assert(tdescTy && "Expecting the source is a TensorDescType value.");
  auto loc = tensorDesc.getLoc();
  int64_t size = static_cast<int64_t>(offsets.size());
  auto type = VectorType::get({size}, builder.getIndexType());
  auto values = getValueOrCreateConstantIndexOp(builder, loc, offsets);
  auto offset = builder.create<vector::FromElementsOp>(loc, type, values);
  build(builder, state, tdescTy, tensorDesc, offset);
}

void UpdateOffsetOp::build(OpBuilder &builder, OperationState &state,
                           Value tensorDesc, llvm::ArrayRef<int64_t> offsets) {
  auto ofrs = getAsIndexOpFoldResult(builder.getContext(), offsets);
  build(builder, state, tensorDesc, ofrs);
}

//===----------------------------------------------------------------------===//
// XeGPU_DpasOp
//===----------------------------------------------------------------------===//
LogicalResult DpasOp::verify() {
  int64_t lhsRank = getLhsType().getRank();
  int64_t rhsRank = getRhsType().getRank();
  int64_t resultRank = getResultType().getRank();
  auto lhsShape = getLhsType().getShape();
  auto rhsShape = getRhsType().getShape();
  auto resultShape = getResultType().getShape();

  auto sgMapA = getSgMapAAttr();
  auto sgMapB = getSgMapBAttr();
  auto sgMapC = getSgMapCAttr();

  // If sg_maps are not present, then the operation is in SIMD mode.
  if (!sgMapA && !sgMapB && !sgMapC) {
    if (lhsRank != 2 || (rhsRank != 2 && rhsRank != 3) || resultRank != 2)
      return emitOpError(
          "expecting lhs and result to be a 2D vector, and rhs to be either "
          "2D or 3D (packed) vector.");
    auto bK = rhsRank == 3 ? rhsShape[0] * rhsShape[2] : rhsShape[0];
    if (bK != lhsShape[1])
      return emitOpError("K-dimension mismatch.");
    if (lhsShape[0] != resultShape[0])
      return emitOpError("M-dimension mismatch.");
    if (rhsShape[1] != resultShape[1])
      return emitOpError("N-dimension mismatch.");
    return success();
  }
  // Otherwise, in SIMT mode we expect sg_map attributes for all operands and
  // result of DPAS operation.
  if (!sgMapA || !sgMapB || !sgMapC)
    return emitOpError("sg_map attributes for all operands and outputs are "
                       "expected in SIMT xegpu::Dpas operation");

  // In SIMT mode, All data fragments must be 2D
  if (lhsRank != 2 || rhsRank != 2 || resultRank != 2)
    return emitOpError("expecting lhs, rhs, and result to be a 2D vector.");

  auto wiLayoutA = sgMapA.getWiLayout();
  auto wiLayoutB = sgMapB.getWiLayout();
  auto wiLayoutC = sgMapC.getWiLayout();
  // Obtain the expanded shapes of the operands and result using wi_layout.
  // NOTE: For B, get rid of the packed dimension for the expanded shape.
  SmallVector<int64_t> expandedShapeA = {lhsShape[0] * wiLayoutA[0],
                                         lhsShape[1] * wiLayoutA[1]};
  SmallVector<int64_t> expandedShapeB = {
      rhsShape[0] * rhsShape[1] * wiLayoutB[0], 1 * wiLayoutB[1]};
  SmallVector<int64_t> expandedShapeC = {resultShape[0] * wiLayoutC[0],
                                         resultShape[1] * wiLayoutC[1]};
  auto bK = expandedShapeB[0];
  if (bK != expandedShapeA[1])
    return emitOpError("K-dimension mismatch.");
  if (expandedShapeA[0] != expandedShapeC[0])
    return emitOpError("M-dimension mismatch.");
  if (expandedShapeB[1] != expandedShapeC[1])
    return emitOpError("N-dimension mismatch.");

  return success();
}
} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUEnums.cpp.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPU.cpp.inc>
