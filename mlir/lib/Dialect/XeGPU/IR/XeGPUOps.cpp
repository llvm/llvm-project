//===- XeGPUOps.cpp - MLIR XeGPU ops implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  os.flush();
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
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticOffsets);

  auto staticOffsetsAttr = builder.getDenseI64ArrayAttr(staticOffsets);
  auto staticShapeAttr = builder.getDenseI64ArrayAttr(staticShape);
  auto staticStridesAttr = builder.getDenseI64ArrayAttr(staticStrides);

  build(builder, state, tdesc, source, dynamicOffsets, dynamicShape,
        dynamicStrides, staticOffsetsAttr, staticShapeAttr, staticStridesAttr);
}

LogicalResult CreateNdDescOp::verify() {
  auto rank = (int64_t)getMixedOffsets().size();
  bool invalidRank = (rank != 2);
  bool invalidElemTy = false;

  // check source type matches the rank if it is a memref.
  // It also should have the same ElementType as TensorDesc.
  auto memrefTy = dyn_cast<MemRefType>(getSourceType());
  if (memrefTy) {
    invalidRank |= (memrefTy.getRank() != rank);
    invalidElemTy |= memrefTy.getElementType() != getElementType();
  }

  // check result type matches the rank
  invalidRank = (getType().getRank() != rank);

  // mismatches among shape, strides, and offsets are
  // already handeled by OffsetSizeAndStrideOpInterface.
  // So they are not check here.
  if (invalidRank)
    return emitOpError(
        "Expecting the rank of shape, strides, offsets, "
        "source memref type (if source is a memref) and TensorDesc "
        "should match with each other. They currenlty are 2D.");

  if (invalidElemTy)
    return emitOpError("TensorDesc should have the same element "
                       "type with the source if it is a memref.\n");

  if (getType().getScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_PrefetchNdOp
//===----------------------------------------------------------------------===//
LogicalResult PrefetchNdOp::verify() {
  auto tdescTy = getTensorDescType();
  if (tdescTy.getScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invlid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invlid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invlid l3_hint: ") << getL3HintAttr();

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadNdOp
//===----------------------------------------------------------------------===//
LogicalResult LoadNdOp::verify() {
  auto tdescTy = getTensorDescType();
  auto valueTy = getType();

  if (tdescTy.getRank() != 2)
    return emitOpError("Expecting a 2D TensorDesc.\n");

  if (tdescTy.getScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  if (!valueTy)
    return emitOpError("Invalid result, it should be a VectorType.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invlid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invlid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invlid l3_hint: ") << getL3HintAttr();

  auto array_len = tdescTy.getArrayLength();
  auto tdescShape = getShapeOf(tdescTy);
  auto valueShape = getShapeOf(valueTy);

  if (getTranspose()) {
    auto trans = getTranspose().value();
    if (tdescShape.size() >= trans.size())
      transpose(trans, tdescShape);
    else
      emitWarning("Invalid transpose attr. It is ignored.");
  }

  if (getVnniAxis()) {
    auto axis = getVnniAxis().value();
    auto vnni_factor = valueShape.back();
    tdescShape[axis] /= vnni_factor;
    tdescShape.push_back(vnni_factor);
  }

  if (array_len > 1) {
    auto it = tdescShape.begin();
    tdescShape.insert(it, array_len);
  }

  if (tdescShape != valueShape)
    return emitOpError() << "Result shape doesn't match TensorDesc shape."
                         << "The expected shape is " << makeString(tdescShape)
                         << ". But the given shape is "
                         << makeString(valueShape) << ".\n";
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreNdOp
//===----------------------------------------------------------------------===//
LogicalResult StoreNdOp::verify() {
  auto dstTy = getTensorDescType(); // Tile
  auto valTy = getValueType();      // Vector

  if (dstTy.getRank() != 2)
    return emitOpError("Expecting a 2D TensorDesc.\n");

  if (dstTy.getScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  if (!valTy)
    return emitOpError("Exepcting a VectorType result.\n");

  if (!isWriteHintOrNone(getL1HintAttr()))
    return emitOpError("invlid l1_hint: ") << getL1HintAttr();

  if (!isWriteHintOrNone(getL2HintAttr()))
    return emitOpError("invlid l2_hint: ") << getL2HintAttr();

  if (!isWriteHintOrNone(getL3HintAttr()))
    return emitOpError("invlid l3_hint: ") << getL3HintAttr();

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_UpdateNDOffsetOp
//===----------------------------------------------------------------------===//
LogicalResult UpdateNdOffsetOp::verify() {
  auto ty = getTensorDescType();
  if (ty.getScattered())
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
                         llvm::ArrayRef<OpFoldResult> offsets,
                         uint32_t chunk_size) {
  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  build(builder, state, TensorDesc, source, dynamicOffsets, staticOffsets,
        chunk_size);
}

LogicalResult CreateDescOp::verify() {
  auto tdescTy = getTensorDescType();
  auto chunkSize = getChunkSize();

  if (getRankOf(getSource()) > 1)
    return emitOpError(
        "Expecting the source is a 1D memref or pointer (uint64_t).");

  if (!tdescTy.getScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

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
  if (!tdescTy.getScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invlid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invlid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invlid l3_hint: ") << getL3HintAttr();

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadGatherOp
//===----------------------------------------------------------------------===//
LogicalResult LoadGatherOp::verify() {
  auto tdescTy = getTensorDescType();
  auto maskTy = getMaskType();
  auto valueTy = getValueType();

  if (!tdescTy.getScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invlid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invlid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invlid l3_hint: ") << getL3HintAttr();

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

  if (getTransposeAttr()) {
    auto trans = getTranspose().value();
    if (tdescShape.size() < trans.size())
      emitWarning("Invalid transpose attr. It is ignored.");
    else
      transpose(trans, tdescShape);
  }

  if (valueShape != tdescShape)
    return emitOpError("Unexpected result shape")
           << "(Expected shape: " << makeString(tdescShape)
           << ", Given shape: " << makeString(valueShape) << ").\n";

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreScatterOp
//===----------------------------------------------------------------------===//
LogicalResult StoreScatterOp::verify() {
  auto tdescTy = getTensorDescType();
  if (!tdescTy.getScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  if (!isWriteHintOrNone(getL1HintAttr()))
    return emitOpError("invlid l1_hint: ") << getL1HintAttr();

  if (!isWriteHintOrNone(getL2HintAttr()))
    return emitOpError("invlid l2_hint: ") << getL2HintAttr();

  if (!isWriteHintOrNone(getL3HintAttr()))
    return emitOpError("invlid l3_hint: ") << getL3HintAttr();

  auto maskTy = getMaskType();
  auto maskShape = getShapeOf(maskTy);
  auto tdescShape = getShapeOf(tdescTy);
  if (tdescShape[0] != maskShape[0])
    return emitOpError("dim-0 of the Mask and TensorDesc should be the same.");

  return success();
}
//===----------------------------------------------------------------------===//
// XeGPU_DpasOp
//===----------------------------------------------------------------------===//
LogicalResult DpasOp::verify() {
  int64_t lhsRank = getLhsType().getRank();
  int64_t rhsRank = getRhsType().getRank();

  if (lhsRank != rhsRank || lhsRank != 3)
    return emitOpError(
        "lhs and rhs rank does not match for dpas op, or their rank is not 3.");

  if (getAcc() && getAccType() != getResultType())
    return emitOpError("Accumulator and Result for dpas op should have the "
                       "same type (both shape and element type).");

  auto lhsShape = getLhsType().getShape();
  auto rhsShape = getRhsType().getShape();
  if (lhsShape[1] != rhsShape[0] || lhsShape[2] != rhsShape[2])
    return emitOpError("K-dimension or vnni-factor mismatch.");

  return success();
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUEnums.cpp.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPU.cpp.inc>
