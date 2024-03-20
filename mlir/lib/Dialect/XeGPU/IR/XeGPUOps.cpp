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

#define DEBUG_TYPE "xegpu"

namespace mlir {
namespace xegpu {

static void transpose(llvm::ArrayRef<int64_t> trans,
                      std::vector<int64_t> &shape) {
  std::vector<int64_t> old = shape;
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
  auto memrefTy = getSourceType().dyn_cast<MemRefType>();
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

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadNdOp
//===----------------------------------------------------------------------===//
LogicalResult LoadNdOp::verify() {
  auto tdescTy = getTensorDescType();
  auto valueTy = getType();

  if (tdescTy.getRank() != 2)
    return emitOpError(
        "The TensorDesc for LoadNdOp should be a 2D TensorDesc.");

  if (!valueTy)
    return emitOpError("Invalid result, it should be a VectorType.\n");

  auto tdescElemTy = tdescTy.getElementType();
  auto valueElemTy = valueTy.getElementType();

  if (tdescElemTy != valueElemTy)
    return emitOpError(
        "Value should have the same element type as TensorDesc.");

  auto array_len = tdescTy.getArrayLength();
  auto tdescShape = tdescTy.getShape().vec();
  auto valueShape = valueTy.getShape().vec();

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
  auto dstTy = getTensorDesc().getType();               // Tile
  auto valTy = getValue().getType().cast<VectorType>(); // Vector

  if (dstTy.getRank() != 2)
    return emitOpError("Expecting a 2D TensorDesc shape.\n");

  if (!valTy)
    return emitOpError("Exepcting a VectorType result.\n");

  auto dstElemTy = dstTy.getElementType();
  auto valElemTy = valTy.getElementType();

  if (dstElemTy != valElemTy) {
    return emitOpError() << "The element type of the value should "
                            "match the elementtype of the TensorDesc.\n";
  }

  if (dstTy.getShape() != valTy.getShape())
    return emitOpError()
           << "The result shape should match the TensorDesc shape.\n";
  return success();
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUEnums.cpp.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPU.cpp.inc>
