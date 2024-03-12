//===- XeGPUOps.cpp - MLIR XeGPU ops implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>
#include <mlir/IR/Builders.h>

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "xegpu"

namespace mlir {
namespace xegpu {

static size_t getRankOf(Value value) {
  if (value.getType().isIntOrIndexOrFloat())
    return 0;
  if (auto ty = llvm::dyn_cast_if_present<MemRefType>(value.getType()))
    return ty.getRank();
  if (auto ty = llvm::dyn_cast_if_present<VectorType>(value.getType()))
    return ty.getRank();
  llvm_unreachable("Unsupported value for getRankOf");
}

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
// void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
//                            Type TensorDesc, Value source, ValueRange offsets,
//                            ValueRange shape, ValueRange strides,
//                            llvm::ArrayRef<int64_t> static_offsets) {
//   auto offsetRank = static_offsets.size();
//   auto shapeRank = shape.size() ? shape.size() : getRankOf(source);

//   size_t dynOffsetRank =
//       std::count_if(static_offsets.begin(), static_offsets.end(),
//                     [](int64_t d) { return ShapedType::isDynamic(d); });

//   // shape and strides should exists at the same time
//   // and the final rank for shape and offset (dynamic + static)
//   // should be the same
//   assert(shape.size() == strides.size() && shapeRank == offsetRank &&
//          offsets.size() == dynOffsetRank);

//   state.addOperands(source);
//   state.addOperands(offsets);
//   state.addOperands(shape);
//   state.addOperands(strides);
//   state.addAttribute(
//       getOperandSegmentSizesAttrName(state.name),
//       builder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()),
//                                     static_cast<int32_t>(shape.size()),
//                                     static_cast<int32_t>(strides.size())}));
//   state.addAttribute(getStaticOffsetsAttrName(state.name),
//                      builder.getDenseI64ArrayAttr(static_offsets));
//   state.addTypes(TensorDesc);
// }

// void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
//                            Type tdesc, Value source,
//                            llvm::ArrayRef<OpFoldResult> offsets) {
//   auto ty = llvm::dyn_cast_if_present<MemRefType>(source.getType());
//   assert(ty && ty.hasStaticShape() && offsets.size() == getRankOf(source));

//   llvm::SmallVector<int64_t> staticOffsets;
//   llvm::SmallVector<Value> dynamicOffsets;
//   dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

//   build(builder, state, tdesc, source, dynamicOffsets /* dynamic offsets */,
//         ValueRange({}) /* empty dynamic shape */,
//         ValueRange({}) /* empty dynamic strides */,
//         staticOffsets /* static offsets */);
// }

// void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
//                            Type tdesc, Value source,
//                            llvm::ArrayRef<OpFoldResult> offsets,
//                            ValueRange shape, ValueRange stride) {
//   assert(shape.size() && offsets.size() && stride.size() &&
//          shape.size() == stride.size() && shape.size() == offsets.size());

//   llvm::SmallVector<int64_t> staticOffsets;
//   llvm::SmallVector<Value> dynamicOffsets;

//   dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

//   build(builder, state, tdesc, source, /* dynamic_offsets = */ dynamicOffsets,
//         /* dynamic shape = */ shape , /* dynamic strides = */ stride,
//         /* static offsets = */ staticOffsets);
// }


LogicalResult CreateNdDescOp::verify() {
  // auto offsetRank = getEffectiveOffsets().size();
  // auto shapeRank = getEffectiveShape().size();
  // auto stridesRank = getEffectiveStrides().size();
  // auto baseRank = getRankOf(getSource()) ? getRankOf(getSource()) : 2;

  llvm::dbgs() << "\nNum of mixed Offsets: " << getMixedOffsets().size()
               << "\nNum of mixed Sizes: " << getMixedSizes().size()
               << "\nNum of mixed Strides: " << getMixedStrides().size()
               << "\n";

  // if (offsetRank != shapeRank || shapeRank != stridesRank ||
  //     shapeRank != baseRank)

  //   return emitOpError(
  //       "Expecting the rank of shape, strides, offsets and memref type "
  //       "should match with each other (they currently should be 2D).");
  return success();
}

// compute consolidated offsets from dynamic_offsets and static_offsets
// parameters
llvm::SmallVector<OpFoldResult> CreateNdDescOp::getEffectiveOffsets() {
  llvm::SmallVector<OpFoldResult> offsets;
  auto dynamicOffsets = getOffsets(); // offsets variable
  auto staticOffsets = getStaticOffsets();   // static_offsets attribute

  // in case static_offsets is missing, dynamic_offsets will be used
  if (staticOffsets.size() == 0) {
    offsets.assign(dynamicOffsets.begin(), dynamicOffsets.end());
    return offsets;
  }

  // use static offsets for each dim if it has valid value, 
  // othwise use the value from dynamic_offsets
  for (size_t i = 0, j = 0; i < staticOffsets.size(); i++) {
    if (ShapedType::isDynamic(staticOffsets[i])) {
      assert(j < dynamicOffsets.size());
      offsets.push_back(dynamicOffsets[j++]);
    } else {
      auto ty = IndexType::get(getContext());
      auto attr = IntegerAttr::get(ty, staticOffsets[i]);
      offsets.push_back(attr);
    }
  }
  return offsets;
}

// get the consolidated shape of the 2D memory region. 
// It prefer dynamic_shape than the static shape of 
// memref type.
llvm::SmallVector<OpFoldResult> CreateNdDescOp::getEffectiveShape() {
  llvm::SmallVector<OpFoldResult> shape;
  auto dynShape = getShape();
  if (dynShape.size()) {
    shape.append(dynShape.begin(), dynShape.end());
    return shape;
  }

  auto ty = llvm::dyn_cast_if_present<MemRefType>(getSourceType());
  if (ty && ty.hasStaticShape()) {
    for (auto dim : ty.getShape()) {
      auto attr = IntegerAttr::get(IndexType::get(getContext()), dim);
      shape.push_back(attr);
    }
    return shape;
  }
  
  this->emitError("The shape information of the memory is missing.\n");
  return {};
}

// get the consolidated strides of the 2D memory region. 
// It prefer dynamic_stride than the static strides of 
// memref type.
llvm::SmallVector<OpFoldResult> CreateNdDescOp::getEffectiveStrides() {
  llvm::SmallVector<OpFoldResult> strides;

  auto dynStrides = getStrides();
  if (dynStrides.size()) {
    strides.append(dynStrides.begin(), dynStrides.end());
    return strides;
  }

  auto ty = llvm::dyn_cast_if_present<MemRefType>(getSourceType());
  if (ty && ty.hasStaticShape()) {
    auto [staticStrides, offset] = getStridesAndOffset(ty);
    for (auto dim : staticStrides) {
      auto attr = IntegerAttr::get(IndexType::get(getContext()), dim);
      strides.push_back(attr);
    }
    return strides;
  }

  this->emitError("The strides information of the memory is missing.\n");
  return {};
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadNDOp
//===----------------------------------------------------------------------===//
LogicalResult LoadNDOp::verify() {
  auto tdescTy = getTensorDescType();
  auto valueTy = getType();

  if (tdescTy.getRank() != 2)
    return emitOpError(
        "The TensorDesc for LoadNDOp should be a 2D TensorDesc.");

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
    return emitOpError() <<"Result shape doesn't match TensorDesc shape."
           << "The expected shape is " << makeString(tdescShape) << ". "
           << "But the given shape is " << makeString(valueShape) << ".\n";
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreNDOp
//===----------------------------------------------------------------------===//
LogicalResult StoreNDOp::verify() {
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
    return emitOpError() << "The result shape should match the TensorDesc shape.\n";
  return success();
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUEnums.cpp.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPU.cpp.inc>
