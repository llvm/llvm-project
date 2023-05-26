//===- TypeUtilities.cpp - Helper function for type queries ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines generic type utilities.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <numeric>

using namespace mlir;

Type mlir::getElementTypeOrSelf(Type type) {
  if (auto st = llvm::dyn_cast<ShapedType>(type))
    return st.getElementType();
  return type;
}

Type mlir::getElementTypeOrSelf(Value val) {
  return getElementTypeOrSelf(val.getType());
}

Type mlir::getElementTypeOrSelf(Attribute attr) {
  if (auto typedAttr = llvm::dyn_cast<TypedAttr>(attr))
    return getElementTypeOrSelf(typedAttr.getType());
  return {};
}

SmallVector<Type, 10> mlir::getFlattenedTypes(TupleType t) {
  SmallVector<Type, 10> fTypes;
  t.getFlattenedTypes(fTypes);
  return fTypes;
}

/// Return true if the specified type is an opaque type with the specified
/// dialect and typeData.
bool mlir::isOpaqueTypeWithName(Type type, StringRef dialect,
                                StringRef typeData) {
  if (auto opaque = llvm::dyn_cast<mlir::OpaqueType>(type))
    return opaque.getDialectNamespace() == dialect &&
           opaque.getTypeData() == typeData;
  return false;
}

/// Returns success if the given two shapes are compatible. That is, they have
/// the same size and each pair of the elements are equal or one of them is
/// dynamic.
LogicalResult mlir::verifyCompatibleShape(ArrayRef<int64_t> shape1,
                                          ArrayRef<int64_t> shape2) {
  if (shape1.size() != shape2.size())
    return failure();
  for (auto dims : llvm::zip(shape1, shape2)) {
    int64_t dim1 = std::get<0>(dims);
    int64_t dim2 = std::get<1>(dims);
    if (!ShapedType::isDynamic(dim1) && !ShapedType::isDynamic(dim2) &&
        dim1 != dim2)
      return failure();
  }
  return success();
}

/// Returns success if the given two types have compatible shape. That is,
/// they are both scalars (not shaped), or they are both shaped types and at
/// least one is unranked or they have compatible dimensions. Dimensions are
/// compatible if at least one is dynamic or both are equal. The element type
/// does not matter.
LogicalResult mlir::verifyCompatibleShape(Type type1, Type type2) {
  auto sType1 = llvm::dyn_cast<ShapedType>(type1);
  auto sType2 = llvm::dyn_cast<ShapedType>(type2);

  // Either both or neither type should be shaped.
  if (!sType1)
    return success(!sType2);
  if (!sType2)
    return failure();

  if (!sType1.hasRank() || !sType2.hasRank())
    return success();

  return verifyCompatibleShape(sType1.getShape(), sType2.getShape());
}

/// Returns success if the given two arrays have the same number of elements and
/// each pair wise entries have compatible shape.
LogicalResult mlir::verifyCompatibleShapes(TypeRange types1, TypeRange types2) {
  if (types1.size() != types2.size())
    return failure();
  for (auto it : llvm::zip_first(types1, types2))
    if (failed(verifyCompatibleShape(std::get<0>(it), std::get<1>(it))))
      return failure();
  return success();
}

LogicalResult mlir::verifyCompatibleDims(ArrayRef<int64_t> dims) {
  if (dims.empty())
    return success();
  auto staticDim = std::accumulate(
      dims.begin(), dims.end(), dims.front(), [](auto fold, auto dim) {
        return ShapedType::isDynamic(dim) ? fold : dim;
      });
  return success(llvm::all_of(dims, [&](auto dim) {
    return ShapedType::isDynamic(dim) || dim == staticDim;
  }));
}

/// Returns success if all given types have compatible shapes. That is, they are
/// all scalars (not shaped), or they are all shaped types and any ranked shapes
/// have compatible dimensions. Dimensions are compatible if all non-dynamic
/// dims are equal. The element type does not matter.
LogicalResult mlir::verifyCompatibleShapes(TypeRange types) {
  auto shapedTypes = llvm::map_to_vector<8>(
      types, [](auto type) { return llvm::dyn_cast<ShapedType>(type); });
  // Return failure if some, but not all are not shaped. Return early if none
  // are shaped also.
  if (llvm::none_of(shapedTypes, [](auto t) { return t; }))
    return success();
  if (!llvm::all_of(shapedTypes, [](auto t) { return t; }))
    return failure();

  // Return failure if some, but not all, are scalable vectors.
  bool hasScalableVecTypes = false;
  bool hasNonScalableVecTypes = false;
  for (Type t : types) {
    auto vType = llvm::dyn_cast<VectorType>(t);
    if (vType && vType.isScalable())
      hasScalableVecTypes = true;
    else
      hasNonScalableVecTypes = true;
    if (hasScalableVecTypes && hasNonScalableVecTypes)
      return failure();
  }

  // Remove all unranked shapes
  auto shapes = llvm::to_vector<8>(llvm::make_filter_range(
      shapedTypes, [](auto shapedType) { return shapedType.hasRank(); }));
  if (shapes.empty())
    return success();

  // All ranks should be equal
  auto firstRank = shapes.front().getRank();
  if (llvm::any_of(shapes,
                   [&](auto shape) { return firstRank != shape.getRank(); }))
    return failure();

  for (unsigned i = 0; i < firstRank; ++i) {
    // Retrieve all ranked dimensions
    auto dims = llvm::map_to_vector<8>(
        llvm::make_filter_range(
            shapes, [&](auto shape) { return shape.getRank() >= i; }),
        [&](auto shape) { return shape.getDimSize(i); });
    if (verifyCompatibleDims(dims).failed())
      return failure();
  }

  return success();
}

Type OperandElementTypeIterator::mapElement(Value value) const {
  return llvm::cast<ShapedType>(value.getType()).getElementType();
}

Type ResultElementTypeIterator::mapElement(Value value) const {
  return llvm::cast<ShapedType>(value.getType()).getElementType();
}
