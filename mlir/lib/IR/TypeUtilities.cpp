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

using namespace mlir;

Type mlir::getElementTypeOrSelf(Type type) {
  if (auto st = type.dyn_cast<ShapedType>())
    return st.getElementType();
  return type;
}

Type mlir::getElementTypeOrSelf(Value val) {
  return getElementTypeOrSelf(val.getType());
}

Type mlir::getElementTypeOrSelf(Attribute attr) {
  return getElementTypeOrSelf(attr.getType());
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
  if (auto opaque = type.dyn_cast<mlir::OpaqueType>())
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
  auto sType1 = type1.dyn_cast<ShapedType>();
  auto sType2 = type2.dyn_cast<ShapedType>();

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

OperandElementTypeIterator::OperandElementTypeIterator(
    Operation::operand_iterator it)
    : llvm::mapped_iterator<Operation::operand_iterator, Type (*)(Value)>(
          it, &unwrap) {}

Type OperandElementTypeIterator::unwrap(Value value) {
  return value.getType().cast<ShapedType>().getElementType();
}

ResultElementTypeIterator::ResultElementTypeIterator(
    Operation::result_iterator it)
    : llvm::mapped_iterator<Operation::result_iterator, Type (*)(Value)>(
          it, &unwrap) {}

Type ResultElementTypeIterator::unwrap(Value value) {
  return value.getType().cast<ShapedType>().getElementType();
}
