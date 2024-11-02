//===- BuiltinAttributeInterfaces.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
/// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// ElementsAttr
//===----------------------------------------------------------------------===//

Type ElementsAttr::getElementType(ElementsAttr elementsAttr) {
  return elementsAttr.getType().getElementType();
}

int64_t ElementsAttr::getNumElements(ElementsAttr elementsAttr) {
  return elementsAttr.getType().getNumElements();
}

bool ElementsAttr::isValidIndex(ShapedType type, ArrayRef<uint64_t> index) {
  // Verify that the rank of the indices matches the held type.
  int64_t rank = type.getRank();
  if (rank == 0 && index.size() == 1 && index[0] == 0)
    return true;
  if (rank != static_cast<int64_t>(index.size()))
    return false;

  // Verify that all of the indices are within the shape dimensions.
  ArrayRef<int64_t> shape = type.getShape();
  return llvm::all_of(llvm::seq<int>(0, rank), [&](int i) {
    int64_t dim = static_cast<int64_t>(index[i]);
    return 0 <= dim && dim < shape[i];
  });
}
bool ElementsAttr::isValidIndex(ElementsAttr elementsAttr,
                                ArrayRef<uint64_t> index) {
  return isValidIndex(elementsAttr.getType(), index);
}

uint64_t ElementsAttr::getFlattenedIndex(Type type, ArrayRef<uint64_t> index) {
  ShapedType shapeType = type.cast<ShapedType>();
  assert(isValidIndex(shapeType, index) &&
         "expected valid multi-dimensional index");

  // Reduce the provided multidimensional index into a flattended 1D row-major
  // index.
  auto rank = shapeType.getRank();
  ArrayRef<int64_t> shape = shapeType.getShape();
  uint64_t valueIndex = 0;
  uint64_t dimMultiplier = 1;
  for (int i = rank - 1; i >= 0; --i) {
    valueIndex += index[i] * dimMultiplier;
    dimMultiplier *= shape[i];
  }
  return valueIndex;
}

//===----------------------------------------------------------------------===//
// MemRefLayoutAttrInterface
//===----------------------------------------------------------------------===//

LogicalResult mlir::detail::verifyAffineMapAsLayout(
    AffineMap m, ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emitError) {
  if (m.getNumDims() != shape.size())
    return emitError() << "memref layout mismatch between rank and affine map: "
                       << shape.size() << " != " << m.getNumDims();

  return success();
}
