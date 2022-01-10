//===- BuiltinTypeInterfaces.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
/// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// ShapedType
//===----------------------------------------------------------------------===//

constexpr int64_t ShapedType::kDynamicSize;
constexpr int64_t ShapedType::kDynamicStrideOrOffset;

int64_t ShapedType::getNumElements(ArrayRef<int64_t> shape) {
  int64_t num = 1;
  for (int64_t dim : shape) {
    num *= dim;
    assert(num >= 0 && "integer overflow in element count computation");
  }
  return num;
}

int64_t ShapedType::getSizeInBits() const {
  assert(hasStaticShape() &&
         "cannot get the bit size of an aggregate with a dynamic shape");

  auto elementType = getElementType();
  if (elementType.isIntOrFloat())
    return elementType.getIntOrFloatBitWidth() * getNumElements();

  if (auto complexType = elementType.dyn_cast<ComplexType>()) {
    elementType = complexType.getElementType();
    return elementType.getIntOrFloatBitWidth() * getNumElements() * 2;
  }
  return getNumElements() * elementType.cast<ShapedType>().getSizeInBits();
}
