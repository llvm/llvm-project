//===- BuiltinTypeInterfaces.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"
#include <climits>

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
/// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// DenseElementTypeInterface implementations for float types
//===----------------------------------------------------------------------===//

size_t mlir::detail::getFloatTypeDenseElementBitSize(Type type) {
  return cast<FloatType>(type).getWidth();
}

Attribute mlir::detail::convertFloatTypeToAttribute(Type type,
                                                    ArrayRef<char> rawData) {
  auto floatType = cast<FloatType>(type);
  APInt intVal = readBits(rawData.data(), /*bitPos=*/0, floatType.getWidth());
  APFloat floatVal(floatType.getFloatSemantics(), intVal);
  return FloatAttr::get(type, floatVal);
}

LogicalResult
mlir::detail::convertFloatTypeFromAttribute(Type type, Attribute attr,
                                            SmallVectorImpl<char> &result) {
  auto floatType = cast<FloatType>(type);
  auto floatAttr = dyn_cast<FloatAttr>(attr);
  if (!floatAttr || floatAttr.getType() != type)
    return failure();
  size_t byteSize =
      llvm::divideCeil(floatType.getWidth(), static_cast<unsigned>(CHAR_BIT));
  size_t bitPos = result.size() * CHAR_BIT;
  result.resize(result.size() + byteSize);
  writeBits(result.data(), bitPos, floatAttr.getValue().bitcastToAPInt());
  return success();
}

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

unsigned FloatType::getWidth() {
  return APFloat::semanticsSizeInBits(getFloatSemantics());
}

unsigned FloatType::getFPMantissaWidth() {
  return APFloat::semanticsPrecision(getFloatSemantics());
}

//===----------------------------------------------------------------------===//
// ShapedType
//===----------------------------------------------------------------------===//

std::optional<int64_t> ShapedType::tryGetNumElements(ArrayRef<int64_t> shape) {
  int64_t num = 1;
  for (int64_t dim : shape) {
    auto result = llvm::checkedMul(num, dim);
    if (!result)
      return std::nullopt;
    num = *result;
  }
  return num;
}

int64_t ShapedType::getNumElements(ArrayRef<int64_t> shape) {
#ifndef NDEBUG
  std::optional<int64_t> num = tryGetNumElements(shape);
  assert(num.has_value() && "integer overflow in element count computation");
  return *num;
#else
  int64_t num = 1;
  for (int64_t dim : shape)
    num *= dim;
  return num;
#endif
}
