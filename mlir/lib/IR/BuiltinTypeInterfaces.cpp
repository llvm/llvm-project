//===- BuiltinTypeInterfaces.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/CheckedArithmetic.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
/// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.cpp.inc"

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
