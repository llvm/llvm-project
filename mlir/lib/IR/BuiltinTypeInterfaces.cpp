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

constexpr int64_t ShapedType::kDynamic;

int64_t ShapedType::getNumElements(ArrayRef<int64_t> shape) {
  int64_t num = 1;
  for (int64_t dim : shape) {
    num *= dim;
    assert(num >= 0 && "integer overflow in element count computation");
  }
  return num;
}

bool ShapedType::isShapeRefinementOf(ArrayRef<int64_t> source,
                                     ArrayRef<int64_t> target) {
  if (source.size() != target.size())
    return false;
  for (auto [srcDim, tgtDim] : llvm::zip_equal(source, target)) {
    // If the source dimension is dynamic, then the target dimension can be
    // dynamic or static.
    if (isDynamic(srcDim))
      continue;
    // Static source dim and dynamic result dim -> not a refinement.
    if (isDynamic(tgtDim))
      return false;
    // Static source dim != static result dim -> not a refinement.
    if (srcDim != tgtDim)
      return false;
  }
  return true;
}

bool ShapedType::isShapeGeneralizationOf(ArrayRef<int64_t> source,
                                         ArrayRef<int64_t> target) {
  return isShapeRefinementOf(target, source);
}
