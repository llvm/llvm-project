//===- QuantOps.cpp - Quantization Type and Ops Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QuantDialectBytecode.h"
#include "TypeDetail.h"

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

#include "mlir/Dialect/Quant/IR/QuantOpsDialect.cpp.inc"


namespace mlir {
namespace quant {

namespace {

Type getPrimitiveType(Type ty) {
  if (auto tensorType = dyn_cast<TensorType>(ty))
    return tensorType.getElementType();
  return ty;
}

} // namespace


//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void QuantDialect::initialize() {
  addTypes<AnyQuantizedType, CalibratedQuantizedType, UniformQuantizedType,
           UniformQuantizedPerAxisType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quant/IR/QuantOps.cpp.inc"
      >();
  detail::addBytecodeInterface(this);
}


//===----------------------------------------------------------------------===//
// QuantizeCastOp
//===----------------------------------------------------------------------===//

FloatType QuantizeCastOp::getFloatType() {
  return cast<FloatType>(getPrimitiveType(getInput().getType()));
}

UniformQuantizedType QuantizeCastOp::getQuantizedType() {
  return cast<UniformQuantizedType>(getPrimitiveType(getResult().getType()));
}


//===----------------------------------------------------------------------===//
// StorageCastOp
//===----------------------------------------------------------------------===//

OpFoldResult StorageCastOp::fold(FoldAdaptor adaptor) {
  // Matches x -> [scast -> scast] -> y, replacing the second scast with the
  // value of x if the casts invert each other.
  auto srcScastOp = getInput().getDefiningOp<StorageCastOp>();
  if (!srcScastOp || srcScastOp.getInput().getType() != getType())
    return OpFoldResult();
  return srcScastOp.getInput();
}

} // namespace quant
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Quant/IR/QuantOps.cpp.inc"

