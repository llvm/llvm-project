//===- Utils.cpp - Utils for APFloat Conversion ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

mlir::Value mlir::getAPFloatSemanticsValue(OpBuilder &b, Location loc,
                                           FloatType floatTy) {
  int32_t sem = llvm::APFloatBase::SemanticsToEnum(floatTy.getFloatSemantics());
  return arith::ConstantOp::create(b, loc, b.getI32Type(),
                                   b.getIntegerAttr(b.getI32Type(), sem));
}
