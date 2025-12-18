//===- Utils.h - Utils for APFloat Conversion - C++ -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHANDMATHTOAPFLOAT_UTILS_H_
#define MLIR_CONVERSION_ARITHANDMATHTOAPFLOAT_UTILS_H_

namespace mlir {
class Value;
class OpBuilder;
class Location;
class FloatType;

Value getAPFloatSemanticsValue(OpBuilder &b, Location loc, FloatType floatTy);
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHANDMATHTOAPFLOAT_UTILS_H_
