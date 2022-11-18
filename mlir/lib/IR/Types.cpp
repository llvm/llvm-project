//===- Types.cpp - MLIR Type Classes --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

MLIRContext *Type::getContext() const { return getDialect().getContext(); }

bool Type::isFloat8E5M2() const { return isa<Float8E5M2Type>(); }
bool Type::isFloat8E4M3FN() const { return isa<Float8E4M3FNType>(); }
bool Type::isBF16() const { return isa<BFloat16Type>(); }
bool Type::isF16() const { return isa<Float16Type>(); }
bool Type::isF32() const { return isa<Float32Type>(); }
bool Type::isF64() const { return isa<Float64Type>(); }
bool Type::isF80() const { return isa<Float80Type>(); }
bool Type::isF128() const { return isa<Float128Type>(); }

bool Type::isIndex() const { return isa<IndexType>(); }

/// Return true if this is an integer type with the specified width.
bool Type::isInteger(unsigned width) const {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.getWidth() == width;
  return false;
}

bool Type::isSignlessInteger() const {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isSignless();
  return false;
}

bool Type::isSignlessInteger(unsigned width) const {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isSignless() && intTy.getWidth() == width;
  return false;
}

bool Type::isSignedInteger() const {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isSigned();
  return false;
}

bool Type::isSignedInteger(unsigned width) const {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isSigned() && intTy.getWidth() == width;
  return false;
}

bool Type::isUnsignedInteger() const {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isUnsigned();
  return false;
}

bool Type::isUnsignedInteger(unsigned width) const {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isUnsigned() && intTy.getWidth() == width;
  return false;
}

bool Type::isSignlessIntOrIndex() const {
  return isSignlessInteger() || isa<IndexType>();
}

bool Type::isSignlessIntOrIndexOrFloat() const {
  return isSignlessInteger() || isa<IndexType, FloatType>();
}

bool Type::isSignlessIntOrFloat() const {
  return isSignlessInteger() || isa<FloatType>();
}

bool Type::isIntOrIndex() const { return isa<IntegerType>() || isIndex(); }

bool Type::isIntOrFloat() const { return isa<IntegerType, FloatType>(); }

bool Type::isIntOrIndexOrFloat() const { return isIntOrFloat() || isIndex(); }

unsigned Type::getIntOrFloatBitWidth() const {
  assert(isIntOrFloat() && "only integers and floats have a bitwidth");
  if (auto intType = dyn_cast<IntegerType>())
    return intType.getWidth();
  return cast<FloatType>().getWidth();
}
