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
// AbstractType
//===----------------------------------------------------------------------===//

void AbstractType::walkImmediateSubElements(
    Type type, function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkImmediateSubElementsFn(type, walkAttrsFn, walkTypesFn);
}

Type AbstractType::replaceImmediateSubElements(Type type,
                                               ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  return replaceImmediateSubElementsFn(type, replAttrs, replTypes);
}

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

MLIRContext *Type::getContext() const { return getDialect().getContext(); }

bool Type::isFloat4E2M1FN() const { return llvm::isa<Float4E2M1FNType>(*this); }
bool Type::isFloat6E2M3FN() const { return llvm::isa<Float6E2M3FNType>(*this); }
bool Type::isFloat6E3M2FN() const { return llvm::isa<Float6E3M2FNType>(*this); }
bool Type::isFloat8E5M2() const { return llvm::isa<Float8E5M2Type>(*this); }
bool Type::isFloat8E4M3() const { return llvm::isa<Float8E4M3Type>(*this); }
bool Type::isFloat8E4M3FN() const { return llvm::isa<Float8E4M3FNType>(*this); }
bool Type::isFloat8E5M2FNUZ() const {
  return llvm::isa<Float8E5M2FNUZType>(*this);
}
bool Type::isFloat8E4M3FNUZ() const {
  return llvm::isa<Float8E4M3FNUZType>(*this);
}
bool Type::isFloat8E4M3B11FNUZ() const {
  return llvm::isa<Float8E4M3B11FNUZType>(*this);
}
bool Type::isFloat8E3M4() const { return llvm::isa<Float8E3M4Type>(*this); }
bool Type::isBF16() const { return llvm::isa<BFloat16Type>(*this); }
bool Type::isF16() const { return llvm::isa<Float16Type>(*this); }
bool Type::isTF32() const { return llvm::isa<FloatTF32Type>(*this); }
bool Type::isF32() const { return llvm::isa<Float32Type>(*this); }
bool Type::isF64() const { return llvm::isa<Float64Type>(*this); }
bool Type::isF80() const { return llvm::isa<Float80Type>(*this); }
bool Type::isF128() const { return llvm::isa<Float128Type>(*this); }

bool Type::isIndex() const { return llvm::isa<IndexType>(*this); }

bool Type::isInteger() const { return llvm::isa<IntegerType>(*this); }

/// Return true if this is an integer type with the specified width.
bool Type::isInteger(unsigned width) const {
  if (auto intTy = llvm::dyn_cast<IntegerType>(*this))
    return intTy.getWidth() == width;
  return false;
}

bool Type::isSignlessInteger() const {
  if (auto intTy = llvm::dyn_cast<IntegerType>(*this))
    return intTy.isSignless();
  return false;
}

bool Type::isSignlessInteger(unsigned width) const {
  if (auto intTy = llvm::dyn_cast<IntegerType>(*this))
    return intTy.isSignless() && intTy.getWidth() == width;
  return false;
}

bool Type::isSignedInteger() const {
  if (auto intTy = llvm::dyn_cast<IntegerType>(*this))
    return intTy.isSigned();
  return false;
}

bool Type::isSignedInteger(unsigned width) const {
  if (auto intTy = llvm::dyn_cast<IntegerType>(*this))
    return intTy.isSigned() && intTy.getWidth() == width;
  return false;
}

bool Type::isUnsignedInteger() const {
  if (auto intTy = llvm::dyn_cast<IntegerType>(*this))
    return intTy.isUnsigned();
  return false;
}

bool Type::isUnsignedInteger(unsigned width) const {
  if (auto intTy = llvm::dyn_cast<IntegerType>(*this))
    return intTy.isUnsigned() && intTy.getWidth() == width;
  return false;
}

bool Type::isSignlessIntOrIndex() const {
  return isSignlessInteger() || llvm::isa<IndexType>(*this);
}

bool Type::isSignlessIntOrIndexOrFloat() const {
  return isSignlessInteger() || llvm::isa<IndexType, FloatType>(*this);
}

bool Type::isSignlessIntOrFloat() const {
  return isSignlessInteger() || llvm::isa<FloatType>(*this);
}

bool Type::isIntOrIndex() const {
  return llvm::isa<IntegerType>(*this) || isIndex();
}

bool Type::isIntOrFloat() const {
  return llvm::isa<IntegerType, FloatType>(*this);
}

bool Type::isIntOrIndexOrFloat() const { return isIntOrFloat() || isIndex(); }

unsigned Type::getIntOrFloatBitWidth() const {
  assert(isIntOrFloat() && "only integers and floats have a bitwidth");
  if (auto intType = llvm::dyn_cast<IntegerType>(*this))
    return intType.getWidth();
  return llvm::cast<FloatType>(*this).getWidth();
}
