//===- Utils.cpp - Utilities to support the Linalg dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

/// Matches a ConstantIndexOp.
/// TODO: This should probably just be a general matcher that uses matchConstant
/// and checks the operation for an index type.
detail::op_matcher<arith::ConstantIndexOp> mlir::matchConstantIndex() {
  return detail::op_matcher<arith::ConstantIndexOp>();
}

// Returns `success` when any of the elements in `ofrs` was produced by
// arith::ConstantIndexOp. In that case the constant attribute replaces the
// Value. Returns `failure` when no folding happened.
LogicalResult mlir::foldDynamicIndexList(Builder &b,
                                         SmallVectorImpl<OpFoldResult> &ofrs) {
  bool valuesChanged = false;
  for (OpFoldResult &ofr : ofrs) {
    if (ofr.is<Attribute>())
      continue;
    // Newly static, move from Value to constant.
    if (auto cstOp = llvm::dyn_cast_if_present<Value>(ofr)
                         .getDefiningOp<arith::ConstantIndexOp>()) {
      ofr = b.getIndexAttr(cstOp.value());
      valuesChanged = true;
    }
  }
  return success(valuesChanged);
}

llvm::SmallBitVector mlir::getPositionsOfShapeOne(unsigned rank,
                                                  ArrayRef<int64_t> shape) {
  llvm::SmallBitVector dimsToProject(shape.size());
  for (unsigned pos = 0, e = shape.size(); pos < e && rank > 0; ++pos) {
    if (shape[pos] == 1) {
      dimsToProject.set(pos);
      --rank;
    }
  }
  return dimsToProject;
}

Value mlir::getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                            OpFoldResult ofr) {
  if (auto value = llvm::dyn_cast_if_present<Value>(ofr))
    return value;
  auto attr = dyn_cast<IntegerAttr>(llvm::dyn_cast_if_present<Attribute>(ofr));
  assert(attr && "expect the op fold result casts to an integer attribute");
  return b.create<arith::ConstantIndexOp>(loc, attr.getValue().getSExtValue());
}

Value mlir::getValueOrCreateCastToIndexLike(OpBuilder &b, Location loc,
                                            Type targetType, Value value) {
  if (targetType == value.getType())
    return value;

  bool targetIsIndex = targetType.isIndex();
  bool valueIsIndex = value.getType().isIndex();
  if (targetIsIndex ^ valueIsIndex)
    return b.create<arith::IndexCastOp>(loc, targetType, value);

  auto targetIntegerType = dyn_cast<IntegerType>(targetType);
  auto valueIntegerType = dyn_cast<IntegerType>(value.getType());
  assert(targetIntegerType && valueIntegerType &&
         "unexpected cast between types other than integers and index");
  assert(targetIntegerType.getSignedness() == valueIntegerType.getSignedness());

  if (targetIntegerType.getWidth() > valueIntegerType.getWidth())
    return b.create<arith::ExtSIOp>(loc, targetIntegerType, value);
  return b.create<arith::TruncIOp>(loc, targetIntegerType, value);
}

static Value convertScalarToIntDtype(ImplicitLocOpBuilder &b, Value operand,
                                     IntegerType toType, bool isUnsigned) {
  // If operand is floating point, cast directly to the int type.
  if (isa<FloatType>(operand.getType())) {
    if (isUnsigned)
      return b.create<arith::FPToUIOp>(toType, operand);
    return b.create<arith::FPToSIOp>(toType, operand);
  }
  // Cast index operands directly to the int type.
  if (operand.getType().isIndex())
    return b.create<arith::IndexCastOp>(toType, operand);
  if (auto fromIntType = dyn_cast<IntegerType>(operand.getType())) {
    // Either extend or truncate.
    if (toType.getWidth() > fromIntType.getWidth()) {
      if (isUnsigned)
        return b.create<arith::ExtUIOp>(toType, operand);
      return b.create<arith::ExtSIOp>(toType, operand);
    }
    if (toType.getWidth() < fromIntType.getWidth())
      return b.create<arith::TruncIOp>(toType, operand);
    return operand;
  }

  return {};
}

static Value convertScalarToFpDtype(ImplicitLocOpBuilder &b, Value operand,
                                    FloatType toType, bool isUnsigned) {
  // If operand is integer, cast directly to the float type.
  // Note that it is unclear how to cast from BF16<->FP16.
  if (isa<IntegerType>(operand.getType())) {
    if (isUnsigned)
      return b.create<arith::UIToFPOp>(toType, operand);
    return b.create<arith::SIToFPOp>(toType, operand);
  }
  if (auto fromFpTy = dyn_cast<FloatType>(operand.getType())) {
    if (toType.getWidth() > fromFpTy.getWidth())
      return b.create<arith::ExtFOp>(toType, operand);
    if (toType.getWidth() < fromFpTy.getWidth())
      return b.create<arith::TruncFOp>(toType, operand);
    return operand;
  }

  return {};
}

static Value convertScalarToComplexDtype(ImplicitLocOpBuilder &b, Value operand,
                                         ComplexType targetType,
                                         bool isUnsigned) {
  if (auto fromComplexType = dyn_cast<ComplexType>(operand.getType())) {
    if (isa<FloatType>(targetType.getElementType()) &&
        isa<FloatType>(fromComplexType.getElementType())) {
      Value real = b.create<complex::ReOp>(operand);
      Value imag = b.create<complex::ImOp>(operand);
      Type targetETy = targetType.getElementType();
      if (targetType.getElementType().getIntOrFloatBitWidth() <
          fromComplexType.getElementType().getIntOrFloatBitWidth()) {
        real = b.create<arith::TruncFOp>(targetETy, real);
        imag = b.create<arith::TruncFOp>(targetETy, imag);
      } else {
        real = b.create<arith::ExtFOp>(targetETy, real);
        imag = b.create<arith::ExtFOp>(targetETy, imag);
      }
      return b.create<complex::CreateOp>(targetType, real, imag);
    }
  }

  if (auto fromFpType = dyn_cast<FloatType>(operand.getType())) {
    FloatType toFpTy = cast<FloatType>(targetType.getElementType());
    auto toBitwidth = toFpTy.getIntOrFloatBitWidth();
    Value from = operand;
    if (from.getType().getIntOrFloatBitWidth() < toBitwidth) {
      from = b.create<arith::ExtFOp>(toFpTy, from);
    }
    if (from.getType().getIntOrFloatBitWidth() > toBitwidth) {
      from = b.create<arith::TruncFOp>(toFpTy, from);
    }
    Value zero = b.create<mlir::arith::ConstantFloatOp>(
        mlir::APFloat(toFpTy.getFloatSemantics(), 0), toFpTy);
    return b.create<complex::CreateOp>(targetType, from, zero);
  }

  if (auto fromIntType = dyn_cast<IntegerType>(operand.getType())) {
    FloatType toFpTy = cast<FloatType>(targetType.getElementType());
    Value from = operand;
    if (isUnsigned) {
      from = b.create<arith::UIToFPOp>(toFpTy, from);
    } else {
      from = b.create<arith::SIToFPOp>(toFpTy, from);
    }
    Value zero = b.create<mlir::arith::ConstantFloatOp>(
        mlir::APFloat(toFpTy.getFloatSemantics(), 0), toFpTy);
    return b.create<complex::CreateOp>(targetType, from, zero);
  }

  return {};
}

Value mlir::convertScalarToDtype(OpBuilder &b, Location loc, Value operand,
                                 Type toType, bool isUnsignedCast) {
  if (operand.getType() == toType)
    return operand;
  ImplicitLocOpBuilder ib(loc, b);
  Value result;
  if (auto intTy = dyn_cast<IntegerType>(toType)) {
    result = convertScalarToIntDtype(ib, operand, intTy, isUnsignedCast);
  } else if (auto floatTy = dyn_cast<FloatType>(toType)) {
    result = convertScalarToFpDtype(ib, operand, floatTy, isUnsignedCast);
  } else if (auto complexTy = dyn_cast<ComplexType>(toType)) {
    result =
        convertScalarToComplexDtype(ib, operand, complexTy, isUnsignedCast);
  }

  if (result)
    return result;

  emitWarning(loc) << "could not cast operand of type " << operand.getType()
                   << " to " << toType;
  return operand;
}

SmallVector<Value>
mlir::getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                      ArrayRef<OpFoldResult> valueOrAttrVec) {
  return llvm::to_vector<4>(
      llvm::map_range(valueOrAttrVec, [&](OpFoldResult value) -> Value {
        return getValueOrCreateConstantIndexOp(b, loc, value);
      }));
}

Value ArithBuilder::_and(Value lhs, Value rhs) {
  return b.create<arith::AndIOp>(loc, lhs, rhs);
}
Value ArithBuilder::add(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return b.create<arith::AddFOp>(loc, lhs, rhs);
  return b.create<arith::AddIOp>(loc, lhs, rhs);
}
Value ArithBuilder::sub(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return b.create<arith::SubFOp>(loc, lhs, rhs);
  return b.create<arith::SubIOp>(loc, lhs, rhs);
}
Value ArithBuilder::mul(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return b.create<arith::MulFOp>(loc, lhs, rhs);
  return b.create<arith::MulIOp>(loc, lhs, rhs);
}
Value ArithBuilder::sgt(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, lhs, rhs);
  return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs);
}
Value ArithBuilder::slt(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, lhs, rhs);
  return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs);
}
Value ArithBuilder::select(Value cmp, Value lhs, Value rhs) {
  return b.create<arith::SelectOp>(loc, cmp, lhs, rhs);
}
