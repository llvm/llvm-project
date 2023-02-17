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
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

/// Matches a ConstantIndexOp.
/// TODO: This should probably just be a general matcher that uses matchConstant
/// and checks the operation for an index type.
detail::op_matcher<arith::ConstantIndexOp> mlir::matchConstantIndex() {
  return detail::op_matcher<arith::ConstantIndexOp>();
}

// Detects the `values` produced by a ConstantIndexOp and places the new
// constant in place of the corresponding sentinel value.
void mlir::canonicalizeSubViewPart(
    SmallVectorImpl<OpFoldResult> &values,
    llvm::function_ref<bool(int64_t)> isDynamic) {
  for (OpFoldResult &ofr : values) {
    if (ofr.is<Attribute>())
      continue;
    // Newly static, move from Value to constant.
    if (auto cstOp =
            ofr.dyn_cast<Value>().getDefiningOp<arith::ConstantIndexOp>())
      ofr = OpBuilder(cstOp).getIndexAttr(cstOp.value());
  }
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
    if (auto cstOp =
            ofr.dyn_cast<Value>().getDefiningOp<arith::ConstantIndexOp>()) {
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
  if (auto value = ofr.dyn_cast<Value>())
    return value;
  auto attr = ofr.dyn_cast<Attribute>().dyn_cast<IntegerAttr>();
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

  auto targetIntegerType = targetType.dyn_cast<IntegerType>();
  auto valueIntegerType = value.getType().dyn_cast<IntegerType>();
  assert(targetIntegerType && valueIntegerType &&
         "unexpected cast between types other than integers and index");
  assert(targetIntegerType.getSignedness() == valueIntegerType.getSignedness());

  if (targetIntegerType.getWidth() > valueIntegerType.getWidth())
    return b.create<arith::ExtSIOp>(loc, targetIntegerType, value);
  return b.create<arith::TruncIOp>(loc, targetIntegerType, value);
}

Value mlir::convertScalarToDtype(OpBuilder &b, Location loc, Value operand,
                                 Type toType, bool isUnsignedCast) {
  if (operand.getType() == toType)
    return operand;
  if (auto toIntType = toType.dyn_cast<IntegerType>()) {
    // If operand is floating point, cast directly to the int type.
    if (operand.getType().isa<FloatType>()) {
      if (isUnsignedCast)
        return b.create<arith::FPToUIOp>(loc, toType, operand);
      return b.create<arith::FPToSIOp>(loc, toType, operand);
    }
    // Cast index operands directly to the int type.
    if (operand.getType().isIndex())
      return b.create<arith::IndexCastOp>(loc, toType, operand);
    if (auto fromIntType = operand.getType().dyn_cast<IntegerType>()) {
      // Either extend or truncate.
      if (toIntType.getWidth() > fromIntType.getWidth()) {
        if (isUnsignedCast)
          return b.create<arith::ExtUIOp>(loc, toType, operand);
        return b.create<arith::ExtSIOp>(loc, toType, operand);
      }
      if (toIntType.getWidth() < fromIntType.getWidth())
        return b.create<arith::TruncIOp>(loc, toType, operand);
    }
  } else if (auto toFloatType = toType.dyn_cast<FloatType>()) {
    // If operand is integer, cast directly to the float type.
    // Note that it is unclear how to cast from BF16<->FP16.
    if (operand.getType().isa<IntegerType>()) {
      if (isUnsignedCast)
        return b.create<arith::UIToFPOp>(loc, toFloatType, operand);
      return b.create<arith::SIToFPOp>(loc, toFloatType, operand);
    }
    if (auto fromFloatType = operand.getType().dyn_cast<FloatType>()) {
      if (toFloatType.getWidth() > fromFloatType.getWidth())
        return b.create<arith::ExtFOp>(loc, toFloatType, operand);
      if (toFloatType.getWidth() < fromFloatType.getWidth())
        return b.create<arith::TruncFOp>(loc, toFloatType, operand);
    }
  }
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
  if (lhs.getType().isa<FloatType>())
    return b.create<arith::AddFOp>(loc, lhs, rhs);
  return b.create<arith::AddIOp>(loc, lhs, rhs);
}
Value ArithBuilder::sub(Value lhs, Value rhs) {
  if (lhs.getType().isa<FloatType>())
    return b.create<arith::SubFOp>(loc, lhs, rhs);
  return b.create<arith::SubIOp>(loc, lhs, rhs);
}
Value ArithBuilder::mul(Value lhs, Value rhs) {
  if (lhs.getType().isa<FloatType>())
    return b.create<arith::MulFOp>(loc, lhs, rhs);
  return b.create<arith::MulIOp>(loc, lhs, rhs);
}
Value ArithBuilder::sgt(Value lhs, Value rhs) {
  if (lhs.getType().isa<FloatType>())
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, lhs, rhs);
  return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs);
}
Value ArithBuilder::slt(Value lhs, Value rhs) {
  if (lhs.getType().isa<FloatType>())
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, lhs, rhs);
  return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs);
}
Value ArithBuilder::select(Value cmp, Value lhs, Value rhs) {
  return b.create<arith::SelectOp>(loc, cmp, lhs, rhs);
}
