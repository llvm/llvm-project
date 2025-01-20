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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SmallBitVector.h"
#include <numeric>

using namespace mlir;

std::optional<SmallVector<OpFoldResult>>
mlir::inferExpandShapeOutputShape(OpBuilder &b, Location loc,
                                  ShapedType expandedType,
                                  ArrayRef<ReassociationIndices> reassociation,
                                  ArrayRef<OpFoldResult> inputShape) {

  SmallVector<Value> outputShapeValues;
  SmallVector<int64_t> outputShapeInts;
  // For zero-rank inputs, all dims in result shape are unit extent.
  if (inputShape.empty()) {
    outputShapeInts.resize(expandedType.getRank(), 1);
    return getMixedValues(outputShapeInts, outputShapeValues, b);
  }

  // Check for all static shapes.
  if (expandedType.hasStaticShape()) {
    ArrayRef<int64_t> staticShape = expandedType.getShape();
    outputShapeInts.assign(staticShape.begin(), staticShape.end());
    return getMixedValues(outputShapeInts, outputShapeValues, b);
  }

  outputShapeInts.resize(expandedType.getRank(), ShapedType::kDynamic);
  for (const auto &it : llvm::enumerate(reassociation)) {
    ReassociationIndices indexGroup = it.value();

    int64_t indexGroupStaticSizesProductInt = 1;
    bool foundDynamicShape = false;
    for (int64_t index : indexGroup) {
      int64_t outputDimSize = expandedType.getDimSize(index);
      // Cannot infer expanded shape with multiple dynamic dims in the
      // same reassociation group!
      if (ShapedType::isDynamic(outputDimSize)) {
        if (foundDynamicShape)
          return std::nullopt;
        foundDynamicShape = true;
      } else {
        outputShapeInts[index] = outputDimSize;
        indexGroupStaticSizesProductInt *= outputDimSize;
      }
    }
    if (!foundDynamicShape)
      continue;

    int64_t inputIndex = it.index();
    // Call get<Value>() under the assumption that we're not casting
    // dynamism.
    Value indexGroupSize = cast<Value>(inputShape[inputIndex]);
    Value indexGroupStaticSizesProduct =
        b.create<arith::ConstantIndexOp>(loc, indexGroupStaticSizesProductInt);
    Value dynamicDimSize = b.createOrFold<arith::DivUIOp>(
        loc, indexGroupSize, indexGroupStaticSizesProduct);
    outputShapeValues.push_back(dynamicDimSize);
  }

  if ((int64_t)outputShapeValues.size() !=
      llvm::count(outputShapeInts, ShapedType::kDynamic))
    return std::nullopt;

  return getMixedValues(outputShapeInts, outputShapeValues, b);
}

/// Matches a ConstantIndexOp.
/// TODO: This should probably just be a general matcher that uses matchConstant
/// and checks the operation for an index type.
detail::op_matcher<arith::ConstantIndexOp> mlir::matchConstantIndex() {
  return detail::op_matcher<arith::ConstantIndexOp>();
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

Value mlir::getValueOrCreateConstantIntOp(OpBuilder &b, Location loc,
                                          OpFoldResult ofr) {
  if (auto value = dyn_cast_if_present<Value>(ofr))
    return value;
  auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
  return b.create<arith::ConstantOp>(
      loc, b.getIntegerAttr(attr.getType(), attr.getValue().getSExtValue()));
}

Value mlir::getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                            OpFoldResult ofr) {
  if (auto value = dyn_cast_if_present<Value>(ofr))
    return value;
  auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
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

  if (dyn_cast<FloatType>(operand.getType())) {
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

  if (dyn_cast<IntegerType>(operand.getType())) {
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

Value mlir::createScalarOrSplatConstant(OpBuilder &builder, Location loc,
                                        Type type, const APInt &value) {
  TypedAttr attr;
  if (isa<IntegerType>(type)) {
    attr = builder.getIntegerAttr(type, value);
  } else {
    auto vecTy = cast<ShapedType>(type);
    attr = SplatElementsAttr::get(vecTy, value);
  }

  return builder.create<arith::ConstantOp>(loc, attr);
}

Value mlir::createScalarOrSplatConstant(OpBuilder &builder, Location loc,
                                        Type type, int64_t value) {
  unsigned elementBitWidth = 0;
  if (auto intTy = dyn_cast<IntegerType>(type))
    elementBitWidth = intTy.getWidth();
  else
    elementBitWidth = cast<ShapedType>(type).getElementTypeBitWidth();

  return createScalarOrSplatConstant(builder, loc, type,
                                     APInt(elementBitWidth, value));
}

Value mlir::createScalarOrSplatConstant(OpBuilder &builder, Location loc,
                                        Type type, const APFloat &value) {
  if (isa<FloatType>(type))
    return builder.createOrFold<arith::ConstantOp>(
        loc, type, builder.getFloatAttr(type, value));
  TypedAttr splat = SplatElementsAttr::get(cast<ShapedType>(type), value);
  return builder.createOrFold<arith::ConstantOp>(loc, type, splat);
}

Type mlir::getType(OpFoldResult ofr) {
  if (auto value = dyn_cast_if_present<Value>(ofr))
    return value.getType();
  auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
  return attr.getType();
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

namespace mlir::arith {

Value createProduct(OpBuilder &builder, Location loc, ArrayRef<Value> values) {
  return createProduct(builder, loc, values, values.front().getType());
}

Value createProduct(OpBuilder &builder, Location loc, ArrayRef<Value> values,
                    Type resultType) {
  Value one = builder.create<ConstantOp>(loc, resultType,
                                         builder.getOneAttr(resultType));
  ArithBuilder arithBuilder(builder, loc);
  return std::accumulate(
      values.begin(), values.end(), one,
      [&arithBuilder](Value acc, Value v) { return arithBuilder.mul(acc, v); });
}

/// Map strings to float types.
std::optional<FloatType> parseFloatType(MLIRContext *ctx, StringRef name) {
  Builder b(ctx);
  return llvm::StringSwitch<std::optional<FloatType>>(name)
      .Case("f4E2M1FN", b.getFloat4E2M1FNType())
      .Case("f6E2M3FN", b.getFloat6E2M3FNType())
      .Case("f6E3M2FN", b.getFloat6E3M2FNType())
      .Case("f8E5M2", b.getFloat8E5M2Type())
      .Case("f8E4M3", b.getFloat8E4M3Type())
      .Case("f8E4M3FN", b.getFloat8E4M3FNType())
      .Case("f8E5M2FNUZ", b.getFloat8E5M2FNUZType())
      .Case("f8E4M3FNUZ", b.getFloat8E4M3FNUZType())
      .Case("f8E3M4", b.getFloat8E3M4Type())
      .Case("f8E8M0FNU", b.getFloat8E8M0FNUType())
      .Case("bf16", b.getBF16Type())
      .Case("f16", b.getF16Type())
      .Case("f32", b.getF32Type())
      .Case("f64", b.getF64Type())
      .Case("f80", b.getF80Type())
      .Case("f128", b.getF128Type())
      .Default(std::nullopt);
}

} // namespace mlir::arith
