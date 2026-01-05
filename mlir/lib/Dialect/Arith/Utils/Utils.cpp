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
        arith::ConstantIndexOp::create(b, loc, indexGroupStaticSizesProductInt);
    Value dynamicDimSize = b.createOrFold<arith::DivSIOp>(
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
  return arith::ConstantOp::create(
      b, loc, b.getIntegerAttr(attr.getType(), attr.getValue().getSExtValue()));
}

Value mlir::getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                            OpFoldResult ofr) {
  if (auto value = dyn_cast_if_present<Value>(ofr))
    return value;
  auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
  return arith::ConstantIndexOp::create(b, loc, attr.getValue().getSExtValue());
}

Value mlir::getValueOrCreateCastToIndexLike(OpBuilder &b, Location loc,
                                            Type targetType, Value value) {
  if (targetType == value.getType())
    return value;

  bool targetIsIndex = targetType.isIndex();
  bool valueIsIndex = value.getType().isIndex();
  if (targetIsIndex ^ valueIsIndex)
    return arith::IndexCastOp::create(b, loc, targetType, value);

  auto targetIntegerType = dyn_cast<IntegerType>(targetType);
  auto valueIntegerType = dyn_cast<IntegerType>(value.getType());
  assert(targetIntegerType && valueIntegerType &&
         "unexpected cast between types other than integers and index");
  assert(targetIntegerType.getSignedness() == valueIntegerType.getSignedness());

  if (targetIntegerType.getWidth() > valueIntegerType.getWidth())
    return arith::ExtSIOp::create(b, loc, targetIntegerType, value);
  return arith::TruncIOp::create(b, loc, targetIntegerType, value);
}

static Value convertScalarToIntDtype(ImplicitLocOpBuilder &b, Value operand,
                                     IntegerType toType, bool isUnsigned) {
  // If operand is floating point, cast directly to the int type.
  if (isa<FloatType>(operand.getType())) {
    if (isUnsigned)
      return arith::FPToUIOp::create(b, toType, operand);
    return arith::FPToSIOp::create(b, toType, operand);
  }
  // Cast index operands directly to the int type.
  if (operand.getType().isIndex())
    return arith::IndexCastOp::create(b, toType, operand);
  if (auto fromIntType = dyn_cast<IntegerType>(operand.getType())) {
    // Either extend or truncate.
    if (toType.getWidth() > fromIntType.getWidth()) {
      if (isUnsigned)
        return arith::ExtUIOp::create(b, toType, operand);
      return arith::ExtSIOp::create(b, toType, operand);
    }
    if (toType.getWidth() < fromIntType.getWidth())
      return arith::TruncIOp::create(b, toType, operand);
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
      return arith::UIToFPOp::create(b, toType, operand);
    return arith::SIToFPOp::create(b, toType, operand);
  }
  if (auto fromFpTy = dyn_cast<FloatType>(operand.getType())) {
    if (toType.getWidth() > fromFpTy.getWidth())
      return arith::ExtFOp::create(b, toType, operand);
    if (toType.getWidth() < fromFpTy.getWidth())
      return arith::TruncFOp::create(b, toType, operand);
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
      Value real = complex::ReOp::create(b, operand);
      Value imag = complex::ImOp::create(b, operand);
      Type targetETy = targetType.getElementType();
      if (targetType.getElementType().getIntOrFloatBitWidth() <
          fromComplexType.getElementType().getIntOrFloatBitWidth()) {
        real = arith::TruncFOp::create(b, targetETy, real);
        imag = arith::TruncFOp::create(b, targetETy, imag);
      } else {
        real = arith::ExtFOp::create(b, targetETy, real);
        imag = arith::ExtFOp::create(b, targetETy, imag);
      }
      return complex::CreateOp::create(b, targetType, real, imag);
    }
  }

  if (isa<FloatType>(operand.getType())) {
    FloatType toFpTy = cast<FloatType>(targetType.getElementType());
    auto toBitwidth = toFpTy.getIntOrFloatBitWidth();
    Value from = operand;
    if (from.getType().getIntOrFloatBitWidth() < toBitwidth) {
      from = arith::ExtFOp::create(b, toFpTy, from);
    }
    if (from.getType().getIntOrFloatBitWidth() > toBitwidth) {
      from = arith::TruncFOp::create(b, toFpTy, from);
    }
    Value zero = mlir::arith::ConstantFloatOp::create(
        b, toFpTy, mlir::APFloat(toFpTy.getFloatSemantics(), 0));
    return complex::CreateOp::create(b, targetType, from, zero);
  }

  if (isa<IntegerType>(operand.getType())) {
    FloatType toFpTy = cast<FloatType>(targetType.getElementType());
    Value from = operand;
    if (isUnsigned) {
      from = arith::UIToFPOp::create(b, toFpTy, from);
    } else {
      from = arith::SIToFPOp::create(b, toFpTy, from);
    }
    Value zero = mlir::arith::ConstantFloatOp::create(
        b, toFpTy, mlir::APFloat(toFpTy.getFloatSemantics(), 0));
    return complex::CreateOp::create(b, targetType, from, zero);
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

  return arith::ConstantOp::create(builder, loc, attr);
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
  return arith::AndIOp::create(b, loc, lhs, rhs);
}
Value ArithBuilder::add(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return arith::AddFOp::create(b, loc, lhs, rhs);
  return arith::AddIOp::create(b, loc, lhs, rhs, ovf);
}
Value ArithBuilder::sub(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return arith::SubFOp::create(b, loc, lhs, rhs);
  return arith::SubIOp::create(b, loc, lhs, rhs, ovf);
}
Value ArithBuilder::mul(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return arith::MulFOp::create(b, loc, lhs, rhs);
  return arith::MulIOp::create(b, loc, lhs, rhs, ovf);
}
Value ArithBuilder::sgt(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, lhs, rhs);
  return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, lhs, rhs);
}
Value ArithBuilder::slt(Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, lhs, rhs);
  return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, lhs, rhs);
}
Value ArithBuilder::select(Value cmp, Value lhs, Value rhs) {
  return arith::SelectOp::create(b, loc, cmp, lhs, rhs);
}

namespace mlir::arith {

Value createProduct(OpBuilder &builder, Location loc, ArrayRef<Value> values) {
  return createProduct(builder, loc, values, values.front().getType());
}

Value createProduct(OpBuilder &builder, Location loc, ArrayRef<Value> values,
                    Type resultType) {
  Value one = ConstantOp::create(builder, loc, resultType,
                                 builder.getOneAttr(resultType));
  ArithBuilder arithBuilder(builder, loc);
  return llvm::accumulate(values, one, [&arithBuilder](Value acc, Value v) {
    return arithBuilder.mul(acc, v);
  });
}

/// Map strings to float types.
std::optional<FloatType> parseFloatType(MLIRContext *ctx, StringRef name) {
  Builder b(ctx);
  return llvm::StringSwitch<std::optional<FloatType>>(name)
      .Case("f4E2M1FN", b.getType<Float4E2M1FNType>())
      .Case("f6E2M3FN", b.getType<Float6E2M3FNType>())
      .Case("f6E3M2FN", b.getType<Float6E3M2FNType>())
      .Case("f8E5M2", b.getType<Float8E5M2Type>())
      .Case("f8E4M3", b.getType<Float8E4M3Type>())
      .Case("f8E4M3FN", b.getType<Float8E4M3FNType>())
      .Case("f8E5M2FNUZ", b.getType<Float8E5M2FNUZType>())
      .Case("f8E4M3FNUZ", b.getType<Float8E4M3FNUZType>())
      .Case("f8E3M4", b.getType<Float8E3M4Type>())
      .Case("f8E8M0FNU", b.getType<Float8E8M0FNUType>())
      .Case("bf16", b.getType<BFloat16Type>())
      .Case("f16", b.getType<Float16Type>())
      .Case("f32", b.getType<Float32Type>())
      .Case("f64", b.getType<Float64Type>())
      .Case("f80", b.getType<Float80Type>())
      .Case("f128", b.getType<Float128Type>())
      .Default(std::nullopt);
}

} // namespace mlir::arith
