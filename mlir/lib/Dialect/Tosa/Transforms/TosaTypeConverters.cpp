
//===- TosaTypeConverters.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type converters for lowering TOSA to linalg/arith.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

void mlir::tosa::populateTosaTypeConversion(TypeConverter &converter) {
  converter.addConversion([&](Type type) -> std::optional<Type> {
    if (type.isUnsignedInteger()) {
      return IntegerType::get(type.getContext(), type.getIntOrFloatBitWidth(),
                              IntegerType::SignednessSemantics::Signless);
    }
    return type;
  });
  converter.addConversion([&](TensorType type) -> std::optional<Type> {
    auto converted = converter.convertType(type.getElementType());
    if (!converted)
      return {};
    return type.clone(converted);
  });
  converter.addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
  converter.addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}
