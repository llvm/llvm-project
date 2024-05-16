//===- TypeConversions.cpp - Convert signless types into C/C++ types ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include <optional>

using namespace mlir;

namespace {

std::optional<Value> materializeAsUnrealizedCast(OpBuilder &builder,
                                                 Type resultType,
                                                 ValueRange inputs,
                                                 Location loc) {
  if (inputs.size() != 1)
    return std::nullopt;

  return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
      .getResult(0);
}

} // namespace

void mlir::populateEmitCSizeTypeConversions(TypeConverter &converter) {
  converter.addConversion(
      [](IndexType type) { return emitc::SizeTType::get(type.getContext()); });

  converter.addSourceMaterialization(materializeAsUnrealizedCast);
  converter.addTargetMaterialization(materializeAsUnrealizedCast);
  converter.addArgumentMaterialization(materializeAsUnrealizedCast);
}
