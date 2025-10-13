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

Value materializeAsUnrealizedCast(OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) {
  if (inputs.size() != 1)
    return Value();

  return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
      .getResult(0);
}

} // namespace

void mlir::populateEmitCSizeTTypeConversions(TypeConverter &converter) {
  converter.addConversion(
      [](IndexType type) { return emitc::SizeTType::get(type.getContext()); });

  converter.addSourceMaterialization(materializeAsUnrealizedCast);
  converter.addTargetMaterialization(materializeAsUnrealizedCast);
}

/// Get an unsigned integer or size data type corresponding to \p ty.
std::optional<Type> mlir::emitc::getUnsignedTypeFor(Type ty) {
  if (ty.isInteger())
    return IntegerType::get(ty.getContext(), ty.getIntOrFloatBitWidth(),
                            IntegerType::SignednessSemantics::Unsigned);
  if (isa<PtrDiffTType, SignedSizeTType>(ty))
    return SizeTType::get(ty.getContext());
  if (isa<SizeTType>(ty))
    return ty;
  return {};
}

/// Get a signed integer or size data type corresponding to \p ty that supports
/// arithmetic on negative values.
std::optional<Type> mlir::emitc::getSignedTypeFor(Type ty) {
  if (ty.isInteger())
    return IntegerType::get(ty.getContext(), ty.getIntOrFloatBitWidth(),
                            IntegerType::SignednessSemantics::Signed);
  if (isa<SizeTType, SignedSizeTType>(ty))
    return PtrDiffTType::get(ty.getContext());
  if (isa<PtrDiffTType>(ty))
    return ty;
  return {};
}
