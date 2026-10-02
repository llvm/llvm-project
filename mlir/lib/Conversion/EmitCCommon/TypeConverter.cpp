//===- TypeConverter.cpp - Convert builtin to EmitC dialect types ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/EmitCCommon/TypeConverter.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

static bool isMemRefTypeLegalForEmitC(MemRefType memRefType) {
  return memRefType.hasStaticShape() && memRefType.getLayout().isIdentity() &&
         !llvm::is_contained(memRefType.getShape(), 0);
}

static Value materializeAsUnrealizedCast(OpBuilder &builder, Type resultType,
                                         ValueRange inputs, Location loc) {
  if (inputs.size() != 1)
    return Value();

  return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
      .getResult(0);
}

} // namespace

EmitCTypeConverter::EmitCTypeConverter(MLIRContext *ctx) {
  (void)ctx;

  addConversion([](Type type) -> std::optional<Type> {
    if (!emitc::isSupportedEmitCType(type))
      return {};
    return type;
  });

  addConversion([&](MemRefType memRefType) -> std::optional<Type> {
    if (!isMemRefTypeLegalForEmitC(memRefType))
      return {};

    Type convertedElementType = convertType(memRefType.getElementType());
    if (!convertedElementType)
      return {};

    if (memRefType.getRank() == 0)
      return emitc::PointerType::get(convertedElementType);
    return emitc::ArrayType::get(memRefType.getShape(), convertedElementType);
  });

  addSourceMaterialization(materializeAsUnrealizedCast);
  addTargetMaterialization(materializeAsUnrealizedCast);
}
