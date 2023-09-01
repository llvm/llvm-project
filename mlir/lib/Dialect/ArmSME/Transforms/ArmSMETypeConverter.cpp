//===- ArmSMETypeConverter.cpp - Convert builtin to LLVM dialect types ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"

using namespace mlir;
arm_sme::ArmSMETypeConverter::ArmSMETypeConverter(
    MLIRContext *ctx, const LowerToLLVMOptions &options)
    : LLVMTypeConverter(ctx, options) {
  addConversion([&](VectorType type) { return convertVectorType(type); });
}

Type arm_sme::ArmSMETypeConverter::convertVectorType(VectorType type) const {
  if (arm_sme::isValidSMETileVectorType(type))
    return type;
  return LLVMTypeConverter::convertVectorType(type);
}
