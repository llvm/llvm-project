//===- ArmSMETypeConverter.cpp - Convert builtin to LLVM dialect types ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/Transforms/Passes.h"

using namespace mlir;
arm_sme::ArmSMETypeConverter::ArmSMETypeConverter(
    MLIRContext *ctx, const LowerToLLVMOptions &options)
    : LLVMTypeConverter(ctx, options) {
  // Disable LLVM type conversion for vectors. This is to prevent 2-d scalable
  // vectors (common in the context of ArmSME), e.g.
  //    `vector<[16]x[16]xi8>`,
  // entering the LLVM Type converter. LLVM does not support arrays of scalable
  // vectors, but in the case of SME such types are effectively eliminated when
  // emitting ArmSME LLVM IR intrinsics.
  addConversion([&](VectorType type) { return type; });
}
