//===- ConversionTarget.cpp - Target for converting to the LLVM dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

mlir::LLVMConversionTarget::LLVMConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  this->addLegalDialect<LLVM::LLVMDialect>();
  this->addLegalOp<UnrealizedConversionCastOp>();
}
