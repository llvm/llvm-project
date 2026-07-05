//===- LoweringOptions.cpp -  Common config for lowering to LLVM ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

using namespace mlir;

mlir::LowerToLLVMOptions::LowerToLLVMOptions(MLIRContext *ctx)
    : LowerToLLVMOptions(ctx, DataLayout()) {}

mlir::LowerToLLVMOptions::LowerToLLVMOptions(MLIRContext *ctx,
                                             const DataLayout &dl) {
  indexBitwidth = dl.getTypeSizeInBits(IndexType::get(ctx));
}
