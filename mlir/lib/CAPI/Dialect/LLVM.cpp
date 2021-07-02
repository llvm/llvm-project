//===- LLVM.cpp - C Interface for LLVM dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/LLVM.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;
using namespace mlir::LLVM;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(LLVM, llvm, LLVMDialect)

MlirType mlirLLVMPointerTypeGet(MlirType pointee, unsigned addressSpace) {
  return wrap(LLVMPointerType::get(unwrap(pointee), addressSpace));
}
