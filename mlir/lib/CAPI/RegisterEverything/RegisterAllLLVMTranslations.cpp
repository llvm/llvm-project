//===- RegisterAllPasses.cpp - Register all MLIR LLVM translations --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/RegisterAllLLVMTranslations.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

void mlirRegisterAllLLVMTranslations(MlirContext context) {
  auto &ctx = *unwrap(context);
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  ctx.appendDialectRegistry(registry);
}
