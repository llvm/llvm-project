//===- RegisterEverything.cpp - Register all MLIR entities ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/RegisterEverything.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

void mlirRegisterAllDialects(MlirDialectRegistry registry) {
  mlir::registerAllDialects(*unwrap(registry));
  mlir::registerAllExtensions(*unwrap(registry));
}

void mlirRegisterAllLLVMTranslations(MlirContext context) {
  auto &ctx = *unwrap(context);
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  ctx.appendDialectRegistry(registry);
}

void mlirRegisterAllPasses() { mlir::registerAllPasses(); }
