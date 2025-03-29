//===- SPIRVToLLVMIRTranslation.cpp - Translate SPIR-V to LLVM IR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR SPIR-V dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/SPIRV/SPIRVToLLVMIRTranslation.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

using namespace mlir;
using namespace mlir::LLVM;

void mlir::registerSPIRVDialectTranslation(DialectRegistry &registry) {
  registry.insert<spirv::SPIRVDialect>();
}

void mlir::registerSPIRVDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerSPIRVDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
