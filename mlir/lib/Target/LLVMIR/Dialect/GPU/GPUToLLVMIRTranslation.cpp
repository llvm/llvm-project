//===- GPUToLLVMIRTranslation.cpp - Translate GPU dialect to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR GPU dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"

using namespace mlir;

namespace {

class GPUDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    return isa<gpu::GPUModuleOp>(op) ? success() : failure();
  }
};

} // namespace

void mlir::registerGPUDialectTranslation(DialectRegistry &registry) {
  registry.insert<gpu::GPUDialect>();
  registry.addExtension(+[](MLIRContext *ctx, gpu::GPUDialect *dialect) {
    dialect->addInterfaces<GPUDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerGPUDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerGPUDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
