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
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {
LogicalResult launchKernel(gpu::LaunchFuncOp launchOp,
                           llvm::IRBuilderBase &builder,
                           LLVM::ModuleTranslation &moduleTranslation) {
  auto kernelBinary = SymbolTable::lookupNearestSymbolFrom<gpu::BinaryOp>(
      launchOp, launchOp.getKernelModuleName());
  if (!kernelBinary) {
    launchOp.emitError("Couldn't find the binary holding the kernel: ")
        << launchOp.getKernelModuleName();
    return failure();
  }
  auto offloadingHandler =
      dyn_cast<gpu::OffloadingLLVMTranslationAttrInterface>(
          kernelBinary.getOffloadingHandlerAttr());
  assert(offloadingHandler && "Invalid offloading handler.");
  return offloadingHandler.launchKernel(launchOp, kernelBinary, builder,
                                        moduleTranslation);
}

class GPUDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *operation, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    return llvm::TypeSwitch<Operation *, LogicalResult>(operation)
        .Case([&](gpu::GPUModuleOp) { return success(); })
        .Case([&](gpu::BinaryOp op) {
          auto offloadingHandler =
              dyn_cast<gpu::OffloadingLLVMTranslationAttrInterface>(
                  op.getOffloadingHandlerAttr());
          assert(offloadingHandler && "Invalid offloading handler.");
          return offloadingHandler.embedBinary(op, builder, moduleTranslation);
        })
        .Case([&](gpu::LaunchFuncOp op) {
          return launchKernel(op, builder, moduleTranslation);
        })
        .Default([&](Operation *op) {
          return op->emitError("unsupported GPU operation: ") << op->getName();
        });
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
