//===- AGXIRToLLVMIRTranslation.cpp - Translate AGXIR to LLVM IR ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR AGXIR dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/AGXIR/AGXIRToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/AGXIRDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAGX.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

namespace {
class AGXIRDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/AGXIRConversions.inc"

    return failure();
  }

  /// Functions marked as kernels become agx_kernel entry points.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto *dialect = dyn_cast<AGXIR::AGXIRDialect>(attribute.getNameDialect());
    if (dialect->getKernelAttrHelper().getName() == attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return op->emitOpError(Twine(attribute.getName()) +
                               " is only supported on `llvm.func` operations");
      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvmFunc->setCallingConv(llvm::CallingConv::AGX_KERNEL);
    }
    return success();
  }
};
} // namespace

void mlir::registerAGXIRDialectTranslation(DialectRegistry &registry) {
  registry.insert<AGXIR::AGXIRDialect>();
  registry.addExtension(+[](MLIRContext *ctx, AGXIR::AGXIRDialect *dialect) {
    dialect->addInterfaces<AGXIRDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerAGXIRDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerAGXIRDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
