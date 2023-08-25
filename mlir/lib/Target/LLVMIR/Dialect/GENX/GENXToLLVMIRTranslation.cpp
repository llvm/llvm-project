//===- GENXToLLVMIRTranslation.cpp - Translate GENX to LLVM IR ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR GENX dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/GENX/GENXToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

// Create a call to SPIR device function.
static llvm::Value *createDeviceFunctionCall(llvm::IRBuilderBase &builder,
                                             StringRef fnName,
                                             llvm::Type *retType,
                                             ArrayRef<llvm::Type *> argTypes,
                                             ArrayRef<llvm::Value *> args) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  auto *functionType =
      llvm::FunctionType::get(retType, argTypes, /*isVarArg*/ false);
  auto *fn = dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fnName, functionType).getCallee());
  fn->setCallingConv(llvm::CallingConv::SPIR_FUNC);
  return builder.CreateCall(fn, args);
}

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the GENX dialect to LLVM IR.
class GENXDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/GENXConversions.inc"

    return failure();
  }

  /// Attaches metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return failure();

    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());
    StringAttr attrName = attribute.getName();
    Attribute attrVal = attribute.getValue();

    // Set calling convention for kernel
    if (attrName == GENX::GENXDialect::getKernelFuncAttrName())
      llvmFunc->setCallingConv(llvm::CallingConv::SPIR_KERNEL);

    auto attachMetadata = [&](StringRef name) {
      SmallVector<llvm::Metadata *, 3> metadata;
      llvm::Type *i64 = llvm::IntegerType::get(llvmContext, 64);
      for (int64_t i : extractFromI64ArrayAttr(attrVal)) {
        llvm::Constant *constant = llvm::ConstantInt::get(i64, i);
        metadata.push_back(llvm::ConstantAsMetadata::get(constant));
      }
      llvm::MDNode *node = llvm::MDNode::get(llvmContext, metadata);
      llvmFunc->setMetadata(name, node);
    };

    // Set max_work_group_size metadata.
    if (attrName == GENX::GENXDialect::getMaxWorkGroupSizeAttrName()) {
      if (!attrVal.dyn_cast<ArrayAttr>())
        return failure();

      attachMetadata("max_work_group_size");
    }

    // Set reqd_work_group_size metadata.
    if (attrName == GENX::GENXDialect::getReqdWorkGroupSizeAttrName()) {
      if (!attrVal.dyn_cast<ArrayAttr>())
        return failure();

      attachMetadata("reqd_work_group_size");
    }

    // Set intel_reqd_sub_group_size metadata.
    if (attrName == GENX::GENXDialect::getReqdSubGroupSizeAttrName()) {
      if (!attrVal.dyn_cast<ArrayAttr>())
        return failure();

      attachMetadata("intel_reqd_sub_group_size");
    }

    return success();
  }
};
} // namespace

void mlir::registerGENXDialectTranslation(DialectRegistry &registry) {
  registry.insert<GENX::GENXDialect>();
  registry.addExtension(+[](MLIRContext *ctx, GENX::GENXDialect *dialect) {
    dialect->addInterfaces<GENXDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerGENXDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerGENXDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
