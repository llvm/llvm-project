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

// Create a call to GENX Device function
// Currently this routine will work only for calling GENX functions that
// take a single int32 argument.
static llvm::Value *createDeviceFunctionCall(llvm::IRBuilderBase &builder,
                                             StringRef fnName, int parameter) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::FunctionType *functionType = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(module->getContext()), // return type.
      llvm::Type::getInt32Ty(module->getContext()), // parameter type.
      false);                                       // no variadic arguments.
  llvm::Function *fn = dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fnName, functionType).getCallee());
  fn->setCallingConv(llvm::CallingConv::SPIR_FUNC);
  llvm::Value *fnOp0 = llvm::ConstantInt::get(
      llvm::Type::getInt32Ty(module->getContext()), parameter);
  return builder.CreateCall(fn, ArrayRef<llvm::Value *>(fnOp0));
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

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return failure();

    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());

    // Set max_work_group_size metadata.
    if (attribute.getName() ==
        GENX::GENXDialect::getMaxWorkGroupSizeAttrName()) {
      llvmFunc->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
      auto value = attribute.getValue().dyn_cast<ArrayAttr>();
      if (!value)
        return failure();

      SmallVector<llvm::Metadata *, 3> metadata;
      llvm::Type *i64 = llvm::IntegerType::get(llvmContext, 64);
      for (int64_t i : extractFromI64ArrayAttr(attribute.getValue())) {
        llvm::Constant *constant = llvm::ConstantInt::get(i64, i);
        metadata.push_back(llvm::ConstantAsMetadata::get(constant));
      }
      llvm::MDNode *node = llvm::MDNode::get(llvmContext, metadata);
      llvmFunc->setMetadata("max_work_group_size", node);
    }

    // Set reqd_work_group_size metadata.
    if (attribute.getName() ==
        GENX::GENXDialect::getReqdWorkGroupSizeAttrName()) {
      llvmFunc->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
      auto value = attribute.getValue().dyn_cast<ArrayAttr>();
      if (!value)
        return failure();

      SmallVector<llvm::Metadata *, 3> metadata;
      llvm::Type *i64 = llvm::IntegerType::get(llvmContext, 64);
      for (int64_t i : extractFromI64ArrayAttr(attribute.getValue())) {
        llvm::Constant *constant = llvm::ConstantInt::get(i64, i);
        metadata.push_back(llvm::ConstantAsMetadata::get(constant));
      }
      llvm::MDNode *node = llvm::MDNode::get(llvmContext, metadata);
      llvmFunc->setMetadata("reqd_work_group_size", node);
    }

    // Set intel_reqd_sub_group_size metadata.
    if (attribute.getName() ==
        GENX::GENXDialect::getReqdSubGroupSizeAttrName()) {
      llvmFunc->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
      auto value = attribute.getValue().dyn_cast<ArrayAttr>();
      if (!value)
        return failure();

      SmallVector<llvm::Metadata *, 3> metadata;
      llvm::Type *i64 = llvm::IntegerType::get(llvmContext, 64);
      for (int64_t i : extractFromI64ArrayAttr(attribute.getValue())) {
        llvm::Constant *constant = llvm::ConstantInt::get(i64, i);
        metadata.push_back(llvm::ConstantAsMetadata::get(constant));
      }
      llvm::MDNode *node = llvm::MDNode::get(llvmContext, metadata);
      llvmFunc->setMetadata("intel_reqd_sub_group_size", node);
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