//===- ROCDLToLLVMIRTranslation.cpp - Translate ROCDL to LLVM IR ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR ROCDL dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

static llvm::Value *createIntrinsicCallWithRange(llvm::IRBuilderBase &builder,
                                                 llvm::Intrinsic::ID intrinsic,
                                                 DenseI32ArrayAttr maybeRange) {
  auto *inst = llvm::cast<llvm::CallInst>(
      createIntrinsicCall(builder, intrinsic, {}, {}));
  if (maybeRange) {
    SmallVector<llvm::APInt, 2> apInts;
    for (int32_t i : maybeRange.asArrayRef())
      apInts.push_back(llvm::APInt(32, i));
    llvm::MDBuilder mdBuilder(builder.getContext());
    llvm::MDNode *range = mdBuilder.createRange(apInts[0], apInts[1]);
    inst->setMetadata(llvm::LLVMContext::MD_range, range);
  }
  return inst;
}

// Create a call to ROCm-Device-Library function
// Currently this routine will work only for calling ROCDL functions that
// take a single int32 argument. It is likely that the interface of this
// function will change to make it more generic.
static llvm::Value *createDeviceFunctionCall(llvm::IRBuilderBase &builder,
                                             StringRef fnName, int parameter) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::FunctionType *functionType = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(module->getContext()), // return type.
      llvm::Type::getInt32Ty(module->getContext()), // parameter type.
      false);                                       // no variadic arguments.
  llvm::Function *fn = dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fnName, functionType).getCallee());
  llvm::Value *fnOp0 = llvm::ConstantInt::get(
      llvm::Type::getInt32Ty(module->getContext()), parameter);
  return builder.CreateCall(fn, ArrayRef<llvm::Value *>(fnOp0));
}

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the ROCDL dialect to LLVM IR.
class ROCDLDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/ROCDLConversions.inc"

    return failure();
  }

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto *dialect = dyn_cast<ROCDL::ROCDLDialect>(attribute.getNameDialect());
    if (dialect->getKernelAttrHelper().getName() == attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return failure();

      // For GPU kernels,
      // 1. Insert AMDGPU_KERNEL calling convention.
      // 2. Insert amdgpu-flat-work-group-size(1, 256) attribute unless the user
      // has overriden this value - 256 is the default in clang
      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvmFunc->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      if (!llvmFunc->hasFnAttribute("amdgpu-flat-work-group-size")) {
        llvmFunc->addFnAttr("amdgpu-flat-work-group-size", "1,256");
      }
    }
    // Override flat-work-group-size
    // TODO: update clients to rocdl.flat_work_group_size instead,
    // then remove this half of the branch
    if (dialect->getMaxFlatWorkGroupSizeAttrHelper().getName() ==
        attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return failure();
      auto value = dyn_cast<IntegerAttr>(attribute.getValue());
      if (!value)
        return failure();

      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvm::SmallString<8> llvmAttrValue;
      llvm::raw_svector_ostream attrValueStream(llvmAttrValue);
      attrValueStream << "1," << value.getInt();
      llvmFunc->addFnAttr("amdgpu-flat-work-group-size", llvmAttrValue);
    }
    if (dialect->getFlatWorkGroupSizeAttrHelper().getName() ==
        attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return failure();
      auto value = dyn_cast<StringAttr>(attribute.getValue());
      if (!value)
        return failure();

      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvm::SmallString<8> llvmAttrValue;
      llvmAttrValue.append(value.getValue());
      llvmFunc->addFnAttr("amdgpu-flat-work-group-size", llvmAttrValue);
    }

    // Set reqd_work_group_size metadata
    if (dialect->getReqdWorkGroupSizeAttrHelper().getName() ==
        attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return failure();
      auto value = dyn_cast<DenseI32ArrayAttr>(attribute.getValue());
      if (!value)
        return failure();
      llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
      SmallVector<llvm::Metadata *, 3> metadata;
      llvm::Type *i32 = llvm::IntegerType::get(llvmContext, 32);
      for (int32_t i : value.asArrayRef()) {
        llvm::Constant *constant = llvm::ConstantInt::get(i32, i);
        metadata.push_back(llvm::ConstantAsMetadata::get(constant));
      }
      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvm::MDNode *node = llvm::MDNode::get(llvmContext, metadata);
      llvmFunc->setMetadata("reqd_work_group_size", node);
    }
    return success();
  }
};
} // namespace

void mlir::registerROCDLDialectTranslation(DialectRegistry &registry) {
  registry.insert<ROCDL::ROCDLDialect>();
  registry.addExtension(+[](MLIRContext *ctx, ROCDL::ROCDLDialect *dialect) {
    dialect->addInterfaces<ROCDLDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerROCDLDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerROCDLDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
