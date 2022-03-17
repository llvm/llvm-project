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
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

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
  amendOperation(Operation *op, NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    if (attribute.getName() == ROCDL::ROCDLDialect::getKernelFuncAttrName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return failure();

      // For GPU kernels,
      // 1. Insert AMDGPU_KERNEL calling convention.
      // 2. Insert amdgpu-flat-work-group-size(1, 256) attribute unless the user
      // has overriden this value - 256 is the default in clang
      // 3. Insert amdgpu-implicitarg-num-bytes=56 (which must be set on OpenCL
      // and HIP kernels per Clang)
      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvmFunc->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      if (!llvmFunc->hasFnAttribute("amdgpu-flat-work-group-size")) {
        llvmFunc->addFnAttr("amdgpu-flat-work-group-size", "1, 256");
      }
      llvmFunc->addFnAttr("amdgpu-implicitarg-num-bytes", "56");
    }
    // Override flat-work-group-size
    if ("rocdl.max_flat_work_group_size" == attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return failure();
      auto value = attribute.getValue().dyn_cast<IntegerAttr>();
      if (!value)
        return failure();

      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvm::SmallString<8> llvmAttrValue;
      llvm::raw_svector_ostream attrValueStream(llvmAttrValue);
      attrValueStream << "1, " << value.getInt();
      llvmFunc->addFnAttr("amdgpu-flat-work-group-size", llvmAttrValue);
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
