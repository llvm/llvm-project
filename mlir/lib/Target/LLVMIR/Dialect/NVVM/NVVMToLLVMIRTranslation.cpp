//===- NVVMToLLVMIRTranslation.cpp - Translate NVVM to LLVM IR ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR NVVM dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

static llvm::Intrinsic::ID getShflBflyIntrinsicId(llvm::Type *resultType,
                                                  bool withPredicate) {
  if (withPredicate) {
    resultType = cast<llvm::StructType>(resultType)->getElementType(0);
    return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_bfly_f32p
                                   : llvm::Intrinsic::nvvm_shfl_sync_bfly_i32p;
  }
  return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_bfly_f32
                                 : llvm::Intrinsic::nvvm_shfl_sync_bfly_i32;
}

LogicalResult mlir::NVVMDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {
  Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/NVVMConversions.inc"

  return failure();
}

LogicalResult mlir::NVVMDialectLLVMIRTranslationInterface::amendOperation(
    Operation *op, NamedAttribute attribute,
    LLVM::ModuleTranslation &moduleTranslation) const {
  if (attribute.first == NVVM::NVVMDialect::getKernelFuncAttrName()) {
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return failure();

    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());
    llvm::Metadata *llvmMetadata[] = {
        llvm::ValueAsMetadata::get(llvmFunc),
        llvm::MDString::get(llvmContext, "kernel"),
        llvm::ValueAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext), 1))};
    llvm::MDNode *llvmMetadataNode =
        llvm::MDNode::get(llvmContext, llvmMetadata);
    moduleTranslation.getOrInsertNamedModuleMetadata("nvvm.annotations")
        ->addOperand(llvmMetadataNode);
  }
  return success();
}
