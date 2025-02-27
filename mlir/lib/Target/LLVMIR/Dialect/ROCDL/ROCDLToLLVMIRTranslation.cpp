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

#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

// Create a call to ROCm-Device-Library function that returns an ID.
// This is intended to specifically call device functions that fetch things like
// block or grid dimensions, and so is limited to functions that take one
// integer parameter.
static llvm::Value *createDimGetterFunctionCall(llvm::IRBuilderBase &builder,
                                                Operation *op, StringRef fnName,
                                                int parameter) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::FunctionType *functionType = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(module->getContext()), // return type.
      llvm::Type::getInt32Ty(module->getContext()), // parameter type.
      false);                                       // no variadic arguments.
  llvm::Function *fn = dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fnName, functionType).getCallee());
  llvm::Value *fnOp0 = llvm::ConstantInt::get(
      llvm::Type::getInt32Ty(module->getContext()), parameter);
  auto *call = builder.CreateCall(fn, ArrayRef<llvm::Value *>(fnOp0));
  if (auto rangeAttr = op->getAttrOfType<LLVM::ConstantRangeAttr>("range")) {
    // Zero-extend to 64 bits because the GPU dialect uses 32-bit bounds but
    // these ockl functions are defined to be 64-bits
    call->addRangeRetAttr(llvm::ConstantRange(rangeAttr.getLower().zext(64),
                                              rangeAttr.getUpper().zext(64)));
  }
  return call;
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
    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    if (dialect->getKernelAttrHelper().getName() == attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return op->emitOpError(Twine(attribute.getName()) +
                               " is only supported on `llvm.func` operations");
      ;

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

      // MLIR's GPU kernel APIs all assume and produce uniformly-sized
      // workgroups, so the lowering of the `rocdl.kernel` marker encodes this
      // assumption. This assumption may be overridden by setting
      // `rocdl.uniform_work_group_size` on a given function.
      if (!llvmFunc->hasFnAttribute("uniform-work-group-size"))
        llvmFunc->addFnAttr("uniform-work-group-size", "true");
    }
    // Override flat-work-group-size
    // TODO: update clients to rocdl.flat_work_group_size instead,
    // then remove this half of the branch
    if (dialect->getMaxFlatWorkGroupSizeAttrHelper().getName() ==
        attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return op->emitOpError(Twine(attribute.getName()) +
                               " is only supported on `llvm.func` operations");
      auto value = dyn_cast<IntegerAttr>(attribute.getValue());
      if (!value)
        return op->emitOpError(Twine(attribute.getName()) +
                               " must be an integer");

      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvm::SmallString<8> llvmAttrValue;
      llvm::raw_svector_ostream attrValueStream(llvmAttrValue);
      attrValueStream << "1," << value.getInt();
      llvmFunc->addFnAttr("amdgpu-flat-work-group-size", llvmAttrValue);
    }
    if (dialect->getWavesPerEuAttrHelper().getName() == attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return op->emitOpError(Twine(attribute.getName()) +
                               " is only supported on `llvm.func` operations");
      auto value = dyn_cast<IntegerAttr>(attribute.getValue());
      if (!value)
        return op->emitOpError(Twine(attribute.getName()) +
                               " must be an integer");

      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvm::SmallString<8> llvmAttrValue;
      llvm::raw_svector_ostream attrValueStream(llvmAttrValue);
      attrValueStream << value.getInt();
      llvmFunc->addFnAttr("amdgpu-waves-per-eu", llvmAttrValue);
    }
    if (dialect->getFlatWorkGroupSizeAttrHelper().getName() ==
        attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return op->emitOpError(Twine(attribute.getName()) +
                               " is only supported on `llvm.func` operations");
      auto value = dyn_cast<StringAttr>(attribute.getValue());
      if (!value)
        return op->emitOpError(Twine(attribute.getName()) +
                               " must be a string");

      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvm::SmallString<8> llvmAttrValue;
      llvmAttrValue.append(value.getValue());
      llvmFunc->addFnAttr("amdgpu-flat-work-group-size", llvmAttrValue);
    }
    if (ROCDL::ROCDLDialect::getUniformWorkGroupSizeAttrName() ==
        attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return op->emitOpError(Twine(attribute.getName()) +
                               " is only supported on `llvm.func` operations");
      auto value = dyn_cast<BoolAttr>(attribute.getValue());
      if (!value)
        return op->emitOpError(Twine(attribute.getName()) +
                               " must be a boolean");
      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvmFunc->addFnAttr("uniform-work-group-size",
                          value.getValue() ? "true" : "false");
    }
    if (dialect->getUnsafeFpAtomicsAttrHelper().getName() ==
        attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return op->emitOpError(Twine(attribute.getName()) +
                               " is only supported on `llvm.func` operations");
      auto value = dyn_cast<BoolAttr>(attribute.getValue());
      if (!value)
        return op->emitOpError(Twine(attribute.getName()) +
                               " must be a boolean");
      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvmFunc->addFnAttr("amdgpu-unsafe-fp-atomics",
                          value.getValue() ? "true" : "false");
    }
    // Set reqd_work_group_size metadata
    if (dialect->getReqdWorkGroupSizeAttrHelper().getName() ==
        attribute.getName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return op->emitOpError(Twine(attribute.getName()) +
                               " is only supported on `llvm.func` operations");
      auto value = dyn_cast<DenseI32ArrayAttr>(attribute.getValue());
      if (!value)
        return op->emitOpError(Twine(attribute.getName()) +
                               " must be a dense i32 array attribute");
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

    // Atomic and nontemporal metadata
    if (dialect->getLastUseAttrHelper().getName() == attribute.getName()) {
      for (llvm::Instruction *i : instructions)
        i->setMetadata("amdgpu.last.use", llvm::MDNode::get(llvmContext, {}));
    }
    if (dialect->getNoRemoteMemoryAttrHelper().getName() ==
        attribute.getName()) {
      for (llvm::Instruction *i : instructions)
        i->setMetadata("amdgpu.no.remote.memory",
                       llvm::MDNode::get(llvmContext, {}));
    }
    if (dialect->getNoFineGrainedMemoryAttrHelper().getName() ==
        attribute.getName()) {
      for (llvm::Instruction *i : instructions)
        i->setMetadata("amdgpu.no.fine.grained.memory",
                       llvm::MDNode::get(llvmContext, {}));
    }
    if (dialect->getIgnoreDenormalModeAttrHelper().getName() ==
        attribute.getName()) {
      for (llvm::Instruction *i : instructions)
        i->setMetadata("amdgpu.ignore.denormal.mode",
                       llvm::MDNode::get(llvmContext, {}));
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
