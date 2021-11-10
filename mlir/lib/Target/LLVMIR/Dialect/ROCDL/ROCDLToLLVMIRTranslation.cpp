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

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

// Create a call to ROCm-Device-Library function
// Currently this routine will work only for calling ROCDL functions that
// take a single int32 argument. It is likely that the interface of this
// function will change to make it more generic.
static llvm::Value *createDeviceFunctionCall(llvm::IRBuilderBase &builder,
                                             StringRef fn_name, int parameter) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::FunctionType *function_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(module->getContext()), // return type.
      llvm::Type::getInt32Ty(module->getContext()), // parameter type.
      false);                                       // no variadic arguments.
  llvm::Function *fn = dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fn_name, function_type).getCallee());
  llvm::Value *fn_op0 = llvm::ConstantInt::get(
      llvm::Type::getInt32Ty(module->getContext()), parameter);
  return builder.CreateCall(fn, ArrayRef<llvm::Value *>(fn_op0));
}

/// Lowers a shuffle to the corresponding ROCDL op.
/// Ref:
/// https://github.com/ROCm-Developer-Tools/hipamd/blob/master/include/hip/amd_detail/amd_device_functions.h
/// ```
/// __device__
/// inline
/// int __shfl_xor(int var, int offset, int width = warpSize) {
///     int self = __lane_id();
///     int index = self^offset;
///     index = index >= ((self+width)&~(width-1))?self:index;
///     return __builtin_amdgcn_ds_bpermute(index<<2, var);
/// }```
///
/// Lowers a shuffle to the corresponding ROCDL op.
///
/// maskAndClamp (specifying the highest lane which participates in the
/// shuffle).
///
///     %one = llvm.constant(1 : i32) : i32
///     %two = llvm.constant(2 : i32) : i32
///     %width = llvm.add %mask_and_clamp, %one : i32
///     %self = @call __ockl_lane_u32() : i32
///     %index = llvm.xor %self, %offset : i32
///     %self_add = llvm.add %self, %width : i32
///     %bit_rvs_mask = llvm.not %mask_and_clamp: i32
///     %upper_bound = llvm.and %self_add, %bit_rvs_mask: i32
///     %cond_cmp = llvm.icmp %index, %upper_bound { predicate = 'sge' }: i1
///     %dst_index = llvm.select %cond_cmp, %self, %index : i32
///     %shl_index = llvm.shl %dst_index, %two : i32
///     @call __amdgcn_ds_bpermute(shl_index, %var)
///
static llvm::Value *createAMDGPUShflBfly(llvm::Value *value,
                                         llvm::Value *offset,
                                         llvm::Value *mask_and_clamp,
                                         llvm::IRBuilderBase &builder) {

  // CHECK_EQ(value->getType()->getPrimitiveSizeInBits(), 32);
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  auto valueTy = value->getType();
  auto int32Type = builder.getInt32Ty();

  auto function_type = llvm::FunctionType::get(int32Type, false);
  auto fn = dyn_cast<llvm::Function>(
      module->getOrInsertFunction("__ockl_lane_u32", function_type)
          .getCallee());
  auto self = builder.CreateCall(fn);

  auto one = builder.getInt32(1);
  auto two = builder.getInt32(2);
  auto width = builder.CreateAdd(mask_and_clamp, one);
  auto index = builder.CreateXor(self, offset);
  auto self_add = builder.CreateAdd(self, width);
  auto bitnot_mask = builder.CreateNot(mask_and_clamp);
  auto upper_bound = builder.CreateAnd(self_add, bitnot_mask);
  auto cond_cmp = builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SGE, index,
                                     upper_bound);
  auto dst_index = builder.CreateSelect(cond_cmp, self, index);
  auto shl_index = builder.CreateShl(dst_index, two);

  auto i32_value = builder.CreateBitCast(value, int32Type);

  auto function_type2 =
      llvm::FunctionType::get(int32Type, {int32Type, int32Type}, false);
  auto fn2 = dyn_cast<llvm::Function>(
      module->getOrInsertFunction("__amdgcn_ds_bpermute", function_type2)
          .getCallee());
  auto shfl_value = builder.CreateCall(fn2, {shl_index, i32_value});

  return builder.CreateBitCast(shfl_value, valueTy);
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
    if (attribute.first == ROCDL::ROCDLDialect::getKernelFuncAttrName()) {
      auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!func)
        return failure();

      // For GPU kernels,
      // 1. Insert AMDGPU_KERNEL calling convention.
      // 2. Insert amdgpu-flat-workgroup-size(1, 1024) attribute.
      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      llvmFunc->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      llvmFunc->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
    }
    return success();
  }
};
} // end namespace

void mlir::registerROCDLDialectTranslation(DialectRegistry &registry) {
  registry.insert<ROCDL::ROCDLDialect>();
  registry.addDialectInterface<ROCDL::ROCDLDialect,
                               ROCDLDialectLLVMIRTranslationInterface>();
}

void mlir::registerROCDLDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerROCDLDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
