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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

static llvm::Intrinsic::ID getReduxIntrinsicId(llvm::Type *resultType,
                                               NVVM::ReduxKind kind) {
  if (!resultType->isIntegerTy(32))
    llvm_unreachable("unsupported data type for redux");

  switch (kind) {
  case NVVM::ReduxKind::ADD:
    return llvm::Intrinsic::nvvm_redux_sync_add;
  case NVVM::ReduxKind::UMAX:
    return llvm::Intrinsic::nvvm_redux_sync_umax;
  case NVVM::ReduxKind::UMIN:
    return llvm::Intrinsic::nvvm_redux_sync_umin;
  case NVVM::ReduxKind::AND:
    return llvm::Intrinsic::nvvm_redux_sync_and;
  case NVVM::ReduxKind::OR:
    return llvm::Intrinsic::nvvm_redux_sync_or;
  case NVVM::ReduxKind::XOR:
    return llvm::Intrinsic::nvvm_redux_sync_xor;
  case NVVM::ReduxKind::MAX:
    return llvm::Intrinsic::nvvm_redux_sync_max;
  case NVVM::ReduxKind::MIN:
    return llvm::Intrinsic::nvvm_redux_sync_min;
  }
  llvm_unreachable("unknown redux kind");
}

static llvm::Intrinsic::ID getShflIntrinsicId(llvm::Type *resultType,
                                              NVVM::ShflKind kind,
                                              bool withPredicate) {

  if (withPredicate) {
    resultType = cast<llvm::StructType>(resultType)->getElementType(0);
    switch (kind) {
    case NVVM::ShflKind::bfly:
      return resultType->isFloatTy()
                 ? llvm::Intrinsic::nvvm_shfl_sync_bfly_f32p
                 : llvm::Intrinsic::nvvm_shfl_sync_bfly_i32p;
    case NVVM::ShflKind::up:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_up_f32p
                                     : llvm::Intrinsic::nvvm_shfl_sync_up_i32p;
    case NVVM::ShflKind::down:
      return resultType->isFloatTy()
                 ? llvm::Intrinsic::nvvm_shfl_sync_down_f32p
                 : llvm::Intrinsic::nvvm_shfl_sync_down_i32p;
    case NVVM::ShflKind::idx:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_idx_f32p
                                     : llvm::Intrinsic::nvvm_shfl_sync_idx_i32p;
    }
  } else {
    switch (kind) {
    case NVVM::ShflKind::bfly:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_bfly_f32
                                     : llvm::Intrinsic::nvvm_shfl_sync_bfly_i32;
    case NVVM::ShflKind::up:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_up_f32
                                     : llvm::Intrinsic::nvvm_shfl_sync_up_i32;
    case NVVM::ShflKind::down:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_down_f32
                                     : llvm::Intrinsic::nvvm_shfl_sync_down_i32;
    case NVVM::ShflKind::idx:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_idx_f32
                                     : llvm::Intrinsic::nvvm_shfl_sync_idx_i32;
    }
  }
  llvm_unreachable("unknown shuffle kind");
}

/// Return the intrinsic ID associated with ldmatrix for the given paramters.
static llvm::Intrinsic::ID getLdMatrixIntrinsicId(NVVM::MMALayout layout,
                                                  int32_t num) {
  if (layout == NVVM::MMALayout::row) {
    switch (num) {
    case 1:
      return llvm::Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x1_b16;
    case 2:
      return llvm::Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x2_b16;
    case 4:
      return llvm::Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_b16;
    default:
      llvm_unreachable("unsupported number of matrix");
    }

  } else {
    switch (num) {
    case 1:
      return llvm::Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x1_trans_b16;
    case 2:
      return llvm::Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x2_trans_b16;
    case 4:
      return llvm::Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_trans_b16;
    default:
      llvm_unreachable("unsupported number of matrix");
    }
  }
}

static unsigned getUnidirectionalFenceProxyID(NVVM::ProxyKind fromProxy,
                                              NVVM::ProxyKind toProxy,
                                              NVVM::MemScopeKind scope,
                                              bool isRelease) {
  if (fromProxy == NVVM::ProxyKind::GENERIC &&
      toProxy == NVVM::ProxyKind::TENSORMAP) {
    switch (scope) {
    case NVVM::MemScopeKind::CTA: {
      if (isRelease)
        return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_release_cta;
      return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_acquire_cta;
    }
    case NVVM::MemScopeKind::CLUSTER: {
      if (isRelease)
        return llvm::Intrinsic::
            nvvm_fence_proxy_tensormap_generic_release_cluster;
      return llvm::Intrinsic::
          nvvm_fence_proxy_tensormap_generic_acquire_cluster;
    }
    case NVVM::MemScopeKind::GPU: {
      if (isRelease)
        return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_release_gpu;
      return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_acquire_gpu;
    }
    case NVVM::MemScopeKind::SYS: {
      if (isRelease)
        return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_release_sys;
      return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_acquire_sys;
    }
    }
    llvm_unreachable("Unknown scope for uni-directional fence.proxy operation");
  }
  llvm_unreachable("Unsupported proxy kinds");
}

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the NVVM dialect to LLVM IR.
class NVVMDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/NVVMConversions.inc"

    return failure();
  }

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return failure();
    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());

    auto generateMetadata = [&](int dim, StringRef name) {
      llvm::Metadata *llvmMetadata[] = {
          llvm::ValueAsMetadata::get(llvmFunc),
          llvm::MDString::get(llvmContext, name),
          llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
              llvm::Type::getInt32Ty(llvmContext), dim))};
      llvm::MDNode *llvmMetadataNode =
          llvm::MDNode::get(llvmContext, llvmMetadata);
      moduleTranslation.getOrInsertNamedModuleMetadata("nvvm.annotations")
          ->addOperand(llvmMetadataNode);
    };
    if (attribute.getName() == NVVM::NVVMDialect::getMaxntidAttrName()) {
      if (!dyn_cast<DenseI32ArrayAttr>(attribute.getValue()))
        return failure();
      auto values = cast<DenseI32ArrayAttr>(attribute.getValue());
      generateMetadata(values[0], NVVM::NVVMDialect::getMaxntidXName());
      if (values.size() > 1)
        generateMetadata(values[1], NVVM::NVVMDialect::getMaxntidYName());
      if (values.size() > 2)
        generateMetadata(values[2], NVVM::NVVMDialect::getMaxntidZName());
    } else if (attribute.getName() == NVVM::NVVMDialect::getReqntidAttrName()) {
      if (!dyn_cast<DenseI32ArrayAttr>(attribute.getValue()))
        return failure();
      auto values = cast<DenseI32ArrayAttr>(attribute.getValue());
      generateMetadata(values[0], NVVM::NVVMDialect::getReqntidXName());
      if (values.size() > 1)
        generateMetadata(values[1], NVVM::NVVMDialect::getReqntidYName());
      if (values.size() > 2)
        generateMetadata(values[2], NVVM::NVVMDialect::getReqntidZName());
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getClusterDimAttrName()) {
      if (!dyn_cast<DenseI32ArrayAttr>(attribute.getValue()))
        return failure();
      auto values = cast<DenseI32ArrayAttr>(attribute.getValue());
      generateMetadata(values[0], NVVM::NVVMDialect::getClusterDimXName());
      if (values.size() > 1)
        generateMetadata(values[1], NVVM::NVVMDialect::getClusterDimYName());
      if (values.size() > 2)
        generateMetadata(values[2], NVVM::NVVMDialect::getClusterDimZName());
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getClusterMaxBlocksAttrName()) {
      auto value = dyn_cast<IntegerAttr>(attribute.getValue());
      llvmFunc->addFnAttr("nvvm.maxclusterrank", llvm::utostr(value.getInt()));
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getMinctasmAttrName()) {
      auto value = dyn_cast<IntegerAttr>(attribute.getValue());
      llvmFunc->addFnAttr("nvvm.minctasm", llvm::utostr(value.getInt()));
    } else if (attribute.getName() == NVVM::NVVMDialect::getMaxnregAttrName()) {
      auto value = dyn_cast<IntegerAttr>(attribute.getValue());
      llvmFunc->addFnAttr("nvvm.maxnreg", llvm::utostr(value.getInt()));
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getKernelFuncAttrName()) {
      llvmFunc->setCallingConv(llvm::CallingConv::PTX_Kernel);
    }
    return success();
  }

  LogicalResult
  convertParameterAttr(LLVMFuncOp funcOp, int argIdx, NamedAttribute attribute,
                       LLVM::ModuleTranslation &moduleTranslation) const final {

    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc =
        moduleTranslation.lookupFunction(funcOp.getName());
    llvm::NamedMDNode *nvvmAnnotations =
        moduleTranslation.getOrInsertNamedModuleMetadata("nvvm.annotations");

    if (attribute.getName() == NVVM::NVVMDialect::getGridConstantAttrName()) {
      llvm::MDNode *gridConstantMetaData = nullptr;

      // Check if a 'grid_constant' metadata node exists for the given function
      for (llvm::MDNode *opnd : llvm::reverse(nvvmAnnotations->operands())) {
        if (opnd->getNumOperands() == 3 &&
            opnd->getOperand(0) == llvm::ValueAsMetadata::get(llvmFunc) &&
            opnd->getOperand(1) ==
                llvm::MDString::get(llvmContext, "grid_constant")) {
          gridConstantMetaData = opnd;
          break;
        }
      }

      // 'grid_constant' is a function-level meta data node with a list of
      // integers, where each integer n denotes that the nth parameter has the
      // grid_constant annotation (numbering from 1). This requires aggregating
      // the indices of the individual parameters that have this attribute.
      llvm::Type *i32 = llvm::IntegerType::get(llvmContext, 32);
      if (gridConstantMetaData == nullptr) {
        // Create a new 'grid_constant' metadata node
        SmallVector<llvm::Metadata *> gridConstMetadata = {
            llvm::ValueAsMetadata::getConstant(
                llvm::ConstantInt::get(i32, argIdx + 1))};
        llvm::Metadata *llvmMetadata[] = {
            llvm::ValueAsMetadata::get(llvmFunc),
            llvm::MDString::get(llvmContext, "grid_constant"),
            llvm::MDNode::get(llvmContext, gridConstMetadata)};
        llvm::MDNode *llvmMetadataNode =
            llvm::MDNode::get(llvmContext, llvmMetadata);
        nvvmAnnotations->addOperand(llvmMetadataNode);
      } else {
        // Append argIdx + 1 to the 'grid_constant' argument list
        if (auto argList =
                dyn_cast<llvm::MDTuple>(gridConstantMetaData->getOperand(2))) {
          llvm::TempMDTuple clonedArgList = argList->clone();
          clonedArgList->push_back((llvm::ValueAsMetadata::getConstant(
              llvm::ConstantInt::get(i32, argIdx + 1))));
          gridConstantMetaData->replaceOperandWith(
              2, llvm::MDNode::replaceWithUniqued(std::move(clonedArgList)));
        }
      }
    }
    return success();
  }
};
} // namespace

void mlir::registerNVVMDialectTranslation(DialectRegistry &registry) {
  registry.insert<NVVM::NVVMDialect>();
  registry.addExtension(+[](MLIRContext *ctx, NVVM::NVVMDialect *dialect) {
    dialect->addInterfaces<NVVMDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerNVVMDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerNVVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
