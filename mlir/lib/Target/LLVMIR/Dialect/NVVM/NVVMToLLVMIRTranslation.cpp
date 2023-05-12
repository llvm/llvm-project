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
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

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
  amendOperation(Operation *op, NamedAttribute attribute,
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
      if (!dyn_cast<ArrayAttr>(attribute.getValue()))
        return failure();
      SmallVector<int64_t> values =
          extractFromI64ArrayAttr(attribute.getValue());
      generateMetadata(values[0], NVVM::NVVMDialect::getMaxntidXName());
      if (values.size() > 1)
        generateMetadata(values[1], NVVM::NVVMDialect::getMaxntidYName());
      if (values.size() > 2)
        generateMetadata(values[2], NVVM::NVVMDialect::getMaxntidZName());
    } else if (attribute.getName() == NVVM::NVVMDialect::getReqntidAttrName()) {
      if (!dyn_cast<ArrayAttr>(attribute.getValue()))
        return failure();
      SmallVector<int64_t> values =
          extractFromI64ArrayAttr(attribute.getValue());
      generateMetadata(values[0], NVVM::NVVMDialect::getReqntidXName());
      if (values.size() > 1)
        generateMetadata(values[1], NVVM::NVVMDialect::getReqntidYName());
      if (values.size() > 2)
        generateMetadata(values[2], NVVM::NVVMDialect::getReqntidZName());
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getMinctasmAttrName()) {
      auto value = dyn_cast<IntegerAttr>(attribute.getValue());
      generateMetadata(value.getInt(), "minctasm");
    } else if (attribute.getName() == NVVM::NVVMDialect::getMaxnregAttrName()) {
      auto value = dyn_cast<IntegerAttr>(attribute.getValue());
      generateMetadata(value.getInt(), "maxnreg");
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getKernelFuncAttrName()) {
      llvm::Metadata *llvmMetadataKernel[] = {
          llvm::ValueAsMetadata::get(llvmFunc),
          llvm::MDString::get(llvmContext, "kernel"),
          llvm::ValueAsMetadata::get(
              llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext), 1))};
      llvm::MDNode *llvmMetadataNode =
          llvm::MDNode::get(llvmContext, llvmMetadataKernel);
      moduleTranslation.getOrInsertNamedModuleMetadata("nvvm.annotations")
          ->addOperand(llvmMetadataNode);
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
