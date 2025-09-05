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

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

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
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());

    if (attribute.getName() == NVVM::NVVMDialect::getMaxntidAttrName()) {
      if (!isa<DenseI32ArrayAttr>(attribute.getValue()))
        return failure();
      auto values = cast<DenseI32ArrayAttr>(attribute.getValue());
      const std::string attr = llvm::formatv(
          "{0:$[,]}", llvm::make_range(values.asArrayRef().begin(),
                                       values.asArrayRef().end()));
      llvmFunc->addFnAttr("nvvm.maxntid", attr);
    } else if (attribute.getName() == NVVM::NVVMDialect::getReqntidAttrName()) {
      if (!isa<DenseI32ArrayAttr>(attribute.getValue()))
        return failure();
      auto values = cast<DenseI32ArrayAttr>(attribute.getValue());
      const std::string attr = llvm::formatv(
          "{0:$[,]}", llvm::make_range(values.asArrayRef().begin(),
                                       values.asArrayRef().end()));
      llvmFunc->addFnAttr("nvvm.reqntid", attr);
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getClusterDimAttrName()) {
      if (!isa<DenseI32ArrayAttr>(attribute.getValue()))
        return failure();
      auto values = cast<DenseI32ArrayAttr>(attribute.getValue());
      const std::string attr = llvm::formatv(
          "{0:$[,]}", llvm::make_range(values.asArrayRef().begin(),
                                       values.asArrayRef().end()));
      llvmFunc->addFnAttr("nvvm.cluster_dim", attr);
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
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getBlocksAreClustersAttrName()) {
      llvmFunc->addFnAttr("nvvm.blocksareclusters");
    }

    return success();
  }

  LogicalResult
  convertParameterAttr(LLVMFuncOp funcOp, int argIdx, NamedAttribute attribute,
                       LLVM::ModuleTranslation &moduleTranslation) const final {

    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc =
        moduleTranslation.lookupFunction(funcOp.getName());

    if (attribute.getName() == NVVM::NVVMDialect::getGridConstantAttrName()) {
      llvmFunc->addParamAttr(
          argIdx, llvm::Attribute::get(llvmContext, "nvvm.grid_constant"));
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
