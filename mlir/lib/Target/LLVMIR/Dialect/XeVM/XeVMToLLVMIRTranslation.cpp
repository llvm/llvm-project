//===-- XeVMToLLVMIRTranslation.cpp - Translate XeVM to LLVM IR -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR XeVM dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"

#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the XeVM dialect to LLVM IR.
class XeVMDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    StringRef attrName = attribute.getName().getValue();
    if (attrName == mlir::xevm::XeVMDialect::getCacheControlsAttrName()) {
      auto cacheControlsArray = dyn_cast<ArrayAttr>(attribute.getValue());
      if (cacheControlsArray.size() != 2) {
        return op->emitOpError(
            "Expected both L1 and L3 cache control attributes!");
      }
      if (instructions.size() != 1) {
        return op->emitOpError("Expecting a single instruction");
      }
      return handleDecorationCacheControl(instructions.front(),
                                          cacheControlsArray.getValue());
    }
    return success();
  }

private:
  static LogicalResult handleDecorationCacheControl(llvm::Instruction *inst,
                                                    ArrayRef<Attribute> attrs) {
    SmallVector<llvm::Metadata *> decorations;
    llvm::LLVMContext &ctx = inst->getContext();
    llvm::Type *i32Ty = llvm::IntegerType::getInt32Ty(ctx);
    llvm::transform(
        attrs, std::back_inserter(decorations),
        [&ctx, i32Ty](Attribute attr) -> llvm::Metadata * {
          auto valuesArray = dyn_cast<ArrayAttr>(attr).getValue();
          std::array<llvm::Metadata *, 4> metadata;
          llvm::transform(
              valuesArray, metadata.begin(), [i32Ty](Attribute valueAttr) {
                return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    i32Ty, cast<IntegerAttr>(valueAttr).getValue()));
              });
          return llvm::MDNode::get(ctx, metadata);
        });
    constexpr llvm::StringLiteral decorationCacheControlMDName =
        "spirv.DecorationCacheControlINTEL";
    inst->setMetadata(decorationCacheControlMDName,
                      llvm::MDNode::get(ctx, decorations));
    return success();
  }
};
} // namespace

void mlir::registerXeVMDialectTranslation(::mlir::DialectRegistry &registry) {
  registry.insert<xevm::XeVMDialect>();
  registry.addExtension(+[](MLIRContext *ctx, xevm::XeVMDialect *dialect) {
    dialect->addInterfaces<XeVMDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerXeVMDialectTranslation(::mlir::MLIRContext &context) {
  DialectRegistry registry;
  registerXeVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
