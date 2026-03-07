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

  /// Empty amendOperation implementation, can be extended in the future.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    return success();
  }

private:
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
