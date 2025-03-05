//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to link llvm dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Linker/LinkerInterface.h"

using namespace mlir;

struct LLVMLinkerInterface : public link::LinkerInterface {
  using LinkerInterface::LinkerInterface;

  bool isDeclaration(GlobalValueLinkageOpInterface op) const final {
    if (auto func = dyn_cast<LLVM::LLVMFuncOp>(op.getOperation()))
      return isDeclaration(func);
    if (auto global = dyn_cast<LLVM::GlobalOp>(op.getOperation()))
      return isDeclaration(global);
    return false;
  }

  bool isDeclaration(LLVM::LLVMFuncOp op) const { return op.getBody().empty(); }

  bool isDeclaration(LLVM::GlobalOp op) const {
    return op.getInitializerRegion().empty() && !op.getValue();
  }
};

void mlir::LLVM::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMLinkerInterface>();
  });
}
