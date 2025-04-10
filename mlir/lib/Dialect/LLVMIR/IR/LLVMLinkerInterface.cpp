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
#include "mlir/Linker/LLVMLinkerMixin.h"
#include "mlir/Linker/LinkerInterface.h"

using namespace mlir;
using namespace mlir::link;

//===----------------------------------------------------------------------===//
// LLVMSymbolLinkerInterface
//===----------------------------------------------------------------------===//

class LLVMSymbolLinkerInterface
    : public SymbolAttrLLVMLinkerInterface<LLVMSymbolLinkerInterface> {
public:
  LLVMSymbolLinkerInterface(Dialect *dialect)
      : SymbolAttrLLVMLinkerInterface(dialect) {}

  bool canBeLinked(Operation *op) const override {
    return isa<LLVM::GlobalOp>(op) || isa<LLVM::LLVMFuncOp>(op);
  }

  //===--------------------------------------------------------------------===//
  // LLVMLinkerMixin required methods from derived linker interface
  //===--------------------------------------------------------------------===//

  static Linkage getLinkage(Operation *op) {
    if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
      return gv.getLinkage();
    if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
      return fn.getLinkage();
    llvm_unreachable("unexpected operation");
  }

  static Visibility getVisibility(Operation *op) {
    if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
      return gv.getVisibility_();
    if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
      return fn.getVisibility_();
    llvm_unreachable("unexpected operation");
  }

  static void setVisibility(Operation *op, Visibility visibility) {
    if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
      return gv.setVisibility_(visibility);
    if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
      return fn.setVisibility_(visibility);
    llvm_unreachable("unexpected operation");
  }

  // Return true if the primary definition of this global value is outside of
  // the current translation unit.
  static bool isDeclaration(Operation *op) {
    if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
      return gv.getInitializerRegion().empty() && !gv.getValue();
    if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
      return fn.getBody().empty();
    llvm_unreachable("unexpected operation");
  }

  static unsigned getBitWidth(Operation *op) {
    if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
      return gv.getType().getIntOrFloatBitWidth();
    llvm_unreachable("unexpected operation");
  }
};

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::LLVM::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMSymbolLinkerInterface>();
  });
}
