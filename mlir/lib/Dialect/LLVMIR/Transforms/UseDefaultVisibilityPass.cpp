//===- UseDefaultVisibilityPass.cpp - Update default visibility -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/UseDefaultVisibilityPass.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMUSEDEFAULTVISIBILITYPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

static void updateVisibility(Operation *op,
                             LLVM::VisibilityAttr newVisibilityAttr) {
  static constexpr char visibilityAttrName[] = "visibility_";
  if (auto visibilityAttr =
          op->getAttrOfType<LLVM::VisibilityAttr>(visibilityAttrName)) {
    LLVM::Visibility visibility = visibilityAttr.getValue();
    if (visibility == LLVM::Visibility::Default) {
      op->setAttr(visibilityAttrName, newVisibilityAttr);
    }
  }
}

namespace {
class UseDefaultVisibilityPass
    : public LLVM::impl::LLVMUseDefaultVisibilityPassBase<
          UseDefaultVisibilityPass> {
  using Base::Base;

public:
  void runOnOperation() override {
    LLVM::Visibility useDefaultVisibility = useVisibility.getValue();
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    Dialect *llvmDialect = context->getLoadedDialect<LLVM::LLVMDialect>();
    auto newVisibilityAttr =
        LLVM::VisibilityAttr::get(context, useDefaultVisibility);
    op->walk([&](Operation *op) {
      if (op->getDialect() == llvmDialect)
        updateVisibility(op, newVisibilityAttr);
    });
  }
};
} // namespace
