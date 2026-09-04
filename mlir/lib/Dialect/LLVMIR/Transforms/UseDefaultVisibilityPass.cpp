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
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMUSEDEFAULTVISIBILITYPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

namespace {
class UseDefaultVisibilityPass
    : public LLVM::impl::LLVMUseDefaultVisibilityPassBase<
          UseDefaultVisibilityPass> {
  using Base::Base;

public:
  void runOnOperation() override {
    LLVM::Visibility useDefaultVisibility = useVisibility.getValue();
    if (useDefaultVisibility == LLVM::Visibility::Default)
      return;
    Operation *op = getOperation();
    op->walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *, void>(op)
          .Case<LLVM::LLVMFuncOp, LLVM::GlobalOp, LLVM::IFuncOp, LLVM::AliasOp>(
              [&](auto op) {
                if (op.getVisibility_() == LLVM::Visibility::Default)
                  op.setVisibility_(useDefaultVisibility);
              });
    });
  }
};
} // namespace
