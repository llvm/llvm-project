//===- UseDefaultVisibilityPass.cpp - Update default visibility ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/UseDefaultVisibilityPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMUSEDEFAULTVISIBILITYPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

template <typename OpT>
static void updateVisibility(OpT op, LLVM::Visibility useDefaultVisibility) {
  LLVM::Visibility vis = op.getVisibility_();
  if (vis == LLVM::Visibility::Default) {
    op.setVisibility_(useDefaultVisibility);
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
    getOperation()->walk([&](Operation *op) {
      if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        updateVisibility(funcOp, useDefaultVisibility);
      } else if (auto globalOp = dyn_cast<LLVM::GlobalOp>(op)) {
        updateVisibility(globalOp, useDefaultVisibility);
      }
    });
  }
};
} // namespace
