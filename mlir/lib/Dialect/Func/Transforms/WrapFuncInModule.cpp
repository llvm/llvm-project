//===- WrapFuncInModule.cpp - Wrap functions in nested modules ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace func {
#define GEN_PASS_DEF_WRAPFUNCINMODULEPASS
#include "mlir/Dialect/Func/Transforms/Passes.h.inc"
} // namespace func

namespace {
struct WrapFuncInModulePass
    : public func::impl::WrapFuncInModulePassBase<WrapFuncInModulePass> {
  using WrapFuncInModulePassBase::WrapFuncInModulePassBase;

  void runOnOperation() override {
    ModuleOp parentModule = getOperation();

    // Collect all top-level func ops.
    SmallVector<func::FuncOp> funcOps;
    for (auto funcOp : parentModule.getOps<func::FuncOp>()) {
      funcOps.push_back(funcOp);
    }

    for (func::FuncOp funcOp : funcOps) {
      OpBuilder builder(funcOp);
      ModuleOp nestedModule = ModuleOp::create(builder, funcOp.getLoc());

      // Move the funcOp into the body of the nestedModule.
      funcOp->moveBefore(nestedModule.getBody(), nestedModule.getBody()->end());
    }
  }
};
} // namespace
} // namespace mlir
