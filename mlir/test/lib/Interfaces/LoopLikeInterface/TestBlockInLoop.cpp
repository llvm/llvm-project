//===- TestBlockInLoop.cpp - Pass to test mlir::blockIsInLoop -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
/// This is a test pass that tests Blocks's isInLoop method by checking if each
/// block in a function is in a loop and outputing if it is
struct IsInLoopPass
    : public PassWrapper<IsInLoopPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IsInLoopPass)

  StringRef getArgument() const final { return "test-block-is-in-loop"; }
  StringRef getDescription() const final {
    return "Test mlir::blockIsInLoop()";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    func.walk([](mlir::Block *block) {
      llvm::outs() << "Block is ";
      if (LoopLikeOpInterface::blockIsInLoop(block))
        llvm::outs() << "in a loop\n";
      else
        llvm::outs() << "not in a loop\n";
      block->print(llvm::outs());
      llvm::outs() << "\n";
    });
  }
};

} // namespace

namespace mlir {
void registerLoopLikeInterfaceTestPasses() { PassRegistration<IsInLoopPass>(); }
} // namespace mlir
