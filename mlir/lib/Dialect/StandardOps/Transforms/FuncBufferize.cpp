//===- Bufferize.cpp - Bufferization for std ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of std.func's and std.call's.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct FuncBufferizePass : public FuncBufferizeBase<FuncBufferizePass> {
  using FuncBufferizeBase<FuncBufferizePass>::FuncBufferizeBase;

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    BufferizeTypeConverter typeConverter;
    OwningRewritePatternList patterns;
    ConversionTarget target(*context);

    populateFuncOpTypeConversionPattern(patterns, context, typeConverter);
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, context, typeConverter);
    target.addDynamicallyLegalOp<CallOp>(
        [&](CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceAndReturnOpTypeConversionPattern(patterns, context,
                                                              typeConverter);
    target.addLegalOp<ModuleOp, ModuleTerminatorOp, TensorLoadOp,
                      TensorToMemrefOp>();
    target.addDynamicallyLegalOp<ReturnOp>(
        [&](ReturnOp op) { return typeConverter.isLegal(op); });
    // Mark terminators as legal if they have the ReturnLike trait or
    // implement the BranchOpInterface and have valid types. If they do not
    // implement the trait or interface, mark them as illegal no matter what.
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      // If it is not a terminator, ignore it.
      if (op->isKnownNonTerminator())
        return true;
      // If it is not the last operation in the block, also ignore it. We do
      // this to handle unknown operations, as well.
      Block *block = op->getBlock();
      if (!block || &block->back() != op)
        return true;
      // ReturnLike operations have to be legalized with their parent. For
      // return this is handled, for other ops they remain as is.
      if (op->hasTrait<OpTrait::ReturnLike>())
        return true;
      // All successor operands of branch like operations must be rewritten.
      if (auto branchOp = dyn_cast<BranchOpInterface>(op)) {
        for (int p = 0, e = op->getBlock()->getNumSuccessors(); p < e; ++p) {
          auto successorOperands = branchOp.getSuccessorOperands(p);
          if (successorOperands.hasValue() &&
              !typeConverter.isLegal(successorOperands.getValue().getTypes()))
            return false;
        }
        return true;
      }
      return false;
    });

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createFuncBufferizePass() {
  return std::make_unique<FuncBufferizePass>();
}
