//===- TestWhileOpBuilder.cpp - Pass to test WhileOp::build ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test some builder functions of WhileOp. It
// tests the regression explained in https://reviews.llvm.org/D142952, where
// a WhileOp::build overload crashed when fed with operands of different types
// than the result types.
//
// To test the build function, the pass copies each WhileOp found in the body
// of a FuncOp and adds an additional WhileOp with the same operands and result
// types (but dummy computations) using the builder in question.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::scf;

namespace {
struct TestSCFWhileOpBuilderPass
    : public PassWrapper<TestSCFWhileOpBuilderPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSCFWhileOpBuilderPass)

  StringRef getArgument() const final { return "test-scf-while-op-builder"; }
  StringRef getDescription() const final {
    return "test build functions of scf.while";
  }
  explicit TestSCFWhileOpBuilderPass() = default;
  TestSCFWhileOpBuilderPass(const TestSCFWhileOpBuilderPass &pass) = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](WhileOp whileOp) {
      Location loc = whileOp->getLoc();
      ImplicitLocOpBuilder builder(loc, whileOp);

      // Create a WhileOp with the same operands and result types.
      TypeRange resultTypes = whileOp->getResultTypes();
      ValueRange operands = whileOp->getOperands();
      builder.create<WhileOp>(
          loc, resultTypes, operands, /*beforeBuilder=*/
          [&](OpBuilder &b, Location loc, ValueRange args) {
            // Just cast the before args into the right types for condition.
            ImplicitLocOpBuilder builder(loc, b);
            auto castOp =
                builder.create<UnrealizedConversionCastOp>(resultTypes, args);
            auto cmp = builder.create<ConstantIntOp>(/*value=*/1, /*width=*/1);
            builder.create<ConditionOp>(cmp, castOp->getResults());
          },
          /*afterBuilder=*/
          [&](OpBuilder &b, Location loc, ValueRange args) {
            // Just cast the after args into the right types for yield.
            ImplicitLocOpBuilder builder(loc, b);
            auto castOp = builder.create<UnrealizedConversionCastOp>(
                operands.getTypes(), args);
            builder.create<YieldOp>(castOp->getResults());
          });
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestSCFWhileOpBuilderPass() {
  PassRegistration<TestSCFWhileOpBuilderPass>();
}
} // namespace test
} // namespace mlir
