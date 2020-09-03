//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace mlir::edsc;

mlir::scf::ValueVector
mlir::edsc::loopNestBuilder(ValueRange lbs, ValueRange ubs, ValueRange steps,
                            function_ref<void(ValueRange)> fun) {
  // Delegates actual construction to scf::buildLoopNest by wrapping `fun` into
  // the expected function interface.
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return mlir::scf::buildLoopNest(
      ScopedContext::getBuilderRef(), ScopedContext::getLocation(), lbs, ubs,
      steps, [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        ScopedContext context(builder, loc);
        if (fun)
          fun(ivs);
      });
}

mlir::scf::ValueVector
mlir::edsc::loopNestBuilder(Value lb, Value ub, Value step,
                            function_ref<void(Value)> fun) {
  // Delegates to the ValueRange-based version by wrapping the lambda.
  auto wrapper = [&](ValueRange ivs) {
    assert(ivs.size() == 1);
    if (fun)
      fun(ivs[0]);
  };
  return loopNestBuilder(ValueRange(lb), ValueRange(ub), ValueRange(step),
                         wrapper);
}

mlir::scf::ValueVector mlir::edsc::loopNestBuilder(
    Value lb, Value ub, Value step, ValueRange iterArgInitValues,
    function_ref<scf::ValueVector(Value, ValueRange)> fun) {
  // Delegates actual construction to scf::buildLoopNest by wrapping `fun` into
  // the expected function interface.
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return mlir::scf::buildLoopNest(
      ScopedContext::getBuilderRef(), ScopedContext::getLocation(), lb, ub,
      step, iterArgInitValues,
      [&](OpBuilder &builder, Location loc, ValueRange ivs, ValueRange args) {
        assert(ivs.size() == 1 && "expected one induction variable");
        ScopedContext context(builder, loc);
        if (fun)
          return fun(ivs[0], args);
        return scf::ValueVector(iterArgInitValues.begin(),
                                iterArgInitValues.end());
      });
}

static std::function<void(OpBuilder &, Location)>
wrapIfBody(function_ref<scf::ValueVector()> body, TypeRange expectedTypes) {
  (void)expectedTypes;
  return [=](OpBuilder &builder, Location loc) {
    ScopedContext context(builder, loc);
    scf::ValueVector returned = body();
    assert(ValueRange(returned).getTypes() == expectedTypes &&
           "'if' body builder returned values of unexpected type");
    builder.create<scf::YieldOp>(loc, returned);
  };
}

ValueRange
mlir::edsc::conditionBuilder(TypeRange results, Value condition,
                             function_ref<scf::ValueVector()> thenBody,
                             function_ref<scf::ValueVector()> elseBody) {
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  assert(thenBody && "thenBody is mandatory");

  auto ifOp = ScopedContext::getBuilderRef().create<scf::IfOp>(
      ScopedContext::getLocation(), results, condition,
      wrapIfBody(thenBody, results), wrapIfBody(elseBody, results));
  return ifOp.getResults();
}

static std::function<void(OpBuilder &, Location)>
wrapZeroResultIfBody(function_ref<void()> body) {
  return [=](OpBuilder &builder, Location loc) {
    ScopedContext context(builder, loc);
    body();
    builder.create<scf::YieldOp>(loc);
  };
}

ValueRange mlir::edsc::conditionBuilder(Value condition,
                                        function_ref<void()> thenBody,
                                        function_ref<void()> elseBody) {
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  assert(thenBody && "thenBody is mandatory");

  ScopedContext::getBuilderRef().create<scf::IfOp>(
      ScopedContext::getLocation(), condition, wrapZeroResultIfBody(thenBody),
      elseBody ? llvm::function_ref<void(OpBuilder &, Location)>(
                     wrapZeroResultIfBody(elseBody))
               : llvm::function_ref<void(OpBuilder &, Location)>(nullptr));
  return {};
}
