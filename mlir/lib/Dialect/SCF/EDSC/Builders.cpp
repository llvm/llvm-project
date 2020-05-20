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

mlir::edsc::ParallelLoopNestBuilder::ParallelLoopNestBuilder(
    MutableArrayRef<Value> ivs, ArrayRef<Value> lbs, ArrayRef<Value> ubs,
    ArrayRef<Value> steps) {
  assert(ivs.size() == lbs.size() && "Mismatch in number of arguments");
  assert(ivs.size() == ubs.size() && "Mismatch in number of arguments");
  assert(ivs.size() == steps.size() && "Mismatch in number of arguments");

  loops.emplace_back(makeParallelLoopBuilder(ivs, lbs, ubs, steps));
}

void mlir::edsc::ParallelLoopNestBuilder::operator()(
    function_ref<void(void)> fun) {
  if (fun)
    fun();
  // Iterate on the calling operator() on all the loops in the nest.
  // The iteration order is from innermost to outermost because enter/exit needs
  // to be asymmetric (i.e. enter() occurs on LoopBuilder construction, exit()
  // occurs on calling operator()). The asymmetry is required for properly
  // nesting imperfectly nested regions (see LoopBuilder::operator()).
  for (auto lit = loops.rbegin(), eit = loops.rend(); lit != eit; ++lit)
    (*lit)();
}

mlir::edsc::LoopNestBuilder::LoopNestBuilder(MutableArrayRef<Value> ivs,
                                             ArrayRef<Value> lbs,
                                             ArrayRef<Value> ubs,
                                             ArrayRef<Value> steps) {
  assert(ivs.size() == lbs.size() && "expected size of ivs and lbs to match");
  assert(ivs.size() == ubs.size() && "expected size of ivs and ubs to match");
  assert(ivs.size() == steps.size() &&
         "expected size of ivs and steps to match");
  loops.reserve(ivs.size());
  for (auto it : llvm::zip(ivs, lbs, ubs, steps))
    loops.emplace_back(makeLoopBuilder(&std::get<0>(it), std::get<1>(it),
                                       std::get<2>(it), std::get<3>(it)));
  assert(loops.size() == ivs.size() && "Mismatch loops vs ivs size");
}

mlir::edsc::LoopNestBuilder::LoopNestBuilder(
    Value *iv, Value lb, Value ub, Value step,
    MutableArrayRef<Value> iterArgsHandles, ValueRange iterArgsInitValues) {
  assert(iterArgsInitValues.size() == iterArgsHandles.size() &&
         "expected size of arguments and argument_handles to match");
  loops.emplace_back(
      makeLoopBuilder(iv, lb, ub, step, iterArgsHandles, iterArgsInitValues));
}

mlir::edsc::LoopNestBuilder::LoopNestBuilder(Value *iv, Value lb, Value ub,
                                             Value step) {
  SmallVector<Value, 0> noArgs;
  loops.emplace_back(makeLoopBuilder(iv, lb, ub, step, noArgs, {}));
}

ValueRange mlir::edsc::LoopNestBuilder::LoopNestBuilder::operator()(
    std::function<void(void)> fun) {
  if (fun)
    fun();

  for (auto &lit : reverse(loops))
    lit({});

  if (!loops.empty())
    return loops[0].getOp()->getResults();
  return {};
}

LoopBuilder mlir::edsc::makeParallelLoopBuilder(MutableArrayRef<Value> ivs,
                                                ArrayRef<Value> lbs,
                                                ArrayRef<Value> ubs,
                                                ArrayRef<Value> steps) {
  scf::ParallelOp parallelOp = OperationBuilder<scf::ParallelOp>(
      SmallVector<Value, 4>(lbs.begin(), lbs.end()),
      SmallVector<Value, 4>(ubs.begin(), ubs.end()),
      SmallVector<Value, 4>(steps.begin(), steps.end()));
  for (size_t i = 0, e = ivs.size(); i < e; ++i)
    ivs[i] = parallelOp.getBody()->getArgument(i);
  LoopBuilder result;
  result.enter(parallelOp.getBody());
  return result;
}

mlir::edsc::LoopBuilder
mlir::edsc::makeLoopBuilder(Value *iv, Value lb, Value ub, Value step,
                            MutableArrayRef<Value> iterArgsHandles,
                            ValueRange iterArgsInitValues) {
  mlir::edsc::LoopBuilder result;
  scf::ForOp forOp =
      OperationBuilder<scf::ForOp>(lb, ub, step, iterArgsInitValues);
  *iv = Value(forOp.getInductionVar());
  auto *body = scf::getForInductionVarOwner(*iv).getBody();
  for (size_t i = 0, e = iterArgsHandles.size(); i < e; ++i) {
    // Skipping the induction variable.
    iterArgsHandles[i] = body->getArgument(i + 1);
  }
  result.setOp(forOp);
  result.enter(body);
  return result;
}

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
