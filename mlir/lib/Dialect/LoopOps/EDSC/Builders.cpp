//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LoopOps/EDSC/Builders.h"
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

Operation::result_range
mlir::edsc::LoopNestBuilder::LoopNestBuilder::operator()(
    std::function<void(void)> fun) {
  if (fun)
    fun();

  for (auto &lit : reverse(loops))
    lit({});

  return loops[0].getOp()->getResults();
}

LoopBuilder mlir::edsc::makeParallelLoopBuilder(MutableArrayRef<Value> ivs,
                                                ArrayRef<Value> lbs,
                                                ArrayRef<Value> ubs,
                                                ArrayRef<Value> steps) {
  LoopBuilder result;
  auto opHandle = OperationHandle::create<loop::ParallelOp>(lbs, ubs, steps);

  loop::ParallelOp parallelOp =
      cast<loop::ParallelOp>(*opHandle.getOperation());
  for (size_t i = 0, e = ivs.size(); i < e; ++i)
    ivs[i] = parallelOp.getBody()->getArgument(i);
  result.enter(parallelOp.getBody(), /*prev=*/1);
  return result;
}

mlir::edsc::LoopBuilder
mlir::edsc::makeLoopBuilder(Value *iv, Value lb, Value ub, Value step,
                            MutableArrayRef<Value> iterArgsHandles,
                            ValueRange iterArgsInitValues) {
  mlir::edsc::LoopBuilder result;
  auto forOp =
      OperationHandle::createOp<loop::ForOp>(lb, ub, step, iterArgsInitValues);
  *iv = forOp.getInductionVar();
  auto *body = loop::getForInductionVarOwner(*iv).getBody();
  for (size_t i = 0, e = iterArgsHandles.size(); i < e; ++i) {
    // Skipping the induction variable.
    iterArgsHandles[i] = body->getArgument(i + 1);
  }
  result.setOp(forOp);
  result.enter(body, /*prev=*/1);
  return result;
}
