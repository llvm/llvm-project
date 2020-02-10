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
    ArrayRef<ValueHandle *> ivs, ArrayRef<ValueHandle> lbs,
    ArrayRef<ValueHandle> ubs, ArrayRef<ValueHandle> steps) {
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

mlir::edsc::LoopNestBuilder::LoopNestBuilder(ArrayRef<ValueHandle *> ivs,
                                             ArrayRef<ValueHandle> lbs,
                                             ArrayRef<ValueHandle> ubs,
                                             ArrayRef<ValueHandle> steps) {
  assert(ivs.size() == lbs.size() && "expected size of ivs and lbs to match");
  assert(ivs.size() == ubs.size() && "expected size of ivs and ubs to match");
  assert(ivs.size() == steps.size() &&
         "expected size of ivs and steps to match");
  loops.reserve(ivs.size());
  for (auto it : llvm::zip(ivs, lbs, ubs, steps)) {
    loops.emplace_back(makeLoopBuilder(std::get<0>(it), std::get<1>(it),
                                       std::get<2>(it), std::get<3>(it)));
  }
  assert(loops.size() == ivs.size() && "Mismatch loops vs ivs size");
}

void mlir::edsc::LoopNestBuilder::LoopNestBuilder::operator()(
    std::function<void(void)> fun) {
  if (fun)
    fun();
  for (auto &lit : reverse(loops))
    lit({});
}

LoopBuilder mlir::edsc::makeParallelLoopBuilder(ArrayRef<ValueHandle *> ivs,
                                                ArrayRef<ValueHandle> lbHandles,
                                                ArrayRef<ValueHandle> ubHandles,
                                                ArrayRef<ValueHandle> steps) {
  LoopBuilder result;
  auto opHandle = OperationHandle::create<loop::ParallelOp>(
      SmallVector<Value, 4>(lbHandles.begin(), lbHandles.end()),
      SmallVector<Value, 4>(ubHandles.begin(), ubHandles.end()),
      SmallVector<Value, 4>(steps.begin(), steps.end()));

  loop::ParallelOp parallelOp =
      cast<loop::ParallelOp>(*opHandle.getOperation());
  for (size_t i = 0, e = ivs.size(); i < e; ++i)
    *ivs[i] = ValueHandle(parallelOp.getBody()->getArgument(i));
  result.enter(parallelOp.getBody(), /*prev=*/1);
  return result;
}

mlir::edsc::LoopBuilder mlir::edsc::makeLoopBuilder(ValueHandle *iv,
                                                    ValueHandle lbHandle,
                                                    ValueHandle ubHandle,
                                                    ValueHandle stepHandle) {
  mlir::edsc::LoopBuilder result;
  auto forOp =
      OperationHandle::createOp<loop::ForOp>(lbHandle, ubHandle, stepHandle);
  *iv = ValueHandle(forOp.getInductionVar());
  auto *body = loop::getForInductionVarOwner(iv->getValue()).getBody();
  result.enter(body, /*prev=*/1);
  return result;
}
