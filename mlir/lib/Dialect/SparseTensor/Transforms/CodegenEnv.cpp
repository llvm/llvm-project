//===- CodegenEnv.cpp -  Code generation environment class ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenEnv.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// Code generation environment constructor and general methods
//===----------------------------------------------------------------------===//

CodegenEnv::CodegenEnv(linalg::GenericOp linop, SparsificationOptions opts,
                       unsigned numTensors, unsigned numLoops,
                       unsigned numFilterLoops)
    : linalgOp(linop), sparseOptions(opts),
      latticeMerger(numTensors, numLoops, numFilterLoops), loopEmitter(),
      topSort(), sparseOut(nullptr), outerParNest(-1u), insChain(), expValues(),
      expFilled(), expAdded(), expCount(), redVal(), redExp(-1u),
      redCustom(-1u) {}

void CodegenEnv::startEmit(OpOperand *so, unsigned lv) {
  assert(sparseOut == nullptr && insChain == nullptr &&
         "must only start emitting once");
  sparseOut = so;
  outerParNest = lv;
  if (sparseOut) {
    insChain = sparseOut->get();
    latticeMerger.setHasSparseOut(true);
  }
  // Initialize loop emitter.
  SmallVector<Value> tensors;
  for (OpOperand &t : linalgOp->getOpOperands())
    tensors.push_back(t.get());
  loopEmitter.initialize(tensors,
                         StringAttr::get(linalgOp.getContext(),
                                         linalg::GenericOp::getOperationName()),
                         /*hasOutput=*/true,
                         /*isSparseOut=*/sparseOut != nullptr, topSort);
}

Optional<Operation *> CodegenEnv::genLoopBoundary(
    function_ref<Optional<Operation *>(MutableArrayRef<Value> parameters)>
        callback) {
  SmallVector<Value> params;
  if (isReduc())
    params.push_back(redVal);
  if (isExpand())
    params.push_back(expCount);
  if (insChain != nullptr)
    params.push_back(insChain);
  auto r = callback(params); // may update parameters
  unsigned i = 0;
  if (isReduc())
    updateReduc(params[i++]);
  if (isExpand())
    updateExpandCount(params[i++]);
  if (insChain != nullptr)
    updateInsertionChain(params[i]);
  return r;
}

//===----------------------------------------------------------------------===//
// Code generation environment topological sort methods
//===----------------------------------------------------------------------===//

ArrayRef<unsigned> CodegenEnv::getTopSortSlice(size_t n, size_t m) const {
  return ArrayRef<unsigned>(topSort).slice(n, m);
}

ArrayRef<unsigned> CodegenEnv::getLoopCurStack() const {
  return getTopSortSlice(0, loopEmitter.getCurrentDepth());
}

Value CodegenEnv::getLoopIdxValue(size_t loopIdx) const {
  for (unsigned lv = 0, lve = topSort.size(); lv < lve; lv++)
    if (topSort[lv] == loopIdx)
      return loopEmitter.getLoopIV(lv);
  llvm_unreachable("invalid loop index");
}

//===----------------------------------------------------------------------===//
// Code generation environment sparse tensor output and expansion methods
//===----------------------------------------------------------------------===//

void CodegenEnv::updateInsertionChain(Value chain) {
  assert(sparseOut != nullptr && insChain != nullptr);
  insChain = chain;
}

bool CodegenEnv::atExpandLevel(OpOperand *o, unsigned rank, unsigned lv) const {
  return sparseOut == o && outerParNest == rank - 1 && outerParNest == lv;
}

void CodegenEnv::startExpand(Value values, Value filled, Value added,
                             Value count) {
  assert(sparseOut != nullptr && expValues == nullptr);
  expValues = values;
  expFilled = filled;
  expAdded = added;
  expCount = count;
}

void CodegenEnv::updateExpandCount(Value count) {
  assert(sparseOut != nullptr && expValues != nullptr);
  expCount = count;
}

void CodegenEnv::endExpand() {
  assert(sparseOut != nullptr && expValues != nullptr);
  expValues = expFilled = expAdded = expCount = Value();
}

//===----------------------------------------------------------------------===//
// Code generation environment reduction methods
//===----------------------------------------------------------------------===//

void CodegenEnv::startReduc(unsigned exp, Value val) {
  assert(redExp == -1u && exp != -1u);
  redExp = exp;
  updateReduc(val);
}

void CodegenEnv::updateReduc(Value val) {
  assert(redExp != -1u);
  redVal = exp(redExp).val = val;
}

Value CodegenEnv::endReduc() {
  Value val = redVal;
  updateReduc(Value());
  redExp = -1u;
  return val;
}

void CodegenEnv::startCustomReduc(unsigned exp) {
  assert(redCustom == -1u && exp != -1u);
  redCustom = exp;
}

Value CodegenEnv::getCustomRedId() {
  assert(redCustom != -1u);
  return dyn_cast<sparse_tensor::ReduceOp>(exp(redCustom).op).getIdentity();
}

void CodegenEnv::endCustomReduc() {
  assert(redCustom != -1u);
  redCustom = -1u;
}
