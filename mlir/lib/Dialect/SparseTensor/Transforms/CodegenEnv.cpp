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
// Code generation environment constructor and setup
//===----------------------------------------------------------------------===//

CodegenEnv::CodegenEnv(linalg::GenericOp linop, SparsificationOptions opts,
                       unsigned numTensors, unsigned numLoops,
                       unsigned numFilterLoops)
    : linalgOp(linop), options(opts), topSort(),
      merger(numTensors, numLoops, numFilterLoops), loopEmitter(nullptr),
      sparseOut(nullptr), redVal(nullptr), redExp(-1u), redCustom(-1u) {}

void CodegenEnv::startEmit(SparseTensorLoopEmitter *le) {
  assert(!loopEmitter && "must only start emitting once");
  loopEmitter = le;
  if (sparseOut) {
    insChain = sparseOut->get();
    merger.setHasSparseOut(true);
  }
}

//===----------------------------------------------------------------------===//
// Code generation environment methods
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
