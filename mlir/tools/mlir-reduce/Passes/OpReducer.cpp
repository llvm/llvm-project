//===- OpReducer.cpp - Operation Reducer ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the OpReducer class. It defines a variant generator method
// with the purpose of producing different variants by eliminating a
// parameterizable type of operations from the  parent module.
//
//===----------------------------------------------------------------------===//
#include "mlir/Reducer/Passes/OpReducer.h"

using namespace mlir;

OpReducerImpl::OpReducerImpl(
    llvm::function_ref<std::vector<Operation *>(ModuleOp)> getSpecificOps)
    : getSpecificOps(getSpecificOps) {}

/// Return the name of this reducer class.
StringRef OpReducerImpl::getName() {
  return StringRef("High Level Operation Reduction");
}

/// Return the initial transformSpace containing the transformable indices.
std::vector<bool> OpReducerImpl::initTransformSpace(ModuleOp module) {
  auto ops = getSpecificOps(module);
  int numOps = std::distance(ops.begin(), ops.end());
  return ReductionTreeUtils::createTransformSpace(module, numOps);
}

/// Generate variants by removing opType operations from the module in the
/// parent and link the variants as childs in the Reduction Tree Pass.
void OpReducerImpl::generateVariants(
    ReductionNode *parent, const Tester &test, int numVariants,
    llvm::function_ref<void(ModuleOp, int, int)> transform) {
  ReductionTreeUtils::createVariants(parent, test, numVariants, transform,
                                     true);
}
