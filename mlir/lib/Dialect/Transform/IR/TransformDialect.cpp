//===- TransformDialect.cpp - Transform Dialect Definition ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

using namespace mlir;

#include "mlir/Dialect/Transform/IR/TransformDialect.cpp.inc"

void transform::TransformDialect::initialize() {
  // Using the checked version to enable the same assertions as for the ops from
  // extensions.
  addOperationsChecked<
#define GET_OP_LIST
#include "mlir/Dialect/Transform/IR/TransformOps.cpp.inc"
      >();
}

void transform::TransformDialect::mergeInPDLMatchHooks(
    llvm::StringMap<PDLConstraintFunction> &&constraintFns) {
  // Steal the constraint functions form the given map.
  for (auto &it : constraintFns)
    pdlMatchHooks.registerConstraintFunction(it.getKey(), std::move(it.second));
}

const llvm::StringMap<PDLConstraintFunction> &
transform::TransformDialect::getPDLConstraintHooks() const {
  return pdlMatchHooks.getConstraintFunctions();
}
