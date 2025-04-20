//===- SetTargetSpecs.cpp - Sets target specs -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `DltiSetTargetSpecsFromTarget` pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/Transforms/Passes.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_DLTISETTARGETSPECSFROMTARGET
#include "mlir/Dialect/DLTI/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct SetTargetSpecs
    : public impl::DltiSetTargetSpecsFromTargetBase<SetTargetSpecs> {
  using Base::Base;

  void runOnOperation() override {
    if (failed(setTargetSpecsFromTarget(getOperation())))
      return signalPassFailure();
  }
};
} // namespace

LogicalResult mlir::setTargetSpecsFromTarget(Operation *op) {
  auto dlOp = dyn_cast<DataLayoutOpInterface>(op);
  if (!dlOp)
    return op->emitError("Op doesn't implement `DataLayoutOpInterface`.");
  TargetAttrInterface target = dlOp.getTargetAttr();
  if (!target)
    return op->emitError("Op doesn't have a target.");
  TargetSpec spec;
  if (failed(target.setTargetSpec(spec)))
    return failure();
  if (spec.systemSpec)
    dlOp.setTargetSystemSpec(spec.systemSpec);
  if (spec.dataLayout)
    dlOp.setDataLayoutSpec(spec.dataLayout);
  return success();
}
