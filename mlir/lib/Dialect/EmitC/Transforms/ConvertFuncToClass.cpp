//===- ConvertFuncToClass.cpp - Convert functions to classes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace emitc {

#define GEN_PASS_DEF_CONVERTFUNCTOCLASSPASS
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"

namespace {

struct ConvertFuncToClassPass
    : public impl::ConvertFuncToClassPassBase<ConvertFuncToClassPass> {
  void runOnOperation() override {
    emitc::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();

    OpBuilder builder(context);
    createClass(funcOp, builder);
  }
};

} // namespace

} // namespace emitc
} // namespace mlir
