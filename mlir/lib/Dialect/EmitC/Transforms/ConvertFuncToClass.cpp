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

    // Wrap each C operator op with an expression op.
    OpBuilder builder(context);
    createClass(funcOp, builder);

    // // Create the new function inside the class
    // auto funcType = FunctionType::get(funcOp.getContext(),
    // funcOp.getFunctionType().getInputs(),
    // funcOp.getFunctionType().getResults()); auto newFuncOp =
    // builder.create<emitc::FuncOp>(
    //     funcOp.getLoc(),builder.getStringAttr("execute"), funcType );

    // builder.createBlock(&newFuncOp.getBody());
    // builder.setInsertionPointToStart(&newFuncOp.getBody().front());

    // // 7. Remap original arguments to field pointers
    // IRMapping mapper;

    // // 8. move or clone operations from original function
    // for (Operation &opToClone :
    // llvm::make_early_inc_range(funcOp.getBody().front())) {
    //     if (isa<emitc::ConstantOp>(opToClone) ||
    //         isa<emitc::SubscriptOp>(opToClone) ||
    //         isa<emitc::LoadOp>(opToClone) ||
    //         isa<emitc::AddOp>(opToClone) ||
    //         isa<emitc::AssignOp>(opToClone) ||
    //         isa<emitc::ReturnOp>(opToClone )) {
    //     builder.clone(opToClone, mapper);
    //     } else  {
    //     opToClone.emitOpError("Unsupported operation found");
    //     }
    // }
    // if (funcOp->use_empty()) funcOp->erase();
  }
};

} // namespace

} // namespace emitc
} // namespace mlir