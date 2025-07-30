//===- InlinerExtension.cpp - Func Inliner Extension ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::func;

//===----------------------------------------------------------------------===//
// FuncDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {
/// This class defines the interface for handling inlining with func operations.
struct FuncInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// Call operations can be inlined unless specified otherwise by attributes
  /// on either the call or the callbale.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    auto callOp = dyn_cast<func::CallOp>(call);
    auto funcOp = dyn_cast<func::FuncOp>(callable);
    return !(callOp && callOp.getNoInline()) &&
           !(funcOp && funcOp.getNoInline());
  }

  /// All operations can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  /// All function bodies can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, Block *newDest) const final {
    // Only return needs to be handled here.
    auto returnOp = dyn_cast<ReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the return with a branch to the dest.
    OpBuilder builder(op);
    cf::BranchOp::create(builder, op->getLoc(), newDest,
                         returnOp.getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // Only return needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void mlir::func::registerInlinerExtension(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    dialect->addInterfaces<FuncInlinerInterface>();

    // The inliner extension relies on the ControlFlow dialect.
    ctx->getOrLoadDialect<cf::ControlFlowDialect>();
  });
}
