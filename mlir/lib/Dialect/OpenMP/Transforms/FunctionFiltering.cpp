//===- FunctionFiltering.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Filter out functions intended for the host when compiling for a target
// device.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
namespace omp {

#define GEN_PASS_DEF_FUNCTIONFILTERINGPASS
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"

} // namespace omp
} // namespace mlir

using namespace mlir;

namespace {

class FunctionFilteringPass
    : public omp::impl::FunctionFilteringPassBase<FunctionFilteringPass> {

  void runOnOperation() override {
    auto op = dyn_cast<omp::OffloadModuleInterface>(getOperation());
    if (!op || !op.getIsTargetDevice())
      return;

    OpBuilder opBuilder(&getContext());
    op->walk<WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
      // Do not filter functions with target regions inside, because they have
      // to be available for both host and device so that regular and reverse
      // offloading can be supported.
      bool hasTargetRegion =
          funcOp
              ->walk<WalkOrder::PreOrder>([&](omp::TargetOp targetOp) {
                return WalkResult::interrupt();
              })
              .wasInterrupted();

      omp::DeclareTargetDeviceType declareType =
          omp::DeclareTargetDeviceType::host;
      auto declareTargetOp =
          dyn_cast<omp::DeclareTargetInterface>(funcOp.getOperation());
      if (declareTargetOp && declareTargetOp.isDeclareTarget())
        declareType = declareTargetOp.getDeclareTargetDeviceType();

      // Filtering a function here means deleting it if it doesn't contain a
      // target region. Else we explicitly set the omp.declare_target
      // attribute. The second stage of function filtering at the MLIR to LLVM
      // IR translation level will remove functions that contain the target
      // region from the generated LLVM IR.
      if (declareType != omp::DeclareTargetDeviceType::host)
        return WalkResult::advance();

      SymbolTable::UseRange funcUses = *funcOp.getSymbolUses(op);
      for (SymbolTable::SymbolUse use : funcUses) {
        Operation *callOp = use.getUser();

        // Do not delete other functions (which may be device functions) holding
        // the symbol of a host function as an attribute. The remaining
        // attribute will point to an undefined symbol after this pass.
        if (isa<FunctionOpInterface>(callOp))
          continue;

        // If the callOp has users then replace them with poison values before
        // removing it. These should get removed before translation to LLVM IR
        // by the host op filtering pass.
        if (!callOp->use_empty()) {
          SmallVector<Value> poisonResults;
          for (Value res : callOp->getResults()) {
            opBuilder.setInsertionPoint(callOp);
            poisonResults.emplace_back(
                LLVM::PoisonOp::create(opBuilder, res.getLoc(), res.getType()));
          }
          callOp->replaceAllUsesWith(poisonResults);
        }

        callOp->erase();
      }

      if (!hasTargetRegion) {
        funcOp.erase();
        return WalkResult::skip();
      }

      if (declareTargetOp)
        declareTargetOp.setDeclareTarget(
            declareType, omp::DeclareTargetCaptureClause::to,
            declareTargetOp.getDeclareTargetAutomap());
      return WalkResult::advance();
    });
  }
};

} // namespace
