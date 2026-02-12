//===- MarkDeclareTarget.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark functions called from explicit target code as implicitly declare target.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace omp {

#define GEN_PASS_DEF_MARKDECLARETARGETPASS
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"

} // namespace omp
} // namespace mlir

using namespace mlir;
namespace {

class MarkDeclareTargetPass
    : public omp::impl::MarkDeclareTargetPassBase<MarkDeclareTargetPass> {

  struct ParentInfo {
    omp::DeclareTargetDeviceType devTy;
    omp::DeclareTargetCaptureClause capClause;
    bool automap;
  };

  void processSymbolRef(SymbolRefAttr symRef, ParentInfo parentInfo,
                        llvm::SmallPtrSet<Operation *, 16> visited) {
    if (auto currFOp = getOperation().lookupSymbol<func::FuncOp>(symRef)) {
      auto current =
          llvm::dyn_cast<omp::DeclareTargetInterface>(currFOp.getOperation());

      if (current.isDeclareTarget()) {
        auto currentDt = current.getDeclareTargetDeviceType();

        // Found the same function twice, with different device_types,
        // mark as Any as it belongs to both
        if (currentDt != parentInfo.devTy &&
            currentDt != omp::DeclareTargetDeviceType::any) {
          current.setDeclareTarget(omp::DeclareTargetDeviceType::any,
                                   current.getDeclareTargetCaptureClause(),
                                   current.getDeclareTargetAutomap());
        }
      } else {
        current.setDeclareTarget(parentInfo.devTy, parentInfo.capClause,
                                 parentInfo.automap);
      }

      markNestedFuncs(parentInfo, currFOp, visited);
    }
  }

  void processReductionRefs(std::optional<mlir::ArrayAttr> symRefs,
                            ParentInfo parentInfo,
                            llvm::SmallPtrSet<Operation *, 16> visited) {
    if (!symRefs)
      return;

    for (auto symRef : symRefs->getAsRange<mlir::SymbolRefAttr>()) {
      if (auto declareReductionOp =
              getOperation().lookupSymbol<omp::DeclareReductionOp>(symRef)) {
        markNestedFuncs(parentInfo, declareReductionOp, visited);
      }
    }
  }

  void processReductionClauses(Operation *op, ParentInfo parentInfo,
                               llvm::SmallPtrSet<Operation *, 16> visited) {
    llvm::TypeSwitch<Operation &>(*op)
        .Case([&](omp::LoopOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Case([&](omp::ParallelOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Case([&](omp::SectionsOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Case([&](omp::SimdOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Case([&](omp::TargetOp op) {
          processReductionRefs(op.getInReductionSyms(), parentInfo, visited);
        })
        .Case([&](omp::TaskgroupOp op) {
          processReductionRefs(op.getTaskReductionSyms(), parentInfo, visited);
        })
        .Case([&](omp::TaskloopOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
          processReductionRefs(op.getInReductionSyms(), parentInfo, visited);
        })
        .Case([&](omp::TaskOp op) {
          processReductionRefs(op.getInReductionSyms(), parentInfo, visited);
        })
        .Case([&](omp::TeamsOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Case([&](omp::WsloopOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Default([](Operation &) {});
  }

  void markNestedFuncs(ParentInfo parentInfo, Operation *currOp,
                       llvm::SmallPtrSet<Operation *, 16> visited) {
    if (visited.contains(currOp))
      return;
    visited.insert(currOp);

    currOp->walk([&, this](Operation *op) {
      if (auto callOp = llvm::dyn_cast<CallOpInterface>(op)) {
        if (auto symRef = llvm::dyn_cast_if_present<mlir::SymbolRefAttr>(
                callOp.getCallableForCallee())) {
          processSymbolRef(symRef, parentInfo, visited);
        }
      }
      processReductionClauses(op, parentInfo, visited);
    });
  }

  // This pass executes on mlir::ModuleOp's marking functions contained within
  // as implicitly declare target if they are called from within an explicitly
  // marked declare target function or a target region (TargetOp)
  void runOnOperation() override {
    for (auto functionOp : getOperation().getOps<func::FuncOp>()) {
      auto declareTargetOp = llvm::dyn_cast<omp::DeclareTargetInterface>(
          functionOp.getOperation());
      if (declareTargetOp.isDeclareTarget()) {
        llvm::SmallPtrSet<Operation *, 16> visited;
        ParentInfo parentInfo{declareTargetOp.getDeclareTargetDeviceType(),
                              declareTargetOp.getDeclareTargetCaptureClause(),
                              declareTargetOp.getDeclareTargetAutomap()};
        markNestedFuncs(parentInfo, functionOp, visited);
      }
    }

    // TODO: Extend to work with reverse-offloading, this shouldn't
    // require too much effort, just need to check the device clause
    // when it's lowering has been implemented and change the
    // DeclareTargetDeviceType argument from nohost to host depending on
    // the contents of the device clause
    getOperation()->walk([&](omp::TargetOp tarOp) {
      llvm::SmallPtrSet<Operation *, 16> visited;
      ParentInfo parentInfo = {
          /*devTy=*/omp::DeclareTargetDeviceType::nohost,
          /*capClause=*/omp::DeclareTargetCaptureClause::to,
          /*automap=*/false,
      };
      markNestedFuncs(parentInfo, tarOp, visited);
    });
  }
};

} // namespace
