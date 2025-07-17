//===- LegalizeDataValues.cpp - -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_LEGALIZEDATAVALUESINREGION
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

using namespace mlir;

namespace {

static bool insideAccComputeRegion(mlir::Operation *op) {
  mlir::Operation *parent{op->getParentOp()};
  while (parent) {
    if (isa<ACC_COMPUTE_CONSTRUCT_OPS>(parent)) {
      return true;
    }
    parent = parent->getParentOp();
  }
  return false;
}

static void collectVars(mlir::ValueRange operands,
                        llvm::SmallVector<std::pair<Value, Value>> &values,
                        bool hostToDevice) {
  for (auto operand : operands) {
    Value var = acc::getVar(operand.getDefiningOp());
    Value accVar = acc::getAccVar(operand.getDefiningOp());
    if (var && accVar) {
      if (hostToDevice)
        values.push_back({var, accVar});
      else
        values.push_back({accVar, var});
    }
  }
}

template <typename Op>
static void replaceAllUsesInAccComputeRegionsWith(Value orig, Value replacement,
                                                  Region &outerRegion) {
  for (auto &use : llvm::make_early_inc_range(orig.getUses())) {
    if (outerRegion.isAncestor(use.getOwner()->getParentRegion())) {
      if constexpr (std::is_same_v<Op, acc::DataOp> ||
                    std::is_same_v<Op, acc::DeclareOp>) {
        // For data construct regions, only replace uses in contained compute
        // regions.
        if (insideAccComputeRegion(use.getOwner())) {
          use.set(replacement);
        }
      } else {
        use.set(replacement);
      }
    }
  }
}

template <typename Op>
static void replaceAllUsesInUnstructuredComputeRegionWith(
    Op &op, llvm::SmallVector<std::pair<Value, Value>> &values,
    DominanceInfo &domInfo, PostDominanceInfo &postDomInfo) {

  SmallVector<Operation *> exitOps;
  if constexpr (std::is_same_v<Op, acc::DeclareEnterOp>) {
    // For declare enter/exit pairs, collect all exit ops
    for (auto *user : op.getToken().getUsers()) {
      if (auto declareExit = dyn_cast<acc::DeclareExitOp>(user))
        exitOps.push_back(declareExit);
    }
    if (exitOps.empty())
      return;
  }

  for (auto p : values) {
    Value hostVal = std::get<0>(p);
    Value deviceVal = std::get<1>(p);
    for (auto &use : llvm::make_early_inc_range(hostVal.getUses())) {
      Operation *owner = use.getOwner();

      // Check It's the case that the acc entry operation dominates the use.
      if (!domInfo.dominates(op.getOperation(), owner))
        continue;

      // Check It's the case that at least one of the acc exit operations
      // post-dominates the use
      bool hasPostDominatingExit = false;
      for (auto *exit : exitOps) {
        if (postDomInfo.postDominates(exit, owner)) {
          hasPostDominatingExit = true;
          break;
        }
      }

      if (!hasPostDominatingExit)
        continue;

      if (insideAccComputeRegion(owner))
        use.set(deviceVal);
    }
  }
}

template <typename Op>
static void
collectAndReplaceInRegion(Op &op, bool hostToDevice,
                          DominanceInfo *domInfo = nullptr,
                          PostDominanceInfo *postDomInfo = nullptr) {
  llvm::SmallVector<std::pair<Value, Value>> values;

  if constexpr (std::is_same_v<Op, acc::LoopOp>) {
    collectVars(op.getReductionOperands(), values, hostToDevice);
    collectVars(op.getPrivateOperands(), values, hostToDevice);
  } else {
    collectVars(op.getDataClauseOperands(), values, hostToDevice);
    if constexpr (!std::is_same_v<Op, acc::KernelsOp> &&
                  !std::is_same_v<Op, acc::DataOp> &&
                  !std::is_same_v<Op, acc::DeclareOp> &&
                  !std::is_same_v<Op, acc::HostDataOp> &&
                  !std::is_same_v<Op, acc::DeclareEnterOp>) {
      collectVars(op.getReductionOperands(), values, hostToDevice);
      collectVars(op.getPrivateOperands(), values, hostToDevice);
      collectVars(op.getFirstprivateOperands(), values, hostToDevice);
    }
  }

  if constexpr (std::is_same_v<Op, acc::DeclareEnterOp>) {
    assert(domInfo && postDomInfo &&
           "Dominance info required for DeclareEnterOp");
    replaceAllUsesInUnstructuredComputeRegionWith<Op>(op, values, *domInfo,
                                                      *postDomInfo);
  } else {
    for (auto p : values) {
      replaceAllUsesInAccComputeRegionsWith<Op>(std::get<0>(p), std::get<1>(p),
                                                op.getRegion());
    }
  }
}

class LegalizeDataValuesInRegion
    : public acc::impl::LegalizeDataValuesInRegionBase<
          LegalizeDataValuesInRegion> {
public:
  using LegalizeDataValuesInRegionBase<
      LegalizeDataValuesInRegion>::LegalizeDataValuesInRegionBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    bool replaceHostVsDevice = this->hostToDevice.getValue();

    // Initialize dominance info
    DominanceInfo domInfo;
    PostDominanceInfo postDomInfo;
    bool computedDomInfo = false;

    funcOp.walk([&](Operation *op) {
      if (!isa<ACC_COMPUTE_CONSTRUCT_AND_LOOP_OPS>(*op) &&
          !(isa<ACC_DATA_CONSTRUCT_STRUCTURED_OPS>(*op) &&
            applyToAccDataConstruct) &&
          !isa<acc::DeclareEnterOp>(*op))
        return;

      if (auto parallelOp = dyn_cast<acc::ParallelOp>(*op)) {
        collectAndReplaceInRegion(parallelOp, replaceHostVsDevice);
      } else if (auto serialOp = dyn_cast<acc::SerialOp>(*op)) {
        collectAndReplaceInRegion(serialOp, replaceHostVsDevice);
      } else if (auto kernelsOp = dyn_cast<acc::KernelsOp>(*op)) {
        collectAndReplaceInRegion(kernelsOp, replaceHostVsDevice);
      } else if (auto loopOp = dyn_cast<acc::LoopOp>(*op)) {
        collectAndReplaceInRegion(loopOp, replaceHostVsDevice);
      } else if (auto dataOp = dyn_cast<acc::DataOp>(*op)) {
        collectAndReplaceInRegion(dataOp, replaceHostVsDevice);
      } else if (auto declareOp = dyn_cast<acc::DeclareOp>(*op)) {
        collectAndReplaceInRegion(declareOp, replaceHostVsDevice);
      } else if (auto hostDataOp = dyn_cast<acc::HostDataOp>(*op)) {
        collectAndReplaceInRegion(hostDataOp, replaceHostVsDevice);
      } else if (auto declareEnterOp = dyn_cast<acc::DeclareEnterOp>(*op)) {
        if (!computedDomInfo) {
          domInfo = DominanceInfo(funcOp);
          postDomInfo = PostDominanceInfo(funcOp);
          computedDomInfo = true;
        }
        collectAndReplaceInRegion(declareEnterOp, replaceHostVsDevice, &domInfo,
                                  &postDomInfo);
      } else {
        llvm_unreachable("unsupported acc region op");
      }
    });
  }
};

} // end anonymous namespace
