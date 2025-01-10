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

static void collectPtrs(mlir::ValueRange operands,
                        llvm::SmallVector<std::pair<Value, Value>> &values,
                        bool hostToDevice) {
  for (auto operand : operands) {
    Value varPtr = acc::getVarPtr(operand.getDefiningOp());
    Value accPtr = acc::getAccPtr(operand.getDefiningOp());
    if (varPtr && accPtr) {
      if (hostToDevice)
        values.push_back({varPtr, accPtr});
      else
        values.push_back({accPtr, varPtr});
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
static void collectAndReplaceInRegion(Op &op, bool hostToDevice) {
  llvm::SmallVector<std::pair<Value, Value>> values;

  if constexpr (std::is_same_v<Op, acc::LoopOp>) {
    collectPtrs(op.getReductionOperands(), values, hostToDevice);
    collectPtrs(op.getPrivateOperands(), values, hostToDevice);
  } else {
    collectPtrs(op.getDataClauseOperands(), values, hostToDevice);
    if constexpr (!std::is_same_v<Op, acc::KernelsOp> &&
                  !std::is_same_v<Op, acc::DataOp> &&
                  !std::is_same_v<Op, acc::DeclareOp>) {
      collectPtrs(op.getReductionOperands(), values, hostToDevice);
      collectPtrs(op.getPrivateOperands(), values, hostToDevice);
      collectPtrs(op.getFirstprivateOperands(), values, hostToDevice);
    }
  }

  for (auto p : values)
    replaceAllUsesInAccComputeRegionsWith<Op>(std::get<0>(p), std::get<1>(p),
                                              op.getRegion());
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

    funcOp.walk([&](Operation *op) {
      if (!isa<ACC_COMPUTE_CONSTRUCT_AND_LOOP_OPS>(*op) &&
          !(isa<ACC_DATA_CONSTRUCT_STRUCTURED_OPS>(*op) &&
            applyToAccDataConstruct))
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
      } else {
        llvm_unreachable("unsupported acc region op");
      }
    });
  }
};

} // end anonymous namespace
