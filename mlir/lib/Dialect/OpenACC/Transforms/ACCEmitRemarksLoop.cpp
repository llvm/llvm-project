//===- ACCEmitRemarksLoop.cpp - Emit OpenACC loop mapping remarks --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass emits optimization remarks describing how loops inside OpenACC
// compute regions are mapped to parallelism levels and GPU dimensions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCParMapping.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsLoop.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCEMITREMARKSLOOP
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-emit-remarks-loop"

using namespace mlir;

namespace {

static bool shouldEmitLoopRemarks(acc::ComputeRegionOp computeRegion) {
  StringRef origin = computeRegion.getOrigin();
  if (origin == acc::KernelsOp::getOperationName() ||
      origin == acc::ParallelOp::getOperationName() ||
      origin == acc::SerialOp::getOperationName())
    return true;

  if (auto func = computeRegion->getParentOfType<FunctionOpInterface>())
    return acc::isSpecializedAccRoutine(func);
  return false;
}

static std::string getACCParLevelName(acc::GPUParallelDimAttr parDim,
                                      const acc::ACCToGPUMappingPolicy &policy,
                                      acc::ComputeRegionOp computeRegion) {
  std::string accName;
  if (policy.isSeq(parDim))
    accName = "sequential";
  else if (policy.isVector(parDim))
    accName = "vector";
  else if (policy.isWorker(parDim))
    accName = "worker";
  else if (policy.isGang(parDim))
    accName = "gang";

  if (!policy.isSeq(parDim)) {
    if (std::optional<uint64_t> constant =
            computeRegion.getKnownConstantLaunchArg(parDim))
      accName += "(" + std::to_string(*constant) + ")";
  }
  return accName;
}

static std::string getGPUParDimName(acc::GPUParallelDimAttr parDim,
                                    llvm::StringRef separator) {
  auto formatDim = [&](llvm::StringRef prefix, char axis) {
    return (prefix + separator).str() + axis;
  };

  if (parDim.isThreadX())
    return formatDim("threadidx", 'x');
  if (parDim.isThreadY())
    return formatDim("threadidx", 'y');
  if (parDim.isThreadZ())
    return formatDim("threadidx", 'z');
  if (parDim.isBlockX())
    return formatDim("blockidx", 'x');
  if (parDim.isBlockY())
    return formatDim("blockidx", 'y');
  if (parDim.isBlockZ())
    return formatDim("blockidx", 'z');
  return {};
}

static void emitLoopMappingRemark(acc::ComputeRegionOp computeRegion,
                                  LoopLikeOpInterface loopOp,
                                  acc::OpenACCSupport &accSupport,
                                  const acc::ACCToGPUMappingPolicy &policy,
                                  llvm::StringRef gpuDimSeparator) {
  acc::GPUParallelDimsAttr parDimsAttr =
      loopOp->getAttrOfType<acc::GPUParallelDimsAttr>(
          acc::GPUParallelDimsAttr::name);

  SmallVector<acc::GPUParallelDimAttr, 1> seqParDims;
  ArrayRef<acc::GPUParallelDimAttr> parDims;
  if (parDimsAttr) {
    parDims = parDimsAttr.getArray();
  } else if (isa<scf::ForOp>(loopOp.getOperation())) {
    seqParDims.push_back(acc::GPUParallelDimAttr::seqDim(loopOp->getContext()));
    parDims = seqParDims;
  } else {
    return;
  }

  accSupport.emitRemark(
      loopOp,
      [&]() {
        SmallVector<std::string> accMsgs;
        SmallVector<std::string> gpuMsgs;

        for (acc::GPUParallelDimAttr parDim : parDims) {
          accMsgs.push_back(getACCParLevelName(parDim, policy, computeRegion));
          if (std::string gpuName = getGPUParDimName(parDim, gpuDimSeparator);
              !gpuName.empty())
            gpuMsgs.push_back(std::move(gpuName));
        }

        std::string msg = "!$acc loop " + llvm::join(accMsgs, ", ");

        if (uint64_t collapseCount = acc::getCollapseCount(loopOp);
            collapseCount > 1)
          msg += " collapse(" + std::to_string(collapseCount) + ")";

        if (!gpuMsgs.empty())
          msg += " ! " + llvm::join(gpuMsgs, " ");
        return msg;
      },
      DEBUG_TYPE);
}

class ACCEmitRemarksLoop
    : public acc::impl::ACCEmitRemarksLoopBase<ACCEmitRemarksLoop> {
public:
  using ACCEmitRemarksLoopBase<ACCEmitRemarksLoop>::ACCEmitRemarksLoopBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    acc::OpenACCSupport &accSupport = getAnalysis<acc::OpenACCSupport>();
    acc::DefaultACCToGPUMappingPolicy policy;
    if (gpuDimSeparator.empty())
      gpuDimSeparator = ".";

    func.walk([&](acc::ComputeRegionOp computeRegion) {
      if (!shouldEmitLoopRemarks(computeRegion))
        return;

      computeRegion.getRegion().walk([&](LoopLikeOpInterface loopOp) {
        emitLoopMappingRemark(computeRegion, loopOp, accSupport, policy,
                              gpuDimSeparator);
      });
    });
  }
};

} // namespace
