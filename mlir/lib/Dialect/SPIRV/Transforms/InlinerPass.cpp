//===- InlinerPass.cpp - SPIR-V inliner pass ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an inliner pass that treats the SPIR-V DontInline
// function control as a profitability hint.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Transforms/Inliner.h"

namespace mlir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVINLINERPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace mlir

using namespace mlir;

namespace {
class InlinerPass : public spirv::impl::SPIRVInlinerPassBase<InlinerPass> {
public:
  void runOnOperation() override;

private:
  static LogicalResult runPipelineHelper(Pass &pass, OpPassManager &pipeline,
                                         Operation *op) {
    return cast<InlinerPass>(pass).runPipeline(pipeline, op);
  }
};
} // namespace

static bool isProfitableToInline(const Inliner::ResolvedCall &call) {
  auto funcOp = dyn_cast<spirv::FuncOp>(
      call.targetNode->getCallableRegion()->getParentOp());
  return !funcOp ||
         !spirv::bitEnumContainsAll(funcOp.getFunctionControl(),
                                    spirv::FunctionControl::DontInline);
}

void InlinerPass::runOnOperation() {
  InlinerConfig config;
  Inliner inliner(getOperation(), getAnalysis<CallGraph>(), *this,
                  getAnalysisManager(), runPipelineHelper, config,
                  isProfitableToInline);
  if (failed(inliner.doInlining()))
    signalPassFailure();
}
