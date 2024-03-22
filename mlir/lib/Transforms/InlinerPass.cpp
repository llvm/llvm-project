//===- InlinerPass.cpp - Pass to inline function calls --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a basic inlining algorithm that operates bottom up over
// the Strongly Connect Components(SCCs) of the CallGraph. This enables a more
// incremental propagation of inlining decisions from the leafs to the roots of
// the callgraph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Inliner.h"

namespace mlir {
#define GEN_PASS_DEF_INLINER
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "inliner-pass"

using namespace mlir;

/// This function implements the inliner optimization pipeline.
static void defaultInlinerOptPipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
}

//===----------------------------------------------------------------------===//
// InlinerPass
//===----------------------------------------------------------------------===//

namespace {
class InlinerPass : public impl::InlinerBase<InlinerPass> {
public:
  InlinerPass();
  InlinerPass(const InlinerPass &) = default;
  InlinerPass(std::function<void(OpPassManager &)> defaultPipeline);
  InlinerPass(std::function<void(OpPassManager &)> defaultPipeline,
              llvm::StringMap<OpPassManager> opPipelines);
  void runOnOperation() override;

  /// A callback provided to the inliner driver to execute
  /// the specified pass pipeline on the given operation
  /// within the context of the current inliner pass,
  /// which is passed as the first argument.
  /// runPipeline API is protected within the Pass class,
  /// so this helper is required to call it from the foreign
  /// inliner driver.
  static LogicalResult runPipelineHelper(Pass &pass, OpPassManager &pipeline,
                                         Operation *op) {
    return mlir::cast<InlinerPass>(pass).runPipeline(pipeline, op);
  }

private:
  /// Attempt to initialize the options of this pass from the given string.
  /// Derived classes may override this method to hook into the point at which
  /// options are initialized, but should generally always invoke this base
  /// class variant.
  LogicalResult initializeOptions(StringRef options) override;

  /// Inliner configuration parameters created from the pass options.
  InlinerConfig config;
};
} // namespace

InlinerPass::InlinerPass() : InlinerPass(defaultInlinerOptPipeline) {}

InlinerPass::InlinerPass(
    std::function<void(OpPassManager &)> defaultPipelineArg)
    : InlinerPass(std::move(defaultPipelineArg),
                  llvm::StringMap<OpPassManager>{}) {}

InlinerPass::InlinerPass(std::function<void(OpPassManager &)> defaultPipeline,
                         llvm::StringMap<OpPassManager> opPipelines)
    : config(std::move(defaultPipeline), maxInliningIterations) {
  if (opPipelines.empty())
    return;

  // Update the option for the op specific optimization pipelines.
  for (auto &it : opPipelines)
    opPipelineList.addValue(it.second);
  config.setOpPipelines(std::move(opPipelines));
}

// Return true if the inlining ratio does not exceed the threshold.
static bool isProfitableToInline(const Inliner::ResolvedCall &resolvedCall,
                                 unsigned inliningThreshold) {
  // Return early, ratio <= 0U will always be false.
  if (inliningThreshold == 0U)
    return false;
  // Return early, ratio <= -1U will always be true.
  if (inliningThreshold == -1U)
    return true;

  Region *callerRegion = resolvedCall.sourceNode->getCallableRegion();
  Region *calleeRegion = resolvedCall.targetNode->getCallableRegion();

  // We should not get external nodes here, but just return true
  // for now to preserve the original behavior of the inliner pass.
  if (!callerRegion || !calleeRegion)
    return true;

  auto countOps = [](Region *region) {
    unsigned count = 0;
    region->walk([&](Operation *) { ++count; });
    return count;
  };

  unsigned callerOps = countOps(callerRegion);

  // Always inline empty callees (if it is possible at all).
  if (callerOps == 0)
    return true;

  unsigned ratio = countOps(calleeRegion) * 100 / callerOps;
  LLVM_DEBUG(llvm::dbgs() << "Callee / caller operation ratio (max: "
                          << inliningThreshold << "%): " << ratio << "%\n");
  return ratio <= inliningThreshold;
}

void InlinerPass::runOnOperation() {
  CallGraph &cg = getAnalysis<CallGraph>();

  // The inliner should only be run on operations that define a symbol table,
  // as the callgraph will need to resolve references.
  Operation *op = getOperation();
  if (!op->hasTrait<OpTrait::SymbolTable>()) {
    op->emitOpError() << " was scheduled to run under the inliner, but does "
                         "not define a symbol table";
    return signalPassFailure();
  }

  // By default, assume that any inlining is profitable.
  auto profitabilityCb = [=](const Inliner::ResolvedCall &call) {
    return isProfitableToInline(call, inliningThreshold);
  };

  // Get an instance of the inliner.
  Inliner inliner(op, cg, *this, getAnalysisManager(), runPipelineHelper,
                  config, profitabilityCb);

  // Run the inlining.
  if (failed(inliner.doInlining()))
    signalPassFailure();
  return;
}

LogicalResult InlinerPass::initializeOptions(StringRef options) {
  if (failed(Pass::initializeOptions(options)))
    return failure();

  // Initialize the pipeline builder for operations without the dedicated
  // optimization pipeline in opPipelineList to use the option string.
  // TODO: Use a generic pass manager for the pre-inline pipeline, and remove
  // this.
  if (!defaultPipelineStr.empty()) {
    std::string defaultPipelineCopy = defaultPipelineStr;
    config.setDefaultPipeline([=](OpPassManager &pm) {
      (void)parsePassPipeline(defaultPipelineCopy, pm);
    });
  } else if (defaultPipelineStr.getNumOccurrences()) {
    config.setDefaultPipeline(nullptr);
  }

  // Initialize the op specific pass pipelines.
  llvm::StringMap<OpPassManager> pipelines;
  for (OpPassManager pipeline : opPipelineList)
    if (!pipeline.empty())
      pipelines.try_emplace(pipeline.getOpAnchorName(), pipeline);
  config.setOpPipelines(std::move(pipelines));

  config.setMaxInliningIterations(maxInliningIterations);

  return success();
}

std::unique_ptr<Pass> mlir::createInlinerPass() {
  return std::make_unique<InlinerPass>();
}
std::unique_ptr<Pass>
mlir::createInlinerPass(llvm::StringMap<OpPassManager> opPipelines) {
  return std::make_unique<InlinerPass>(defaultInlinerOptPipeline,
                                       std::move(opPipelines));
}
std::unique_ptr<Pass> mlir::createInlinerPass(
    llvm::StringMap<OpPassManager> opPipelines,
    std::function<void(OpPassManager &)> defaultPipelineBuilder) {
  return std::make_unique<InlinerPass>(std::move(defaultPipelineBuilder),
                                       std::move(opPipelines));
}
