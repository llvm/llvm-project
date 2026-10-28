//===- AIRAttachTarget.cpp - Attach an AIR target -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GpuAIRAttachTarget` pass, attaching
// `#air.target` attributes to GPU modules.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/AIR/IR/AIRDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Regex.h"

namespace mlir {
#define GEN_PASS_DEF_GPUAIRATTACHTARGET
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct AIRAttachTarget : public impl::GpuAIRAttachTargetBase<AIRAttachTarget> {
  using Base::Base;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<air::AIRDialect>();
  }
};
} // namespace

void AIRAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());
  auto target = builder.getAttr<air::AIRTargetAttr>(optLevel, triple, chip,
                                                    features);
  llvm::Regex matcher(moduleMatcher);
  for (Region &region : getOperation()->getRegions())
    for (Block &block : region.getBlocks())
      for (auto module : block.getOps<gpu::GPUModuleOp>()) {
        // Check if the name of the module matches.
        if (!moduleMatcher.empty() && !matcher.match(module.getName()))
          continue;
        // Create the target array.
        SmallVector<Attribute> targets;
        if (std::optional<ArrayAttr> attrs = module.getTargets())
          targets.append(attrs->getValue().begin(), attrs->getValue().end());
        targets.push_back(target);
        // Remove any duplicate targets.
        targets.erase(llvm::unique(targets), targets.end());
        // Update the target attribute array.
        module.setTargetsAttr(builder.getArrayAttr(targets));
      }
}
