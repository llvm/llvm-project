//===- AGXIRAttachTarget.cpp - Attach an AGXIR target ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GpuAGXIRAttachTarget` pass, attaching
// `#agxir.target` attributes to GPU modules.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/AGXIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVM/AGXIR/Target.h"
#include "llvm/Support/Regex.h"

namespace mlir {
#define GEN_PASS_DEF_GPUAGXIRATTACHTARGET
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::AGXIR;

namespace {
struct AGXIRAttachTarget
    : public impl::GpuAGXIRAttachTargetBase<AGXIRAttachTarget> {
  using Base::Base;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AGXIR::AGXIRDialect>();
  }
};
} // namespace

void AGXIRAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());
  ArrayRef<std::string> libs(linkLibs);
  SmallVector<StringRef> filesToLink(libs);
  auto target = builder.getAttr<AGXIRTargetAttr>(
      optLevel, triple, chip, features,
      filesToLink.empty() ? nullptr : builder.getStrArrayAttr(filesToLink));
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
