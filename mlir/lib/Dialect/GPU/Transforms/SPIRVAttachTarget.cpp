//===- SPIRVAttachTarget.cpp - Attach an SPIRV target ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GpuSPIRVAttachTarget` pass, attaching
// `#spirv.target` attributes to GPU modules.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/SPIRV/Target.h"
#include "llvm/Support/Regex.h"

namespace mlir {
#define GEN_PASS_DEF_GPUSPIRVATTACHTARGET
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::spirv;

namespace {
struct SPIRVAttachTarget
    : public impl::GpuSPIRVAttachTargetBase<SPIRVAttachTarget> {
  using Base::Base;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }
};
} // namespace

void SPIRVAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());
  if (!symbolizeVersion(spirvVersion))
    return signalPassFailure();
  if (!symbolizeClientAPI(clientApi))
    return signalPassFailure();
  if (!symbolizeVendor(deviceVendor))
    return signalPassFailure();
  if (!symbolizeDeviceType(deviceType))
    return signalPassFailure();

  Version version = symbolizeVersion(spirvVersion).value();
  SmallVector<Capability, 4> capabilities;
  SmallVector<Extension, 8> extensions;
  for (auto cap : spirvCapabilities) {
    if (symbolizeCapability(cap))
      capabilities.push_back(symbolizeCapability(cap).value());
  }
  ArrayRef<Capability> caps(capabilities);
  for (auto ext : spirvExtensions) {
    if (symbolizeCapability(ext))
      extensions.push_back(symbolizeExtension(ext).value());
  }
  ArrayRef<Extension> exts(extensions);
  VerCapExtAttr vce = VerCapExtAttr::get(version, caps, exts, &getContext());
  auto target = builder.getAttr<SPIRVTargetAttr>(
      vce, getDefaultResourceLimits(&getContext()),
      symbolizeClientAPI(clientApi).value(),
      symbolizeVendor(deviceVendor).value(),
      symbolizeDeviceType(deviceType).value(), deviceId);
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
        targets.erase(std::unique(targets.begin(), targets.end()),
                      targets.end());
        // Update the target attribute array.
        module.setTargetsAttr(builder.getArrayAttr(targets));
      }
}
