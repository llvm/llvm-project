//===- SPIRVAttachTarget.cpp - Attach an SPIR-V target --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GPUSPIRVAttachTarget` pass, attaching
// `#spirv.target_env` attributes to GPU modules.
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
  auto versionSymbol = symbolizeVersion(spirvVersion);
  if (!versionSymbol)
    return signalPassFailure();
  auto apiSymbol = symbolizeClientAPI(clientApi);
  if (!apiSymbol)
    return signalPassFailure();
  auto vendorSymbol = symbolizeVendor(deviceVendor);
  if (!vendorSymbol)
    return signalPassFailure();
  auto deviceTypeSymbol = symbolizeDeviceType(deviceType);
  if (!deviceTypeSymbol)
    return signalPassFailure();
  // Set the default device ID if none was given
  if (!deviceId.hasValue())
    deviceId = mlir::spirv::TargetEnvAttr::kUnknownDeviceID;

  Version version = versionSymbol.value();
  SmallVector<Capability, 4> capabilities;
  SmallVector<Extension, 8> extensions;
  for (const auto &cap : spirvCapabilities) {
    auto capSymbol = symbolizeCapability(cap);
    if (capSymbol)
      capabilities.push_back(capSymbol.value());
  }
  ArrayRef<Capability> caps(capabilities);
  for (const auto &ext : spirvExtensions) {
    auto extSymbol = symbolizeExtension(ext);
    if (extSymbol)
      extensions.push_back(extSymbol.value());
  }
  ArrayRef<Extension> exts(extensions);
  VerCapExtAttr vce = VerCapExtAttr::get(version, caps, exts, &getContext());
  auto target = TargetEnvAttr::get(vce, getDefaultResourceLimits(&getContext()),
                                   apiSymbol.value(), vendorSymbol.value(),
                                   deviceTypeSymbol.value(), deviceId);
  llvm::Regex matcher(moduleMatcher);
  getOperation()->walk([&](gpu::GPUModuleOp gpuModule) {
    // Check if the name of the module matches.
    if (!moduleMatcher.empty() && !matcher.match(gpuModule.getName()))
      return;
    // Create the target array.
    SmallVector<Attribute> targets;
    if (std::optional<ArrayAttr> attrs = gpuModule.getTargets())
      targets.append(attrs->getValue().begin(), attrs->getValue().end());
    targets.push_back(target);
    // Remove any duplicate targets.
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    // Update the target attribute array.
    gpuModule.setTargetsAttr(builder.getArrayAttr(targets));
  });
}
