//===- ROCDLAttachTarget.cpp - Attach an ROCDL target ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GpuROCDLAttachTarget` pass, attaching
// `#rocdl.target` attributes to GPU modules.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLTargetInfo.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "llvm/Support/Regex.h"

namespace mlir {
#define GEN_PASS_DEF_GPUROCDLATTACHTARGET
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::ROCDL;

namespace {
struct ROCDLAttachTarget
    : public impl::GpuROCDLAttachTargetBase<ROCDLAttachTarget> {
  using Base::Base;

  DictionaryAttr getFlags(OpBuilder &builder, bool isWave64) const;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ROCDL::ROCDLDialect>();
  }
};
} // namespace

DictionaryAttr ROCDLAttachTarget::getFlags(OpBuilder &builder,
                                           bool isWave64) const {
  UnitAttr unitAttr = builder.getUnitAttr();
  SmallVector<NamedAttribute, 6> flags;
  auto addFlag = [&](StringRef flag) {
    flags.push_back(builder.getNamedAttr(flag, unitAttr));
  };
  if (!isWave64)
    addFlag("no_wave64");
  if (fastFlag)
    addFlag("fast");
  if (dazFlag)
    addFlag("daz");
  if (finiteOnlyFlag)
    addFlag("finite_only");
  if (unsafeMathFlag)
    addFlag("unsafe_math");
  if (!correctSqrtFlag)
    addFlag("unsafe_sqrt");
  if (!flags.empty())
    return builder.getDictionaryAttr(flags);
  return nullptr;
}

void ROCDLAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());

  // #rocdl.target feeds the TargetMachine, whose -mcpu is a bare processor
  // name. Split an `arch` into the pieces the attribute wants, mirroring
  // Clang.
  std::string resolvedTriple = triple;
  std::string resolvedChip = chip;
  std::string resolvedFeatures = features;
  // `wavesize` is the only wavefront-size control. Without an `arch` there is
  // no target to ask, so leaving it at 0 keeps the Wave64 that `#rocdl.target`
  // has always assumed.
  bool resolvedWave64 = waveSize != 32;
  // Set when `arch` was given and used so its xnack/sramecc modifiers can
  // become module flags.
  std::optional<ROCDL::TargetInfo> targetInfo;
  if (!arch.empty()) {
    std::optional<llvm::AMDGPU::TargetID> id =
        ROCDL::TargetInfo::parseTargetID(arch);
    if (!id) {
      emitError(UnknownLoc::get(&getContext()))
          << "'" << arch << "' is not a valid AMDGPU architecture";
      return signalPassFailure();
    }
    // #rocdl.target requires a chip, so a triple that names no GPU (the legacy
    // subarch-less "amdgcn-amd-amdhsa") cannot be attached.
    if (id->getGPUKind() == llvm::AMDGPU::GK_NONE) {
      emitError(UnknownLoc::get(&getContext()))
          << "'" << arch
          << "' names no GPU; a chip is required to attach a "
             "target";
      return signalPassFailure();
    }

    llvm::Triple parsed(id->getTargetTripleString());

    StringRef archName = parsed.getArchName();
    llvm::Triple::SubArchType subArch =
        llvm::AMDGPU::getSubArch(id->getGPUKind());
    if (StringRef subArchName = llvm::AMDGPU::getSubArchName(subArch);
        !subArchName.empty())
      archName = subArchName;

    // Drop the "unknown" environment part of triples since a lot of the
    // toolchain expects a 3-component form.
    resolvedTriple =
        (parsed.getEnvironment() == llvm::Triple::UnknownEnvironment
             ? llvm::Triple(archName, parsed.getVendorName(),
                            parsed.getOSName())
             : llvm::Triple(archName, parsed.getVendorName(),
                            parsed.getOSName(), parsed.getEnvironmentName()))
            .str();
    resolvedChip = llvm::AMDGPU::getArchNameAMDGCN(id->getGPUKind()).str();

    // Take the wavefront size from the target, since `wavesize`'s Wave64
    // fallback is wrong on targets like gfx10.
    FailureOr<ROCDL::TargetInfo> info =
        ROCDL::TargetInfo::get(arch, waveSize, [&] {
          return emitError(UnknownLoc::get(&getContext()));
        });
    if (failed(info))
      return signalPassFailure();
    targetInfo = *info;
    resolvedWave64 = info->getWavefrontSize() == 64;

    // Record the wavefrontsize option into the features set so that the device
    // libraries and codegen agree with each other in cases where both
    // wavefontsize32 and wavefrontsize64 are permitted options.
    if (info->supportsBothWavefrontSizes()) {
      if (!resolvedFeatures.empty())
        resolvedFeatures += ",";
      resolvedFeatures +=
          resolvedWave64 ? "+wavefrontsize64" : "+wavefrontsize32";
    }
  } else if (waveSize != 0 && waveSize != 32 && waveSize != 64) {
    emitError(UnknownLoc::get(&getContext()))
        << "wavefront size must be 32 or 64, got " << waveSize;
    return signalPassFailure();
  }

  ArrayRef<std::string> libs(linkLibs);
  SmallVector<StringRef> filesToLink(libs);
  auto target = builder.getAttr<ROCDLTargetAttr>(
      optLevel, resolvedTriple, resolvedChip, resolvedFeatures, abiVersion,
      getFlags(builder, resolvedWave64),
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
        if (targetInfo)
          targetInfo->migrateArchFeaturesToModuleFlags(module);
      }
}
