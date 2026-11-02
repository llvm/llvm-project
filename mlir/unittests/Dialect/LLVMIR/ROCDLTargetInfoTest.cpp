//===- ROCDLTargetInfoTest.cpp - Unit tests for ROCDL::TargetInfo ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/ROCDLTargetInfo.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

#include <optional>

using namespace mlir;
using namespace mlir::ROCDL;

namespace {

FailureOr<TargetInfo> resolve(StringRef arch, unsigned waveSize,
                              std::string &error) {
  MLIRContext ctx;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    error = diag.str();
    return success();
  });
  return TargetInfo::get(arch, waveSize,
                         [&] { return emitError(UnknownLoc::get(&ctx)); });
}

TargetInfo getTarget(StringRef arch, unsigned waveSize = 0) {
  std::string error;
  FailureOr<TargetInfo> target = resolve(arch, waveSize, error);
  EXPECT_TRUE(succeeded(target)) << "'" << arch << "': " << error;
  return succeeded(target) ? *target : TargetInfo();
}

std::string getTargetError(StringRef arch, unsigned waveSize = 0) {
  std::string error;
  FailureOr<TargetInfo> target = resolve(arch, waveSize, error);
  EXPECT_TRUE(failed(target)) << "expected '" << arch << "' to fail";
  return error;
}

TEST(TargetInfoTest, ParseGPUName) {
  TargetInfo gfx942 = getTarget("gfx942");
  EXPECT_EQ(gfx942.getArchName(), "gfx942");
  EXPECT_EQ(gfx942.getSubArch(), llvm::Triple::AMDGPUSubArch942);
  EXPECT_FALSE(gfx942.isUnknown());
  EXPECT_FALSE(gfx942.isGeneric());

  llvm::AMDGPU::IsaVersion version = gfx942.getIsaVersion();
  EXPECT_EQ(version.Major, 9u);
  EXPECT_EQ(version.Minor, 4u);
  EXPECT_EQ(version.Stepping, 2u);
}

TEST(TargetInfoTest, ParseTriple) {
  TargetInfo fromTriple = getTarget("amdgpu9.42-amd-amdhsa");
  EXPECT_EQ(fromTriple.getArchName(), "gfx942");
  EXPECT_EQ(fromTriple.getSubArch(), llvm::Triple::AMDGPUSubArch942);

  TargetInfo legacy = getTarget("amdgcn-amd-amdhsa");
  EXPECT_TRUE(legacy.isUnknown());
  EXPECT_EQ(legacy.getSubArch(), llvm::Triple::NoSubArch);
  EXPECT_FALSE(legacy.has(llvm::AMDGPU::FEAT_GFX9_INSTS));

  TargetInfo full = getTarget("amdgcn-amd-amdhsa--gfx942");
  EXPECT_EQ(full.getArchName(), "gfx942");
  EXPECT_EQ(full.getSubArch(), llvm::Triple::AMDGPUSubArch942);

  EXPECT_EQ(getTarget("amdgcn-amd-amdhsa-unknown-gfx942").getArchName(),
            "gfx942");

  TargetInfo family = getTarget("amdgpu9.4-amd-amdhsa--gfx950");
  EXPECT_EQ(family.getArchName(), "gfx950");
  EXPECT_EQ(family.getSubArch(), llvm::Triple::AMDGPUSubArch950);
}

TEST(TargetInfoTest, ParseGeneric) {
  TargetInfo generic = getTarget("gfx9-4-generic");
  EXPECT_TRUE(generic.isGeneric());
  EXPECT_EQ(generic.getArchName(), "gfx9-4-generic");

  EXPECT_TRUE(generic.has(llvm::AMDGPU::FEAT_GFX940_INSTS));
  EXPECT_FALSE(generic.has(llvm::AMDGPU::FEAT_GFX950_INSTS));
  EXPECT_FALSE(generic.has(llvm::AMDGPU::FEAT_XF32_INSTS));
  EXPECT_TRUE(getTarget("gfx942").has(llvm::AMDGPU::FEAT_XF32_INSTS));
}

TEST(TargetInfoTest, ParseInvalid) {
  EXPECT_NE(getTargetError("gfx999"), "");
  EXPECT_NE(getTargetError("gfx000"), "");
  EXPECT_NE(getTargetError("navi33"), "");
  EXPECT_NE(getTargetError("sm_80"), "");
  EXPECT_NE(getTargetError("GFX942"), "");
  EXPECT_NE(getTargetError(""), "");

  EXPECT_NE(getTargetError("amdgpu9.99-amd-amdhsa"), "");
  EXPECT_NE(getTargetError("amdgputypo-amd-amdhsa"), "");

  EXPECT_NE(getTargetError("amdgpu9.42-amd-amdhsa--gfx1030"), "");
  EXPECT_NE(getTargetError("amdgcn-amd-amdhsa--gfx999"), "");
}

TEST(TargetInfoTest, TargetIDModifiers) {
  using llvm::AMDGPU::TargetIDSetting;

  EXPECT_EQ(getTarget("gfx90a").getXnackSetting(), TargetIDSetting::Any);
  EXPECT_EQ(getTarget("gfx90a:xnack+").getXnackSetting(), TargetIDSetting::On);
  EXPECT_EQ(getTarget("gfx90a:xnack-").getXnackSetting(), TargetIDSetting::Off);
  EXPECT_EQ(getTarget("gfx90a:sramecc+").getSramEccSetting(),
            TargetIDSetting::On);
  EXPECT_EQ(getTarget("gfx90a:sramecc-:xnack+").getSramEccSetting(),
            TargetIDSetting::Off);

  EXPECT_EQ(getTarget("gfx600").getXnackSetting(),
            TargetIDSetting::Unsupported);
  EXPECT_NE(getTargetError("gfx600:xnack+"), "");

  EXPECT_NE(getTargetError("gfx908:xnack"), "");
  EXPECT_NE(getTargetError("gfx942:not-a-feature+"), "");
  EXPECT_NE(getTargetError("gfx1030:wavefrontsize64+"), "");

  EXPECT_TRUE(getTarget("gfx90a:xnack+").has(llvm::AMDGPU::FEAT_MAI_INSTS));
}

TEST(TargetInfoTest, WavefrontSize) {
  EXPECT_EQ(getTarget("gfx90a").getWavefrontSize(), 64u);
  EXPECT_EQ(getTarget("gfx942").getWavefrontSize(), 64u);
  EXPECT_EQ(getTarget("gfx1250").getWavefrontSize(), 32u);

  for (StringRef gpu : {"gfx1030", "gfx1100", "gfx1200"}) {
    EXPECT_EQ(getTarget(gpu).getWavefrontSize(), 32u) << gpu;
    EXPECT_EQ(getTarget(gpu, /*waveSize=*/64).getWavefrontSize(), 64u) << gpu;
    EXPECT_EQ(getTarget(gpu, /*waveSize=*/32).getWavefrontSize(), 32u) << gpu;
  }

  EXPECT_EQ(getTarget("gfx942", /*waveSize=*/64).getWavefrontSize(), 64u);
  EXPECT_EQ(getTarget("gfx1250", /*waveSize=*/32).getWavefrontSize(), 32u);

  EXPECT_NE(getTargetError("gfx942", /*waveSize=*/32), "");
  EXPECT_NE(getTargetError("gfx1250", /*waveSize=*/64), "");

  EXPECT_NE(getTargetError("gfx1030", /*waveSize=*/17), "");
  EXPECT_NE(getTargetError("gfx1030", /*waveSize=*/128), "");

  EXPECT_EQ(getTarget("amdgcn-amd-amdhsa").getWavefrontSize(), std::nullopt);
  EXPECT_NE(getTargetError("amdgcn-amd-amdhsa", /*waveSize=*/17), "");
}

TEST(TargetInfoTest, SupportsBothWavefrontSizes) {
  for (StringRef gpu : {"gfx1030", "gfx1100", "gfx1200"})
    EXPECT_TRUE(getTarget(gpu).supportsBothWavefrontSizes()) << gpu;
  for (StringRef gpu : {"gfx90a", "gfx942", "gfx1250"})
    EXPECT_FALSE(getTarget(gpu).supportsBothWavefrontSizes()) << gpu;

  EXPECT_TRUE(
      getTarget("gfx1030", /*waveSize=*/64).supportsBothWavefrontSizes());
  EXPECT_FALSE(getTarget("amdgcn-amd-amdhsa").supportsBothWavefrontSizes());
}

TEST(TargetInfoTest, Fp8Formats) {
  EXPECT_TRUE(getTarget("gfx942").hasFnuzFp8());
  EXPECT_FALSE(getTarget("gfx942").hasOcpFp8());
  for (StringRef gpu : {"gfx950", "gfx1170", "gfx1200"}) {
    EXPECT_FALSE(getTarget(gpu).hasFnuzFp8()) << gpu;
    EXPECT_TRUE(getTarget(gpu).hasOcpFp8()) << gpu;
  }
  for (StringRef gpu : {"gfx908", "gfx90a", "gfx900"}) {
    EXPECT_FALSE(getTarget(gpu).hasFnuzFp8()) << gpu;
    EXPECT_FALSE(getTarget(gpu).hasOcpFp8()) << gpu;
  }
}

TEST(TargetInfoTest, BufferResourceNumRecordsWidth) {
  for (StringRef gpu : {"gfx900", "gfx1030", "gfx1200", "gfx1201"})
    EXPECT_EQ(getTarget(gpu).getBufferResourceNumRecordsWidth(), 32u) << gpu;
  for (StringRef gpu : {"gfx1250", "gfx1251", "gfx1250-strict"})
    EXPECT_EQ(getTarget(gpu).getBufferResourceNumRecordsWidth(), 45u) << gpu;

  EXPECT_EQ(getTarget("gfx12-generic").getBufferResourceNumRecordsWidth(), 32u);
  EXPECT_EQ(getTarget("gfx12-5-generic").getBufferResourceNumRecordsWidth(),
            45u);

  EXPECT_EQ(getTarget("amdgcn-amd-amdhsa").getBufferResourceNumRecordsWidth(),
            std::nullopt);
  EXPECT_EQ(TargetInfo().getBufferResourceNumRecordsWidth(), std::nullopt);
}

TEST(TargetInfoTest, MaxAddressableLocalMemorySize) {
  EXPECT_EQ(getTarget("gfx900").getMaxAddressableLocalMemorySize(), 65536u);
  EXPECT_EQ(getTarget("gfx1030").getMaxAddressableLocalMemorySize(), 65536u);
  EXPECT_EQ(getTarget("gfx950").getMaxAddressableLocalMemorySize(), 163840u);
  EXPECT_EQ(getTarget("gfx1250").getMaxAddressableLocalMemorySize(), 327680u);

  EXPECT_EQ(getTarget("amdgcn-amd-amdhsa").getMaxAddressableLocalMemorySize(),
            std::nullopt);
}

TEST(TargetInfoTest, RegisterAndLDSProperties) {
  for (StringRef gpu : {"gfx900", "gfx90a", "gfx942", "gfx1030", "gfx1250"}) {
    TargetInfo target = getTarget(gpu);
    llvm::AMDGPU::GPUKind kind = target.getGPUKind();
    EXPECT_EQ(target.getTotalNumSGPRs(), llvm::AMDGPU::getTotalNumSGPRs(kind))
        << gpu;
    EXPECT_EQ(target.getAddressableNumSGPRs(),
              llvm::AMDGPU::getAddressableNumSGPRs(kind))
        << gpu;
    EXPECT_EQ(target.getSGPRAllocGranule(),
              llvm::AMDGPU::getSGPRAllocGranule(kind))
        << gpu;
    EXPECT_EQ(target.getLDSBankCount(), llvm::AMDGPU::getLDSBankCount(kind))
        << gpu;
    EXPECT_EQ(target.getMaxWavesPerEU(), llvm::AMDGPU::getMaxWavesPerEU(kind))
        << gpu;
  }

  TargetInfo wave32 = getTarget("gfx1030", /*waveSize=*/32);
  TargetInfo wave64 = getTarget("gfx1030", /*waveSize=*/64);
  EXPECT_EQ(wave32.getVGPRAllocGranule(),
            llvm::AMDGPU::getVGPRAllocGranule(wave32.getGPUKind(),
                                              /*IsWave32=*/true));
  EXPECT_EQ(wave64.getVGPRAllocGranule(),
            llvm::AMDGPU::getVGPRAllocGranule(wave64.getGPUKind(),
                                              /*IsWave32=*/false));

  TargetInfo unknown = getTarget("amdgcn-amd-amdhsa");
  EXPECT_EQ(unknown.getTotalNumSGPRs(), std::nullopt);
  EXPECT_EQ(unknown.getAddressableNumSGPRs(), std::nullopt);
  EXPECT_EQ(unknown.getSGPRAllocGranule(), std::nullopt);
  EXPECT_EQ(unknown.getVGPRAllocGranule(), std::nullopt);
  EXPECT_EQ(unknown.getLDSBankCount(), std::nullopt);
  EXPECT_EQ(unknown.getMaxWavesPerEU(), std::nullopt);
  EXPECT_EQ(TargetInfo().getLDSBankCount(), std::nullopt);
}

TEST(TargetInfoTest, Generation) {
  EXPECT_TRUE(getTarget("gfx900").isGeneration(9));
  EXPECT_TRUE(getTarget("gfx942").isGeneration(9));
  EXPECT_TRUE(getTarget("gfx950").isGeneration(9));
  EXPECT_FALSE(getTarget("gfx942").isGeneration(10));
  EXPECT_FALSE(getTarget("gfx942").isGeneration(8));

  EXPECT_TRUE(getTarget("gfx1010").isGeneration(10));
  EXPECT_TRUE(getTarget("gfx1030").isGeneration(10));
  EXPECT_TRUE(getTarget("gfx1100").isGeneration(11));
  EXPECT_TRUE(getTarget("gfx1200").isGeneration(12));
  EXPECT_TRUE(getTarget("gfx1250").isGeneration(12));
  EXPECT_TRUE(getTarget("gfx803").isGeneration(8));
  EXPECT_TRUE(getTarget("gfx700").isGeneration(7));
  EXPECT_TRUE(getTarget("gfx600").isGeneration(6));

  EXPECT_TRUE(getTarget("gfx9-4-generic").isGeneration(9));
  EXPECT_TRUE(getTarget("gfx11-generic").isGeneration(11));
  EXPECT_TRUE(getTarget("gfx12-generic").isGeneration(12));

  EXPECT_FALSE(getTarget("amdgcn-amd-amdhsa").isGeneration(9));
}

/// The two module attributes `migrateArchFeaturesToModuleFlags` writes, read
/// back as values so that they outlive the context they were created in.
struct MigratedSettings {
  std::optional<bool> xnack;
  std::optional<bool> sramecc;
};

MigratedSettings
migrate(StringRef arch,
        function_ref<void(ROCDLDialect *, Operation *)> preset = nullptr) {
  MLIRContext ctx;
  auto *dialect = ctx.getOrLoadDialect<ROCDLDialect>();
  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&ctx));
  if (preset)
    preset(dialect, *module);
  getTarget(arch).migrateArchFeaturesToModuleFlags(*module);
  auto read = [](BoolAttr attr) -> std::optional<bool> {
    if (!attr)
      return std::nullopt;
    return attr.getValue();
  };
  return {read(dialect->getXnackAttrHelper().getAttr(*module)),
          read(dialect->getSrameccAttrHelper().getAttr(*module))};
}

TEST(TargetInfoTest, MigrateArchFeaturesToModuleFlags) {
  EXPECT_EQ(migrate("gfx90a:xnack+").xnack, true);
  EXPECT_EQ(migrate("gfx90a:xnack-").xnack, false);
  EXPECT_EQ(migrate("gfx90a:sramecc-").sramecc, false);

  MigratedSettings both = migrate("gfx90a:sramecc+:xnack-");
  EXPECT_EQ(both.sramecc, true);
  EXPECT_EQ(both.xnack, false);

  EXPECT_EQ(migrate("gfx90a:xnack+").sramecc, std::nullopt);

  MigratedSettings any = migrate("gfx90a");
  EXPECT_EQ(any.xnack, std::nullopt);
  EXPECT_EQ(any.sramecc, std::nullopt);
  EXPECT_EQ(migrate("gfx600").xnack, std::nullopt);

  auto presetXnackTrue = [](ROCDLDialect *dialect, Operation *op) {
    dialect->getXnackAttrHelper().setAttr(
        op, BoolAttr::get(op->getContext(), true));
  };
  EXPECT_EQ(migrate("gfx90a", presetXnackTrue).xnack, true);
  EXPECT_EQ(migrate("gfx90a:xnack-", presetXnackTrue).xnack, false);
}

TEST(TargetInfoTest, ResolveArchOption) {
  EXPECT_EQ(resolveArchOption("gfx942", ""), "gfx942");
  EXPECT_EQ(resolveArchOption("gfx942", "gfx90a"), "gfx942");

  EXPECT_EQ(resolveArchOption("invalid", "gfx90a"), "gfx90a");
  EXPECT_EQ(resolveArchOption("", "gfx90a"), "gfx90a");

  EXPECT_EQ(resolveArchOption("invalid", "gfx90a:xnack+"), "gfx90a:xnack+");

  EXPECT_EQ(resolveArchOption("invalid", ""), "invalid");
  EXPECT_EQ(resolveArchOption("", ""), "");
}

TEST(TargetInfoTest, DefaultIsUnknown) {
  TargetInfo target;
  EXPECT_TRUE(target.isUnknown());
  EXPECT_FALSE(target.has(llvm::AMDGPU::FEAT_GFX9_INSTS));
  EXPECT_EQ(target.getArchName(), "");
  EXPECT_EQ(target.getWavefrontSize(), std::nullopt);
}
} // namespace
