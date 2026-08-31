//===- ROCDLTargetInfoTest.cpp - Unit tests for ROCDL::TargetInfo ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/ROCDLTargetInfo.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

namespace mlir::ROCDL {
namespace {

/// Resolves a target, collecting anything reported through emitError.
FailureOr<TargetInfo> resolve(StringRef tripleOrChip, StringRef chip,
                              StringRef features, std::string &error) {
  MLIRContext ctx;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    error = diag.str();
    return success();
  });
  return TargetInfo::get(tripleOrChip, chip, features,
                         [&] { return emitError(UnknownLoc::get(&ctx)); });
}

/// Resolves a target, asserting that it succeeded.
TargetInfo getTarget(StringRef tripleOrChip, StringRef chip = "",
                     StringRef features = "") {
  std::string error;
  FailureOr<TargetInfo> target = resolve(tripleOrChip, chip, features, error);
  EXPECT_TRUE(succeeded(target)) << "'" << tripleOrChip << "': " << error;
  return succeeded(target) ? *target : TargetInfo();
}

/// Returns the error message produced when resolving a target, or "" if it
/// unexpectedly succeeded.
std::string getTargetError(StringRef tripleOrChip, StringRef chip = "",
                           StringRef features = "") {
  std::string error;
  FailureOr<TargetInfo> target = resolve(tripleOrChip, chip, features, error);
  EXPECT_TRUE(failed(target)) << "expected '" << tripleOrChip << "' to fail";
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
  // A subarch-bearing triple identifies the GPU on its own.
  TargetInfo fromTriple = getTarget("amdgpu9.42-amd-amdhsa");
  EXPECT_EQ(fromTriple.getArchName(), "gfx942");
  EXPECT_EQ(fromTriple.getSubArch(), llvm::Triple::AMDGPUSubArch942);

  // The legacy subarch-less triple names no GPU.
  TargetInfo legacy = getTarget("amdgcn-amd-amdhsa");
  EXPECT_TRUE(legacy.isUnknown());
  EXPECT_EQ(legacy.getSubArch(), llvm::Triple::NoSubArch);
  EXPECT_FALSE(legacy.has(llvm::AMDGPU::FEAT_GFX9_INSTS));

  // A chip refines it, the way -mcpu does.
  TargetInfo withChip = getTarget("amdgcn-amd-amdhsa", "gfx942");
  EXPECT_EQ(withChip.getArchName(), "gfx942");
  EXPECT_EQ(withChip.getSubArch(), llvm::Triple::AMDGPUSubArch942);

  // A family triple plus a chip narrows to the exact GPU.
  TargetInfo family = getTarget("amdgpu9.4-amd-amdhsa", "gfx950");
  EXPECT_EQ(family.getArchName(), "gfx950");
  EXPECT_EQ(family.getSubArch(), llvm::Triple::AMDGPUSubArch950);
}

TEST(TargetInfoTest, ParseGeneric) {
  // Generic targets are representable, and carry only the features common to
  // every GPU they cover.
  TargetInfo generic = getTarget("gfx9-4-generic");
  EXPECT_TRUE(generic.isGeneric());
  EXPECT_EQ(generic.getArchName(), "gfx9-4-generic");

  // gfx9-4-generic covers gfx942 and gfx950, so it must not claim anything
  // exclusive to either.
  EXPECT_TRUE(generic.has(llvm::AMDGPU::FEAT_GFX940_INSTS));
  EXPECT_FALSE(generic.has(llvm::AMDGPU::FEAT_GFX950_INSTS));
  EXPECT_FALSE(generic.has(llvm::AMDGPU::FEAT_XF32_INSTS));
  EXPECT_TRUE(getTarget("gfx942").has(llvm::AMDGPU::FEAT_XF32_INSTS));
}

TEST(TargetInfoTest, ParseInvalid) {
  // Unlike Chipset::parse, a well-formed but nonexistent GPU is rejected.
  EXPECT_NE(getTargetError("gfx999"), "");
  EXPECT_NE(getTargetError("gfx000"), "");
  EXPECT_NE(getTargetError("navi33"), "");
  EXPECT_NE(getTargetError("sm_80"), "");
  EXPECT_NE(getTargetError("GFX942"), "");
  EXPECT_NE(getTargetError(""), "");

  // Triple parsing maps any unrecognized "amdgpu..." to NoSubArch rather than
  // reporting an error, so a typo must not be mistaken for the legacy triple.
  EXPECT_NE(getTargetError("amdgpu9.99-amd-amdhsa"), "");
  EXPECT_NE(getTargetError("amdgputypo-amd-amdhsa"), "");

  // A chip inconsistent with the triple's subarch is rejected.
  EXPECT_NE(getTargetError("amdgpu9.42-amd-amdhsa", "gfx1030"), "");
  EXPECT_NE(getTargetError("amdgcn-amd-amdhsa", "gfx999"), "");
}

TEST(TargetInfoTest, FeatureModifiers) {
  TargetInfo target = getTarget("gfx942", /*chip=*/"", "-mai-insts,+dpp");
  EXPECT_FALSE(target.has(llvm::AMDGPU::FEAT_MAI_INSTS));
  EXPECT_TRUE(target.has(llvm::AMDGPU::FEAT_DPP));
  EXPECT_TRUE(getTarget("gfx942").has(llvm::AMDGPU::FEAT_MAI_INSTS));

  EXPECT_NE(getTargetError("gfx942", "", "+not-a-feature"), "");
  EXPECT_NE(getTargetError("gfx942", "", "mai-insts"), "");
}

TEST(TargetInfoTest, WavefrontSize) {
  // Single-mode targets report their only size.
  EXPECT_EQ(getTarget("gfx90a").getWavefrontSize(), 64u);
  EXPECT_EQ(getTarget("gfx942").getWavefrontSize(), 64u);
  EXPECT_EQ(getTarget("gfx1250").getWavefrontSize(), 32u);

  // Targets supporting both default to wave32, and honour an explicit request.
  // This is the case a triple alone cannot express.
  for (StringRef gpu : {"gfx1030", "gfx1100", "gfx1200"}) {
    EXPECT_EQ(getTarget(gpu).getWavefrontSize(), 32u) << gpu;
    EXPECT_EQ(getTarget(gpu, "", "+wavefrontsize64").getWavefrontSize(), 64u)
        << gpu;
  }

  // Asking a single-mode target for the other size is an error, not a silent
  // mis-lowering.
  EXPECT_NE(getTargetError("gfx942", "", "+wavefrontsize32"), "");
  EXPECT_NE(getTargetError("gfx1250", "", "+wavefrontsize64"), "");
  EXPECT_NE(getTargetError("gfx1030", "", "+wavefrontsize32,+wavefrontsize64"),
            "");

  // An unknown target has no wavefront size.
  EXPECT_EQ(getTarget("amdgcn-amd-amdhsa").getWavefrontSize(), std::nullopt);
}

TEST(TargetInfoTest, SupportsBothWavefrontSizes) {
  // Only these leave the choice to the features; the rest pin a size, which is
  // why the pipeline's wave64 option must not be forced onto them.
  for (StringRef gpu : {"gfx1030", "gfx1100", "gfx1200"})
    EXPECT_TRUE(getTarget(gpu).supportsBothWavefrontSizes()) << gpu;
  for (StringRef gpu : {"gfx90a", "gfx942", "gfx1250"})
    EXPECT_FALSE(getTarget(gpu).supportsBothWavefrontSizes()) << gpu;

  // Naming a size does not change what the GPU is capable of.
  EXPECT_TRUE(getTarget("gfx1030", "", "+wavefrontsize64")
                  .supportsBothWavefrontSizes());
  EXPECT_FALSE(getTarget("amdgcn-amd-amdhsa").supportsBothWavefrontSizes());
}

TEST(TargetInfoTest, Fp8Formats) {
  // hasFnuzFp8 is not the negation of hasOcpFp8: a target with no fp8
  // conversions at all has neither.
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
  // A width rather than a capability bit, so that a third width would not need
  // every caller to learn a new predicate.
  for (StringRef gpu : {"gfx900", "gfx1030", "gfx1200", "gfx1201"})
    EXPECT_EQ(getTarget(gpu).getBufferResourceNumRecordsWidth(), 32u) << gpu;
  for (StringRef gpu : {"gfx1250", "gfx1251", "gfx1250-strict"})
    EXPECT_EQ(getTarget(gpu).getBufferResourceNumRecordsWidth(), 45u) << gpu;

  // Generic targets take the width of the family they cover.
  EXPECT_EQ(getTarget("gfx12-generic").getBufferResourceNumRecordsWidth(), 32u);
  EXPECT_EQ(getTarget("gfx12-5-generic").getBufferResourceNumRecordsWidth(),
            45u);

  // An unknown target has no width, so a lowering that needs one must bail
  // rather than assume the narrow case.
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

  // Generic targets report the generation of the family they cover, which
  // comparing ISA versions would get wrong.
  EXPECT_TRUE(getTarget("gfx9-4-generic").isGeneration(9));
  EXPECT_TRUE(getTarget("gfx11-generic").isGeneration(11));
  EXPECT_TRUE(getTarget("gfx12-generic").isGeneration(12));

  // An unknown target is in no generation.
  EXPECT_FALSE(getTarget("amdgcn-amd-amdhsa").isGeneration(9));
}

TEST(TargetInfoTest, DefaultIsUnknown) {
  TargetInfo target;
  EXPECT_TRUE(target.isUnknown());
  EXPECT_FALSE(target.has(llvm::AMDGPU::FEAT_GFX9_INSTS));
  EXPECT_EQ(target.getArchName(), "");
  EXPECT_EQ(target.getWavefrontSize(), std::nullopt);
}
} // namespace
} // namespace mlir::ROCDL
