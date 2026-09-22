// RUN: llvm-tblgen -gen-amdgpu-target-def -I %p/../../include -I %p/../../lib/Target/AMDGPU %s \
// RUN:   | FileCheck %s

// Parse the real target only once: loading all AMDGPU instruction definitions
// for each negative case can exceed the test timeout. The lightweight
// AMDGPUTargetDefGenericFeatures.td test covers the validation diagnostics.
include "AMDGPU.td"

// Check the classification of the real workarounds, fixup, timing, and numeric
// features, including features that are not exposed in the frontend bitset.
foreach Feature = [FeatureUserSGPRInit16Bug,
                   FeatureMADIntraFwdBug,
                   FeatureMSAALoadDstSelBug,
                   FeaturePrivEnabledTrap2NopBug,
                   FeatureVALUTransUseHazard,
                   FeatureRequiredExportPriority,
                   FeatureSetregVGPRMSBFixup,
                   FeatureGFX125xLowestRateWMMA,
                   FeatureLDSBankCount16,
                   FeatureLDSBankCount32,
                   FeatureLDSBankCount64,
                   FeatureAddressableLocalMemorySize32768,
                   FeatureAddressableLocalMemorySize65536,
                   FeatureAddressableLocalMemorySize163840,
                   FeatureAddressableLocalMemorySize196608,
                   FeatureAddressableLocalMemorySize327680] in
  assert !isa<AMDGPUGenericAnyFeature>(Feature),
         Feature.Name # " must use the any-covered-GPU rule";

// Instruction capabilities keep the all-covered-GPUs rule. RequiresCOV6 is a
// generic-target requirement, not a feature supplied by a covered GPU.
foreach Feature = [FeatureWMMA256bInsts, FeatureRequiresCOV6] in
  assert !not(!isa<AMDGPUGenericAnyFeature>(Feature)),
         Feature.Name # " must not use the any-covered-GPU rule";

// Also validate the real generic targets, including gfx11's combined
// workarounds, gfx12-5's fixup and WMMA timing, and gfx9-4's LDS properties.
// CHECK: GK_GFX9_4_GENERIC
// CHECK: GK_GFX11_GENERIC
// CHECK: GK_GFX12_5_GENERIC
