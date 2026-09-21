// RUN: split-file %s %t
// RUN: llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/valid.td \
// RUN:   | FileCheck %t/valid.td
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/missing.td 2>&1 \
// RUN:   | FileCheck %t/missing.td -DFILE=%t/missing.td --implicit-check-not="error:"

//--- common.td
include "llvm/Target/Target.td"
include "AMDGPUFeatures.td"
def MyTarget : Target;

def AMDGPUFrontendVisibleFeatures {
  list<SubtargetFeature> Features = [FeatureLDSAllocGranularity512,
                                   FeatureLDSAllocGranularity1024,
                                   FeatureLDSAllocGranularity1280];
}

def GFX942 : AMDGPUProcessorModel<"gfx942", NoSchedModel,
    [FeatureAddressableLocalMemorySize65536, FeatureLDSAllocGranularity512],
    [9, 4, 2]>;
def GFX950 : AMDGPUProcessorModel<"gfx950", NoSchedModel,
    [FeatureAddressableLocalMemorySize163840, FeatureLDSAllocGranularity1280],
    [9, 5, 0]>;

//--- valid.td
include "common.td"
// Capacity and granularity can each come from a different covered GPU.
def : AMDGPUProcessorModel<"gfx9-4-generic", NoSchedModel,
    [FeatureAddressableLocalMemorySize65536, FeatureLDSAllocGranularity1280],
    [9, 4, 0]> {
  let CoveredGPUs = [GFX942, GFX950];
}
// CHECK: Triple::AMDGPUSubArch9_4, AMDGPUFeatureBitset({FEAT_LDS_ALLOC_GRANULARITY_1280}), {9, 4, 0}, [[#]], 10, 65536, 32, 0},

//--- missing.td
include "common.td"
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-4-generic' exposes feature 'lds-alloc-granularity-1024' not supported by any covered GPU
def : AMDGPUProcessorModel<"gfx9-4-generic", NoSchedModel,
    [FeatureAddressableLocalMemorySize65536, FeatureLDSAllocGranularity1024],
    [9, 4, 0]> {
  let CoveredGPUs = [GFX942, GFX950];
}
