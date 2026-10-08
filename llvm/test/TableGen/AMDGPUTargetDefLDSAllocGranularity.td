// RUN: split-file %s %t
// RUN: llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/valid.td \
// RUN:   | FileCheck %t/valid.td
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/missing.td 2>&1 \
// RUN:   | FileCheck %t/missing.td -DFILE=%t/missing.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/missing-encoding.td 2>&1 \
// RUN:   | FileCheck %t/missing-encoding.td -DFILE=%t/missing-encoding.td --implicit-check-not="error:"

//--- common.td
include "llvm/Target/Target.td"
include "AMDGPUFeatures.td"
def MyTarget : Target;

def AMDGPUFrontendVisibleFeatures {
  list<SubtargetFeature> Features = [FeatureLDSAllocGranularity512,
                                   FeatureLDSAllocGranularity1024,
                                   FeatureLDSAllocGranularity1280,
                                   FeatureLDSEncodingGranularity512,
                                   FeatureLDSEncodingGranularity1024,
                                   FeatureLDSEncodingGranularity1280];
}

def GFX942 : AMDGPUProcessorModel<"gfx942", NoSchedModel,
    [FeatureAddressableLocalMemorySize65536, FeatureLDSAllocGranularity512,
     FeatureLDSEncodingGranularity512],
    [9, 4, 2]>;
def GFX950 : AMDGPUProcessorModel<"gfx950", NoSchedModel,
    [FeatureAddressableLocalMemorySize163840, FeatureLDSAllocGranularity1280,
     FeatureLDSEncodingGranularity1280],
    [9, 5, 0]>;

//--- valid.td
include "common.td"
// Capacity and granularity can each come from a different covered GPU.
def : AMDGPUProcessorModel<"gfx9-4-generic", NoSchedModel,
    [FeatureAddressableLocalMemorySize65536, FeatureLDSAllocGranularity1280,
     FeatureLDSEncodingGranularity1280],
    [9, 4, 0]> {
  let CoveredGPUs = [GFX942, GFX950];
}
// CHECK: Triple::AMDGPUSubArch9_4, AMDGPUFeatureBitset({FEAT_LDS_ALLOC_GRANULARITY_1280, FEAT_LDS_ENCODING_GRANULARITY_1280}), {9, 4, 0}, [[#]], 10, 65536, 32, 0},

// Allocation and encoding use different units on gfx10.3, including its generic.
def GFX1030 : AMDGPUProcessorModel<"gfx1030", NoSchedModel,
    [FeatureLDSAllocGranularity1024, FeatureLDSEncodingGranularity512],
    [10, 3, 0]>;
def : AMDGPUProcessorModel<"gfx10-3-generic", NoSchedModel,
    [FeatureLDSAllocGranularity1024, FeatureLDSEncodingGranularity512],
    [10, 3, 0]> {
  let CoveredGPUs = [GFX1030];
}
// CHECK: Triple::AMDGPUSubArch10_3, AMDGPUFeatureBitset({FEAT_LDS_ALLOC_GRANULARITY_1024, FEAT_LDS_ENCODING_GRANULARITY_512})

//--- missing.td
include "common.td"
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-4-generic' exposes feature 'lds-alloc-granularity-1024' not supported by any covered GPU
def : AMDGPUProcessorModel<"gfx9-4-generic", NoSchedModel,
    [FeatureAddressableLocalMemorySize65536, FeatureLDSAllocGranularity1024],
    [9, 4, 0]> {
  let CoveredGPUs = [GFX942, GFX950];
}

//--- missing-encoding.td
include "common.td"
// An allocation feature does not provide support for an encoding feature.
def GFX1030 : AMDGPUProcessorModel<"gfx1030", NoSchedModel,
    [FeatureLDSAllocGranularity1024, FeatureLDSEncodingGranularity512],
    [10, 3, 0]>;
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx10-3-generic' exposes feature 'lds-encoding-granularity-1024' not supported by any covered GPU
def : AMDGPUProcessorModel<"gfx10-3-generic", NoSchedModel,
    [FeatureLDSAllocGranularity1024, FeatureLDSEncodingGranularity1024],
    [10, 3, 0]> {
  let CoveredGPUs = [GFX1030];
}
