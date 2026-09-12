// RUN: split-file %s %t
// RUN: llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/lds-valid.td \
// RUN:   | FileCheck %t/lds-valid.td
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/lds-conflicting-member.td 2>&1 \
// RUN:   | FileCheck %t/lds-conflicting-member.td -DFILE=%t/lds-conflicting-member.td --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/lds-invalid.td 2>&1 \
// RUN:   | FileCheck %t/lds-invalid.td -DFILE=%t/lds-invalid.td --implicit-check-not="error:"

//--- lds-common.td
include "llvm/Target/Target.td"
include "AMDGPUFeatures.td"
def MyTarget : Target;

//--- lds-valid.td
include "lds-common.td"
// Exercise the actual addressable LDS features and their declared policy.
def GFX942 : AMDGPUProcessorModel<"gfx942", NoSchedModel,
    [FeatureAddressableLocalMemorySize65536], [9, 4, 2]>;
def GFX950 : AMDGPUProcessorModel<"gfx950", NoSchedModel,
    [FeatureAddressableLocalMemorySize163840], [9, 5, 0]>;
def : AMDGPUProcessorModel<"gfx9-4-generic", NoSchedModel,
    [FeatureAddressableLocalMemorySize65536], [9, 4, 0]> {
  let CoveredGPUs = [GFX942, GFX950];
}
// CHECK: {[[#]], Triple::AMDGPUSubArch942, {{.*}}, {9, 4, 2}, [[#]], 10, 65536, 32, 0},
// CHECK: {[[#]], Triple::AMDGPUSubArch950, {{.*}}, {9, 5, 0}, [[#]], 10, 163840, 32, 0},
// CHECK: {[[#]], Triple::AMDGPUSubArch9_4, {{.*}}, {9, 4, 0}, [[#]], 10, 65536, 32, 0},

//--- lds-conflicting-member.td
include "lds-common.td"
// The backend's LDS helper checks 32768 first. Taking the maximum here would
// incorrectly accept a 65536-byte generic target covering this ambiguous GPU.
// CHECK: [[FILE]]:[[#@LINE+1]]:5: error: GPU 'gfx900' gets conflicting 'AddressableLocalMemorySize' values from 'addressablelocalmemorysize65536' and 'addressablelocalmemorysize32768'
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel,
    [FeatureAddressableLocalMemorySize32768,
     FeatureAddressableLocalMemorySize65536], [9, 0, 0]>;
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel,
    [FeatureAddressableLocalMemorySize65536], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}

//--- lds-invalid.td
include "lds-common.td"
def GFX900 : AMDGPUProcessorModel<"gfx900", NoSchedModel,
    [FeatureAddressableLocalMemorySize32768], [9, 0, 0]>;
// The real policy must reject a generic target claiming more LDS than a member.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-generic' has 'AddressableLocalMemorySize' value 65536, which must be at most the value 32768 of covered GPU 'gfx900'
def : AMDGPUProcessorModel<"gfx9-generic", NoSchedModel,
    [FeatureAddressableLocalMemorySize65536], [9, 0, 0]> {
  let CoveredGPUs = [GFX900];
}
