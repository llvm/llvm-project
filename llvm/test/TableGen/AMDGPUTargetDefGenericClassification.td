// RUN: split-file %s %t
// RUN: llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/valid.td \
// RUN:   | FileCheck %t/valid.td
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/missing-bug.td 2>&1 \
// RUN:   | FileCheck %t/missing-generic.td -DDIR=%t -DFEATURE=user-sgpr-init16-bug --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/missing-priority.td 2>&1 \
// RUN:   | FileCheck %t/missing-generic.td -DDIR=%t -DFEATURE=required-export-priority --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/missing-timing.td 2>&1 \
// RUN:   | FileCheck %t/missing-generic.td -DDIR=%t -DFEATURE=gfx125x-lowest-rate-wmma --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/missing-banks.td 2>&1 \
// RUN:   | FileCheck %t/missing-generic.td -DDIR=%t -DFEATURE=ldsbankcount32 --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/missing-lds-size.td 2>&1 \
// RUN:   | FileCheck %t/missing-generic.td -DDIR=%t -DFEATURE=addressablelocalmemorysize65536 --implicit-check-not="error:"
// RUN: not llvm-tblgen -gen-amdgpu-target-def -I %t -I %p/../../include -I %p/../../lib/Target/AMDGPU %t/capability.td 2>&1 \
// RUN:   | FileCheck %t/capability.td -DFILE=%t/capability.td --implicit-check-not="error:"

//--- common.td
include "AMDGPU.td"

def EmptyGPU : AMDGPUProcessorModel<"gfx999", NoSchedModel, [], [9, 9, 9]>;

//--- valid.td
include "common.td"
// Check the real generic targets, including gfx11's combined workarounds,
// gfx12-5's fixup and WMMA timing, and gfx9-4's LDS size and bank count.
// CHECK: GK_GFX9_4_GENERIC
// CHECK: GK_GFX11_GENERIC
// CHECK: GK_GFX12_5_GENERIC

//--- missing-bug.td
include "common.td"
defvar CheckedFeature = FeatureUserSGPRInit16Bug;
include "missing-generic.td"

//--- missing-priority.td
include "common.td"
defvar CheckedFeature = FeatureRequiredExportPriority;
include "missing-generic.td"

//--- missing-timing.td
include "common.td"
defvar CheckedFeature = FeatureGFX125xLowestRateWMMA;
include "missing-generic.td"

//--- missing-banks.td
include "common.td"
defvar CheckedFeature = FeatureLDSBankCount32;
include "missing-generic.td"

//--- missing-lds-size.td
include "common.td"
defvar CheckedFeature = FeatureAddressableLocalMemorySize65536;
include "missing-generic.td"

//--- missing-generic.td
// Classified backend-only features still need support from a covered GPU.
// Included files use the native path separator after the include directory.
// CHECK: [[DIR]]{{[/\\]}}missing-generic.td:[[#@LINE+1]]:1: error: generic target 'gfx9-9-generic' exposes feature '[[FEATURE]]' not supported by any covered GPU
def : AMDGPUProcessorModel<"gfx9-9-generic", NoSchedModel, [CheckedFeature], [9, 9, 0]> {
  let CoveredGPUs = [EmptyGPU];
}

//--- capability.td
include "common.td"
def WMMAGPU : AMDGPUProcessorModel<"gfx998", NoSchedModel,
    [FeatureWMMA256bInsts], [9, 9, 8]>;
// Instruction support keeps the default all-covered-GPUs rule.
// CHECK: [[FILE]]:[[#@LINE+1]]:1: error: generic target 'gfx9-9-generic' exposes feature 'wmma-256b-insts' not supported by covered GPU 'gfx999'
def : AMDGPUProcessorModel<"gfx9-9-generic", NoSchedModel, [FeatureWMMA256bInsts], [9, 9, 0]> {
  let CoveredGPUs = [WMMAGPU, EmptyGPU];
}
