// NOTE: Do not autogenerate. Check target-ID XNACK settings and warning locations.
// RUN: split-file %s %t
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=null < %t/on.s 2>&1 | FileCheck %s --check-prefix=ON --implicit-check-not=warning:
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=null < %t/off.s 2>&1 | count 0
// RUN: sed 's/12.50/12.51/g; s/gfx1250/gfx1251/g' %t/on.s | llvm-mc -triple=amdgpu12.51-amd-amdhsa -filetype=null 2>&1 | FileCheck %s --check-prefix=ON --implicit-check-not=warning:
// RUN: sed 's/12.50/12.51/g; s/gfx1250/gfx1251/g' %t/off.s | llvm-mc -triple=amdgpu12.51-amd-amdhsa -filetype=null 2>&1 | count 0
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=null < %t/any.s 2>&1 | FileCheck %s --check-prefix=ANY --implicit-check-not=warning:
// RUN: llvm-mc -triple=amdgpu12.51-amd-amdhsa -filetype=null < %t/any.s 2>&1 | FileCheck %s --check-prefix=ANY --implicit-check-not=warning:
// RUN: llvm-mc -triple=amdgpu12.50s-amd-amdhsa -filetype=null < %t/any.s 2>&1 | FileCheck %s --check-prefix=ANY --implicit-check-not=warning:
// RUN: llvm-mc -triple=amdgpu12.5-amd-amdhsa -filetype=null < %t/any.s 2>&1 | FileCheck %s --check-prefix=ANY --implicit-check-not=warning:

// ON: <stdin>:2:1: warning: no recognized replay workaround
// ANY: <stdin>:1:1: warning: no recognized replay workaround

//--- on.s
.amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250:xnack+"
ds_atomic_async_barrier_arrive_b64 v0
s_endpgm

//--- off.s
.amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250:xnack-"
ds_atomic_async_barrier_arrive_b64 v0
s_endpgm

//--- any.s
ds_atomic_async_barrier_arrive_b64 v0
s_endpgm
