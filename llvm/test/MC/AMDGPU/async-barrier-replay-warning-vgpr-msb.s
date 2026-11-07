// NOTE: Do not autogenerate. Check diagnostic locations and absence across the EXEC-clear sequence.
// RUN: llvm-mc -triple=amdgpu12.50 -filetype=null < %s 2>&1 | FileCheck %s --implicit-check-not=warning:
// RUN: llvm-mc -triple=amdgpu12.51 -filetype=null < %s 2>&1 | FileCheck %s --implicit-check-not=warning:
// RUN: llvm-mc -triple=amdgpu12.50 -filetype=null -amdgpu-asm-warn-async-barrier-replay=false < %s 2>&1 | count 0

// VGPR bank changes preserve EXEC during a handwritten replay sequence.
ds_atomic_async_barrier_arrive_b64 v0
s_mov_b32 s7, exec_lo
s_mov_b32 exec_lo, 0
s_set_vgpr_msb 0x100
global_load_async_to_lds_b8 v0, v[0:1], off
s_mov_b32 exec_lo, s7
s_endpgm

// A bank change alone does not establish that EXEC is clear.
ds_atomic_async_barrier_arrive_b64 v0
// CHECK: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_set_vgpr_msb 0x100
global_load_async_to_lds_b8 v0, v[0:1], off
s_endpgm

// A bank change does not hide a subsequent write to EXEC.
ds_atomic_async_barrier_arrive_b64 v0
// CHECK: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 exec_lo, 0
s_set_vgpr_msb 0x100
s_mov_b32 exec_lo, 1
global_load_async_to_lds_b8 v0, v[0:1], off
s_endpgm

// Other intervening instructions still break the pattern.
ds_atomic_async_barrier_arrive_b64 v0
// CHECK: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 exec_lo, 0
s_nop 0
s_set_vgpr_msb 0x100
global_load_async_to_lds_b8 v0, v[0:1], off
s_endpgm
