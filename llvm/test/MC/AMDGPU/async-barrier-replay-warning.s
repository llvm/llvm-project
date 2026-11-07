// NOTE: Do not autogenerate. Check diagnostic locations and absence across sequence boundaries.
// RUN: llvm-mc -triple=amdgpu12.50 -filetype=null < %s 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning:
// RUN: llvm-mc -triple=amdgpu12.51 -filetype=null < %s 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning:
// RUN: echo "ds_atomic_async_barrier_arrive_b64 v0" | not llvm-mc -triple=amdgpu12.50 -filetype=null --fatal-warnings 2>&1 | FileCheck %s --check-prefix=FATAL --implicit-check-not=warning:
// RUN: llvm-mc -triple=amdgpu12.50 -filetype=null -amdgpu-asm-warn-async-barrier-replay=true < %s 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning:
// RUN: llvm-mc -triple=amdgpu12.50 -filetype=null -amdgpu-asm-warn-async-barrier-replay=false < %s 2>&1 | count 0
// RUN: llvm-mc -triple=amdgpu12.50 -filetype=null --no-warn < %s 2>&1 | count 0
// RUN: echo "ds_atomic_async_barrier_arrive_b64 v0" | llvm-mc -triple=amdgpu13.10 -mattr=+lds-barrier-arrive-atomic -filetype=null 2>&1 | count 0
// RUN: llvm-mc -triple=amdgpu12.50 -mattr=-async-barrier-arrive-replay-bug -filetype=null < %s 2>&1 | count 0
// FATAL: <stdin>:1:1: error: no recognized replay workaround after 'ds_atomic_async_barrier_arrive_b64' on this GPU

// Non-async VMEM ends recognition before the following load.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround after 'ds_atomic_async_barrier_arrive_b64' on this GPU
s_nop 0
global_load_b32 v1, v[2:3], off
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off

// The compiler's dummy-load sequence is recognized without fixing its SGPR.
ds_atomic_async_barrier_arrive_b64 v0
s_mov_b32 s7, exec_lo
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v0, v[0:1], off
s_mov_b32 exec_lo, s7
s_endpgm

// Active B8 loads and scalar-address forms are not the recognized dummy form.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
global_load_async_to_lds_b8 v4, v[2:3], off
global_load_b32 v1, v[2:3], off
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v5, s[2:3]
global_load_b32 v1, v[2:3], off

// B32 forms are not recognized, even with EXEC cleared.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b32 v4, v5, s[2:3]
global_load_b32 v1, v[2:3], off
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b32 v4, v[2:3], off
s_endpgm

// Cluster loads do not suppress the warning, even with EXEC cleared.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 m0, 0
s_mov_b32 exec_lo, 0
cluster_load_async_to_lds_b8 v4, v[2:3], off
s_endpgm

// Clearing another register or using a nonzero EXEC is not the dummy form.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 s7, 0
global_load_async_to_lds_b8 v4, v[2:3], off
s_endpgm
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 exec_lo, 1
global_load_async_to_lds_b8 v4, v[2:3], off
s_endpgm

// Intervening instructions break the immediately preceding EXEC-clear pattern.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 exec_lo, 0
s_nop 0
global_load_async_to_lds_b8 v4, v[2:3], off
s_endpgm

// Source locations do not interrupt a protective sequence.
.file 1 "async-barrier.s"
ds_atomic_async_barrier_arrive_b64 v0
s_mov_b32 s9, exec_lo
s_mov_b32 exec_lo, 0
.loc 1 10 0
global_load_async_to_lds_b8 v4, v[2:3], off
s_mov_b32 exec_lo, s9
s_endpgm

// One qualifying load can follow several arrivals.
ds_atomic_async_barrier_arrive_b64 v0
ds_atomic_async_barrier_arrive_b64 v1
s_mov_b32 s9, exec_lo
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off
s_mov_b32 exec_lo, s9
s_endpgm

// B64/B128 forms are not recognized.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b64 v4, v[2:3], off
global_load_b32 v1, v[2:3], off
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b128 v4, v[2:3], off
s_endpgm

// Recognize scratch, flat and buffer VMEM as well as global VMEM.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
scratch_load_b32 v1, v2, off
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
flat_load_b32 v1, v[2:3]
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
buffer_load_b32 v1, v2, s[4:7], s8 offen
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off

// A load beyond a branch or label does not prove protection on every path.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_branch next
next:
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off

ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_mov_b32 exec_lo, 0
another_path:
global_load_async_to_lds_b8 v4, v[2:3], off

// Calls, indirect branches, traps and exception returns end recognition.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_swap_pc_i64 s[2:3], s[0:1]
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_set_pc_i64 s[0:1]
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_trap 2
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_rfe_i64 s[0:1]
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off

// A conditional branch must not let a fall-through load hide the warning.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
s_cbranch_scc0 1
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off

// Bytes emitted by a directive interrupt recognition.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
.long 0
s_mov_b32 exec_lo, 0
global_load_async_to_lds_b8 v4, v[2:3], off

// Preserve the original location through macro expansion.
.macro arrive
  ds_atomic_async_barrier_arrive_b64 v0
// WARN: <instantiation>:1:1: warning: no recognized replay workaround
.endm
arrive
s_endpgm

// EOF also diagnoses an outstanding arrival.
ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround
