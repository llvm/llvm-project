// RUN: llvm-mc -filetype=obj -triple amdgcn-- -mcpu=kaveri -show-encoding %s | llvm-readobj -relocations | FileCheck %s

// CHECK: Relocations [
// CHECK: R_AMDGPU_ABS32_LO SCRATCH_RSRC_DWORD0 0x0
// CHECK: R_AMDGPU_ABS32_HI SCRATCH_RSRC_DWORD1 0x0
// CHECK: R_AMDGPU_GOTPCREL global_var 0x0
// CHECK: R_AMDGPU_ABS32 var 0x0
// CHECK: ]

kernel:
  s_mov_b32 s0, SCRATCH_RSRC_DWORD0
  s_mov_b32 s1, SCRATCH_RSRC_DWORD1
  s_mov_b32 s2, global_var@GOTPCREL

.globl global_var

.globl SCRATCH_RSRC_DWORD0

.section nonalloc, "w", @progbits
  .long var, common_var
