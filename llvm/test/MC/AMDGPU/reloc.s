// RUN: llvm-mc -filetype=obj -triple amdgcn-- -mcpu=kaveri -show-encoding %s | llvm-readobj -relocations - | FileCheck %s

// CHECK: Relocations [
// CHECK: .rel.text {
// CHECK: R_AMDGPU_ABS32_LO SCRATCH_RSRC_DWORD0
// CHECK: R_AMDGPU_ABS32_LO SCRATCH_RSRC_DWORD1
// CHECK: R_AMDGPU_GOTPCREL global_var0
// CHECK: R_AMDGPU_GOTPCREL32_LO global_var1
// CHECK: R_AMDGPU_GOTPCREL32_HI global_var2
// CHECK: R_AMDGPU_REL32_LO global_var3
// CHECK: R_AMDGPU_REL32_HI global_var4
// CHECK: R_AMDGPU_ABS32 var
// CHECK: }
// CHECK: .rel.data {
// CHECK: R_AMDGPU_ABS64 temp
// CHECK: R_AMDGPU_REL64 temp
// CHECK: }
// CHECK: ]

kernel:
  s_mov_b32 s0, SCRATCH_RSRC_DWORD0
  s_mov_b32 s1, SCRATCH_RSRC_DWORD1
  s_mov_b32 s2, global_var0@GOTPCREL
  s_mov_b32 s3, global_var1@gotpcrel32@lo
  s_mov_b32 s4, global_var2@gotpcrel32@hi
  s_mov_b32 s5, global_var3@rel32@lo
  s_mov_b32 s6, global_var4@rel32@hi

.globl global_var0
.globl global_var1
.globl global_var2
.globl global_var3
.globl global_var4

.globl SCRATCH_RSRC_DWORD0

.section nonalloc, "w", @progbits
  .long var, common_var

// 8 byte relocations
	.type	ptr,@object
	.data
	.globl	ptr
	.globl	foo
	.p2align	3
ptr:
	.quad	temp
	.size	ptr, 8
foo:
	.quad	temp@rel64
	.size	foo, 8
