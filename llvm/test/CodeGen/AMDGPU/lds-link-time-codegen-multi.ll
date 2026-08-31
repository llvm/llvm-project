; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-lower-module-lds=0 -amdgpu-enable-object-linking < %s | FileCheck -check-prefixes=ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-lower-module-lds=0 -amdgpu-enable-object-linking -filetype=obj < %s | llvm-readobj -r --syms - | FileCheck -check-prefixes=ELF %s

; Multiple external LDS declarations with different sizes and alignments,
; accessed by two kernels. Tests that each LDS decl gets its own
; SHN_AMDGPU_LDS symbol and .amdgpu_lds directive. The kernel descriptor
; uses literal 0 for group_segment_fixed_size because the linker patches
; it via direct binary patching.

@__amdgpu_lds.func_a = external addrspace(3) global [256 x i8], align 16
@__amdgpu_lds.func_b = external addrspace(3) global [128 x i8], align 4
@__amdgpu_lds.kernel_one = external addrspace(3) global [64 x i8], align 8

; --- Assembly checks (use DAG for unordered matches) ---
; Each LDS declaration should produce a .amdgpu_lds directive.
; ASM-DAG: .amdgpu_lds __amdgpu_lds.func_a, 256, 16
; ASM-DAG: .amdgpu_lds __amdgpu_lds.func_b, 128, 4
; ASM-DAG: .amdgpu_lds __amdgpu_lds.kernel_one, 64, 8

; HSA metadata should have group_segment_fixed_size = 0.
; ASM-DAG: .group_segment_fixed_size: 0

; --- ELF checks ---
; All three SHN_AMDGPU_LDS symbols should be present.
; ELF-DAG: Name: __amdgpu_lds.func_a
; ELF-DAG: Name: __amdgpu_lds.func_b
; ELF-DAG: Name: __amdgpu_lds.kernel_one

; LDS access relocations in .text.
; ELF-DAG: R_AMDGPU_ABS32_LO __amdgpu_lds.func_a
; ELF-DAG: R_AMDGPU_ABS32_LO __amdgpu_lds.func_b
; ELF-DAG: R_AMDGPU_ABS32_LO __amdgpu_lds.kernel_one

define void @func_a(i32 %idx) {
  %gep = getelementptr [256 x i8], ptr addrspace(3) @__amdgpu_lds.func_a, i32 0, i32 %idx
  store i8 1, ptr addrspace(3) %gep
  ret void
}

define void @func_b(i32 %idx) {
  %gep = getelementptr [128 x i8], ptr addrspace(3) @__amdgpu_lds.func_b, i32 0, i32 %idx
  store i8 2, ptr addrspace(3) %gep
  ret void
}

define amdgpu_kernel void @kernel_one(i32 %idx) {
  %gep = getelementptr [64 x i8], ptr addrspace(3) @__amdgpu_lds.kernel_one, i32 0, i32 %idx
  store i8 3, ptr addrspace(3) %gep
  call void @func_a(i32 %idx)
  ret void
}

define amdgpu_kernel void @kernel_two(i32 %idx) {
  call void @func_b(i32 %idx)
  ret void
}
