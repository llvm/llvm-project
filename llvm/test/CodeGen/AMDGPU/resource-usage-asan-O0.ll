; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=O0 %s
; RUN: llc -O2 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=O2 %s

; Test that at -O0 with sanitize_address on COV5, the assumed external call
; stack size (16384) is preserved so ASan callbacks get enough scratch space.
; Without sanitize_address, or at -O2, the assumed stack stays at 0.

declare void @extern_func()

; O0-LABEL: {{^}}kernel_with_asan:
; O0: .set kernel_with_asan.private_seg_size, {{[0-9]+}}+max(16384)
; O2-LABEL: {{^}}kernel_with_asan:
; O2: .set kernel_with_asan.private_seg_size, {{[0-9]+$}}
define amdgpu_kernel void @kernel_with_asan() sanitize_address {
  call void @extern_func()
  ret void
}

; O0-LABEL: {{^}}kernel_without_asan:
; O0: .set kernel_without_asan.private_seg_size, {{[0-9]+$}}
; O2-LABEL: {{^}}kernel_without_asan:
; O2: .set kernel_without_asan.private_seg_size, {{[0-9]+$}}
define amdgpu_kernel void @kernel_without_asan() {
  call void @extern_func()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
