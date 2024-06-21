; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 < %s | FileCheck --check-prefixes=CHECK %s


; CHECK-LABEL: test_unaligned_to_eight:
; CHECK: .amdhsa_kernarg_size 264
define amdgpu_kernel void @test_unaligned_to_eight(i32 %four)  {
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  store volatile ptr addrspace(4) %implicitarg.ptr, ptr addrspace(1) undef
  ret void
}


; CHECK-LABEL: test_aligned_to_eight:
; CHECK: .amdhsa_kernarg_size 264
define amdgpu_kernel void @test_aligned_to_eight(i64 %eight)  {
  %implicitarg.ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  store volatile ptr addrspace(4) %implicitarg.ptr, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: amdhsa.kernels:
; CHECK:  - .args:
; CHECK-NEXT:      - .name:           four
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_x
; CHECK-NEXT:       - .offset:         12
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_y
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_z
; CHECK:              .kernarg_segment_align: 8
; CHECK-NEXT:         .kernarg_segment_size: 264
; CHECK-LABEL:        .name:           test_unaligned_to_eight

; CHECK:  - .args:
; CHECK-NEXT:      - .name:           eight
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_x
; CHECK-NEXT:       - .offset:         12
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_y
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_z
; CHECK:              .kernarg_segment_align: 8
; CHECK-NEXT:         .kernarg_segment_size: 264
; CHECK-LABEL:        .name:           test_aligned_to_eight

declare ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
