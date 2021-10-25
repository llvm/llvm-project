; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=3 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s

; CHECK:              ---
; CHECK:      amdhsa.kernels:
; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           1
; CHECK-NEXT:         .type_name:      char
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z

; CHECK-NOT:          .value_kind:     hidden_hostcall_buffer

; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_kernel
; CHECK:          .symbol:         test_kernel.kd

define amdgpu_kernel void @test_kernel(i8 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:  amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 0

attributes #0 = { optnone noinline "amdgpu-implicitarg-num-bytes"="48" }

!1 = !{i32 0}
!2 = !{!"none"}
!3 = !{!"char"}
!4 = !{!""}

!opencl.ocl.version = !{!90}
!90 = !{i32 2, i32 0}

; PARSER: AMDGPU HSA Metadata Parser Test: PASS
