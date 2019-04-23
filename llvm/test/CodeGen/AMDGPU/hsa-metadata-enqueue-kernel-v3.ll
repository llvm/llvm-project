; RUN: llc -mtriple=amdgcn-amd-amdhsa -mattr=+code-object-v3 -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mattr=+code-object-v3 -mcpu=gfx900 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s

; CHECK:              ---
; CHECK:      amdhsa.kernels:
; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           1
; CHECK-NEXT:         .type_name:      char
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:         .value_type:     i8
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:         .value_type:     i64
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:         .value_type:     i64
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:         .value_type:     i64
; CHECK-NOT:        .value_kind:    hidden_default_queue
; CHECK-NOT:        .value_kind:    hidden_completion_action
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_non_enqueue_kernel_caller
; CHECK:          .symbol:         test_non_enqueue_kernel_caller.kd
define amdgpu_kernel void @test_non_enqueue_kernel_caller(i8 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           1
; CHECK-NEXT:         .type_name:      char
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:         .value_type:     i8
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:         .value_type:     i64
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:         .value_type:     i64
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:         .value_type:     i64
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:         .value_type:     i8
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_default_queue
; CHECK-NEXT:         .value_type:     i8
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_completion_action
; CHECK-NEXT:         .value_type:     i8
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_enqueue_kernel_caller
; CHECK:          .symbol:         test_enqueue_kernel_caller.kd
define amdgpu_kernel void @test_enqueue_kernel_caller(i8 %a) #1
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:  amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 0
; CHECK-NOT:  amdhsa.printf:

attributes #0 = { "amdgpu-implicitarg-num-bytes"="48" }
attributes #1 = { "calls-enqueue-kernel" "amdgpu-implicitarg-num-bytes"="48" }

!1 = !{i32 0}
!2 = !{!"none"}
!3 = !{!"char"}
!4 = !{!""}

!opencl.ocl.version = !{!90}
!90 = !{i32 2, i32 0}

; PARSER: AMDGPU HSA Metadata Parser Test: PASS
