; RUN: llc -mattr=+code-object-v3 -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck %s

; CHECK:        - .args:           
; CHECK-NEXT:       - .access:         read_only
; CHECK-NEXT:         .address_space:  global
; CHECK-NEXT:         .is_const:       true
; CHECK-NEXT:         .is_restrict:    true
; CHECK-NEXT:         .name:           in
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      'float*'
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:         .value_type:     f32
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           out
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      'float*'
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:         .value_type:     f32
; CHECK:          .name:           test_ro_arg
; CHECK:          .symbol:         test_ro_arg.kd

define amdgpu_kernel void @test_ro_arg(float addrspace(1)* noalias readonly %in, float addrspace(1)* %out)
    !kernel_arg_addr_space !0 !kernel_arg_access_qual !1 !kernel_arg_type !2
    !kernel_arg_base_type !2 !kernel_arg_type_qual !3 {
  ret void
}

!0 = !{i32 1, i32 1}
!1 = !{!"none", !"none"}
!2 = !{!"float*", !"float*"}
!3 = !{!"const restrict", !""}
