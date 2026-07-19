; RUN: llc -mtriple=amdgpu7.00-amd-amdhsa -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgpu8.03-amd-amdhsa -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK --check-prefix=GFX8 %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s

; RUN: llc -mtriple=amdgpu7.00-amd-amdhsa < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgpu8.03-amd-amdhsa < %s | FileCheck --check-prefix=CHECK --check-prefix=GFX8 %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa < %s | FileCheck --check-prefix=CHECK %s

; CHECK:      amdhsa.kernels:
; CHECK-NEXT:       - .args:
; CHECK:      - .offset:         112
; CHECK-NEXT:   .size:           8
; CHECK-NEXT:   .value_kind:     hidden_completion_action
; CHECK:      - .offset:         124
; CHECK-NEXT:   .size:           8
; CHECK-NEXT:   .value_kind:     hidden_assert_fault_buffer
; GFX8:       - .offset:         192
; GFX8-NEXT:    .size:           4
; GFX8-NEXT:    .value_kind:     hidden_private_base

; CHECK:          .name:           test_v5
; CHECK:          .symbol:         test_v5.kd

; CHECK:  amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 2

define amdgpu_kernel void @test_v5() #0 {
entry:
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_assert_fault_buffer", i32 1}
!llvm.printf.fmts = !{!2}
!2 = !{!"1:1:4:%d\5Cn"}

attributes #0 = { optnone noinline }
