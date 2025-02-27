; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -filetype=obj < %s | llvm-objdump -d - --mcpu=fiji | FileCheck %s

; CHECK: <kernel0>:
; CHECK: s_endpgm
define amdgpu_kernel void @kernel0() align 256 {
entry:
  ret void
}

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0  // 0000000000FC: BF800000

; CHECK-EMPTY:
; CHECK-NEXT: <kernel1>:
; CHECK: s_endpgm
define amdgpu_kernel void @kernel1(ptr addrspace(4) %ptr.out) align 256 {
entry:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
