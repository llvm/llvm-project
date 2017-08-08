; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -filetype=obj < %s | llvm-objdump -d - -mcpu=fiji | FileCheck %s

; CHECK: kernel0:
; CHECK-NEXT: s_endpgm
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
; CHECK-NEXT: s_nop 0  // 0000000001FC: BF800000

; CHECK-NEXT: {{^$}}
; CHECK-NEXT: kernel1:
; CHECK-NEXT: s_endpgm
define amdgpu_kernel void @kernel1(i32 addrspace(1)* addrspace(2)* %ptr.out) align 256 {
entry:
  ret void
}
