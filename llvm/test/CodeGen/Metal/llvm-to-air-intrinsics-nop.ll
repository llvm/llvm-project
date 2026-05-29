; Negative test: a kernel with no llvm.* math intrinsics is not modified by
; the llvm-to-air-intrinsics pass; no spurious `air.fast_*` calls appear.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: air.fast_
; CHECK: fadd float
; CHECK: ret void

define void @kernel(float %a, float %b, ptr addrspace(1) %out) {
entry:
  %s = fadd float %a, %b
  store float %s, ptr addrspace(1) %out
  ret void
}
