; End-to-end: a kernel marked with !air.kernel metadata is serialized into a
; full .metallib container (MTLB magic), not bare bitcode.
; RUN: llc -mtriple=air -filetype=obj %s -o %t.metallib
; RUN: od -An -c -N 4 %t.metallib | FileCheck %s

; CHECK: M T L B

define void @vecadd(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
entry:
  %va = load float, ptr addrspace(1) %a
  %vb = load float, ptr addrspace(1) %b
  %sum = fadd float %va, %vb
  store float %sum, ptr addrspace(1) %c
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @vecadd}
