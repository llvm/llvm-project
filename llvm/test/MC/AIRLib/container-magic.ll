; The AIRLib MC writer must emit a container starting with "MTLB" and
; the format-version word produced by AIREmbedderPass.

; RUN: llc -mtriple=air -filetype=obj %s -o %t.metallib
; RUN: od -An -t x1 -N 16 %t.metallib | FileCheck %s

; CHECK: 4d 54 4c 42 01 80 02 00 09 00 00 {{[0-9a-f]+}} {{[0-9a-f]+}} 00 00 00

define void @kern(ptr addrspace(1) %a) {
entry:
  store float 1.0, ptr addrspace(1) %a
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kern}
