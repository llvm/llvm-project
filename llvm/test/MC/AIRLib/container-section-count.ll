; The AIRLib header section-count field at offsets 8..10 is a small fixed
; value for a minimal single-kernel module; pin it so a writer change that
; alters the table layout is caught immediately.

; RUN: llc -mtriple=air -filetype=obj %s -o %t.metallib
; RUN: od -An -t x1 -j 8 -N 4 %t.metallib | FileCheck %s

; CHECK: 09 00 00 81

define void @kern(ptr addrspace(1) %a) {
entry:
  store float 4.0, ptr addrspace(1) %a
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kern}
