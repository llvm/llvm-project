; The .metallib container ends with the wrapped AIR bitcode whose first
; word is the LLVM bitcode-wrapper magic 0x0B17C0DE.

; RUN: llc -mtriple=air -filetype=obj %s -o %t.metallib
; RUN: grep -aoE "DE C0 17 0B|.." %t.metallib > /dev/null
; RUN: od -An -t x1 %t.metallib | FileCheck %s

; The wrapper magic is little-endian DE C0 17 0B somewhere in the file.
; CHECK: de c0 17 0b

define void @kern(ptr addrspace(1) %a) {
entry:
  store float 2.0, ptr addrspace(1) %a
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kern}
