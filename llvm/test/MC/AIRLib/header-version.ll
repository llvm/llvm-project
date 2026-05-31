; The AIRLib container header carries the AIR/metallib version triple
; (file-format = 0x81 = macOS, OS major = 26, metal major = 4) at fixed
; offsets after the magic. Pin those bytes so an accidental change to
; AIRlibOptions or AIREmbedderPass is caught.

; RUN: llc -mtriple=air -filetype=obj %s -o %t.metallib
; RUN: od -An -t x1 -N 16 %t.metallib | FileCheck %s

; bytes 0..3  : "MTLB" magic
; bytes 4..7  : air bitcode format-version (0x01 0x80 0x02 0x00)
; bytes 8..10 : section count tag + word (0x09 0x00 0x00)
; byte  11    : container platform/OS byte (0x81 = macOS)
; bytes 12..15: total-file-size field for this minimal kernel
; CHECK: 4d 54 4c 42 01 80 02 00 09 00 00 81 {{[0-9a-f]+}} 00 00 00

define void @kern(ptr addrspace(1) %a) {
entry:
  store float 1.0, ptr addrspace(1) %a
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kern}
