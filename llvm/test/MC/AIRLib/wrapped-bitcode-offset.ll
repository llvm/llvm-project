; The AIRLib container interleaves header tables before the embedded LLVM
; bitcode wrapper. Verify that the wrapper magic (DE C0 17 0B) appears at a
; non-zero offset, i.e. it is *not* at the start of the file (which would
; indicate the header was lost).

; RUN: llc -mtriple=air -filetype=obj %s -o %t.metallib
; RUN: od -An -t x1 -N 4 %t.metallib | FileCheck %s --check-prefix=START
; RUN: od -An -t x1 %t.metallib | FileCheck %s --check-prefix=WRAP

; The first four bytes must be the AIRLib magic, not the bitcode wrapper.
; START: 4d 54 4c 42
; START-NOT: de c0 17 0b

; And the bitcode wrapper magic must still appear somewhere later.
; WRAP: de c0 17 0b

define void @kern(ptr addrspace(1) %a) {
entry:
  store float 3.0, ptr addrspace(1) %a
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kern}
