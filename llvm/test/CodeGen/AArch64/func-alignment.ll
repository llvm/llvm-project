; RUN: llc -mtriple=aarch64 -O0 -fast-isel < %s | FileCheck %s

; Verify that Clang's C++ ABI alignment for functions of 2 does not
; result in underaligned functions when they are in special sections.
; (#90415)

define void @noSection() align 2 {
; CHECK:	.p2align	2
entry:
  ret void
}

define void @withSection() section "__TEXT,__foo,regular,pure_instructions" align 2 {
; CHECK:	.p2align	2
entry:
  ret void
}


