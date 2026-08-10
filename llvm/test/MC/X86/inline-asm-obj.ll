; RUN: llc %s -o - | llvm-mc -triple=x86_64-pc-linux -o %t1 -filetype=obj
; RUN: llc %s -o %t2 -filetype=obj
; RUN: cmp %t1 %t2

; Test that we can handle inline assembly referring to a temporary label.
; We crashed when using direct object emission in the past.

target triple = "x86_64-unknown-linux-gnu"

define void @fj()  {
  call void asm "bsr $0,%eax", "o"(i32 1)
  ret void
}
