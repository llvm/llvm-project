; REQUIRES: asserts
; RUN: llc < %s -mtriple=i686-- -stats 2>&1 | FileCheck %s
; CHECK: 7 asm-printer

define i32 @g(i32 %a, i32 %b) nounwind {
entry:
  %tmp.1 = shl i32 %b, 1
  %tmp.3 = add i32 %tmp.1, %a
  %tmp.5 = mul i32 %tmp.3, %a
  %tmp.8 = mul i32 %b, %b
  %tmp.9 = add i32 %tmp.5, %tmp.8
  ret i32 %tmp.9
}

