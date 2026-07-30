; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s
; If-conversion defeats the purpose of this test, which is to check CBZ
; generation, so use memory barrier instruction to make sure it doesn't
; happen and we get actual branches.

define i32 @t1(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: t1:
; CHECK: cbz
  %tmp2 = icmp eq i32 %a, 0
  br i1 %tmp2, label %cond_false, label %cond_true

cond_true:
  fence seq_cst
  %tmp5 = add i32 %b, 1
  %tmp6 = and i32 %tmp5, %c
  ret i32 %tmp6

cond_false:
  fence seq_cst
  %tmp7 = add i32 %b, -1
  %tmp8 = xor i32 %tmp7, %c
  ret i32 %tmp8
}
