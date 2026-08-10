; RUN: opt < %s -passes=reassociate,instcombine -S | FileCheck %s

define i32 @f1(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4) {
; CHECK-LABEL: f1
; CHECK-NEXT: ret i32 0

  %tmp.2 = add i32 %a4, %a3
  %tmp.4 = add i32 %tmp.2, %a2
  %tmp.6 = add i32 %tmp.4, %a1
  %tmp.8 = add i32 %tmp.6, %a0
  %tmp.11 = add i32 %a3, %a2
  %tmp.13 = add i32 %tmp.11, %a1
  %tmp.15 = add i32 %tmp.13, %a0
  %tmp.18 = add i32 %a2, %a1
  %tmp.20 = add i32 %tmp.18, %a0
  %tmp.23 = add i32 %a1, %a0
  %tmp.26 = sub i32 %tmp.8, %tmp.15
  %tmp.28 = add i32 %tmp.26, %tmp.20
  %tmp.30 = sub i32 %tmp.28, %tmp.23
  %tmp.32 = sub i32 %tmp.30, %a4
  %tmp.34 = sub i32 %tmp.32, %a2
  %T = mul i32 %tmp.34, %tmp.34
  ret i32 %T
}
