; RUN: rm -f %t.ll
; RUN: cat %s | sed -e 's/!range !0/!range !1/' > %t.ll
; RUN: not llvm-diff %s %t.ll 2>&1 | FileCheck %s
; CHECK:in function foo:
; CHECK:  in block %entry:
; CHECK:    >   %sum = add i32 %a, %b, !range !0
; CHECK:    >   ret i32 %sum
; CHECK:    <   %sum = add i32 %a, %b, !range !0
; CHECK:    <   ret i32 %sum

define i32 @foo(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b, !range !0
  ret i32 %sum
}

!0 = !{i32 0, i32 2147483647}
!1 = !{}
