; RUN: opt -enable-new-pm=0 -print-cfg-sccs -disable-output < %s 2>&1 | FileCheck %s

; CHECK: SCCs for Function test in PostOrder:
; CHECK-NEXT: SCC #1 : %exit,
; CHECK-NEXT: SCC #2 : %0,
; CHECK-NEXT: SCC #3 : %3,
; CHECK-NEXT: SCC #4 : %2, %1,
; CHECK-NEXT: SCC #5 : %entry,
define void @test(i1 %cond) {
entry:
  br i1 %cond, label %0, label %1

0:
  br label %exit

1:
  br label %2

2:
  br i1 %cond, label %1, label %3

3:
  br label %exit

exit:
  ret void
}
