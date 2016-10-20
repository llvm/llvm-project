; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s

; Checks if movl $1 is sinked to critical edge.
; CHECK-NOT: movl $1
; CHECK: jbe
; CHECK: movl $1
define i32 @test(i32 %n, i32 %k) nounwind  {
entry:
  %cmp = icmp ugt i32 %k, %n
  br i1 %cmp, label %ifthen, label %ifend, !prof !1

ifthen:
  %y = add i32 %k, 2
  br label %ifend

ifend:
  %ret = phi i32 [ 1, %entry ] , [ %y, %ifthen]
  ret i32 %ret
}

!1 = !{!"branch_weights", i32 100, i32 1}
