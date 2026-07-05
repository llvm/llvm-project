; RUN: llc -mtriple=i686-- -mcpu=i486 < %s | FileCheck %s

define i32 @test1(i32 %g, ptr %j) {
  %tobool = icmp eq i32 %g, 0
  %cmp = load i32, ptr %j, align 4
  %retval.0 = select i1 %tobool, i32 1, i32 %cmp
  ret i32 %retval.0

; CHECK-LABEL: test1:
; CHECK-NOT: cmov
}
