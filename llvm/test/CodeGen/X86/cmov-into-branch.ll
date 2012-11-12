; RUN: llc -march=x86-64 -mcpu=core2 < %s | FileCheck %s

; cmp with single-use load, should not form cmov.
define i32 @test1(double %a, double* nocapture %b, i32 %x, i32 %y)  {
  %load = load double* %b, align 8
  %cmp = fcmp olt double %load, %a
  %cond = select i1 %cmp, i32 %x, i32 %y
  ret i32 %cond
; CHECK: test1:
; CHECK: ucomisd
; CHECK-NOT: cmov
; CHECK: j
; CHECK-NOT: cmov
}

; Sanity check: no load.
define i32 @test2(double %a, double %b, i32 %x, i32 %y)  {
  %cmp = fcmp ogt double %a, %b
  %cond = select i1 %cmp, i32 %x, i32 %y
  ret i32 %cond
; CHECK: test2:
; CHECK: ucomisd
; CHECK: cmov
}

; Multiple uses of %a, should not form cmov.
define i32 @test3(i32 %a, i32* nocapture %b, i32 %x)  {
  %load = load i32* %b, align 4
  %cmp = icmp ult i32 %load, %a
  %cond = select i1 %cmp, i32 %a, i32 %x
  ret i32 %cond
; CHECK: test3:
; CHECK: cmpl
; CHECK-NOT: cmov
; CHECK: j
; CHECK-NOT: cmov
}

; Multiple uses of the load.
define i32 @test4(i32 %a, i32* nocapture %b, i32 %x, i32 %y)  {
  %load = load i32* %b, align 4
  %cmp = icmp ult i32 %load, %a
  %cond = select i1 %cmp, i32 %x, i32 %y
  %add = add i32 %cond, %load
  ret i32 %add
; CHECK: test4:
; CHECK: cmpl
; CHECK: cmov
}

; Multiple uses of the cmp.
define i32 @test5(i32 %a, i32* nocapture %b, i32 %x, i32 %y) {
  %load = load i32* %b, align 4
  %cmp = icmp ult i32 %load, %a
  %cmp1 = icmp ugt i32 %load, %a
  %cond = select i1 %cmp1, i32 %a, i32 %y
  %cond5 = select i1 %cmp, i32 %cond, i32 %x
  ret i32 %cond5
; CHECK: test5:
; CHECK: cmpl
; CHECK: cmov
; CHECK: cmov
}
