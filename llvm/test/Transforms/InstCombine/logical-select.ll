; RUN: opt < %s -instcombine -S | FileCheck %s


define i32 @foo(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
  %e = icmp slt i32 %a, %b
  %f = sext i1 %e to i32
  %g = and i32 %c, %f
  %h = xor i32 %f, -1
  %i = and i32 %d, %h
  %j = or i32 %g, %i
  ret i32 %j
; CHECK: %e = icmp slt i32 %a, %b
; CHECK: %j = select i1 %e, i32 %c, i32 %d
; CHECK: ret i32 %j
}
define i32 @bar(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
  %e = icmp slt i32 %a, %b
  %f = sext i1 %e to i32
  %g = and i32 %c, %f
  %h = xor i32 %f, -1
  %i = and i32 %d, %h
  %j = or i32 %i, %g
  ret i32 %j
; CHECK: %e = icmp slt i32 %a, %b
; CHECK: %j = select i1 %e, i32 %c, i32 %d
; CHECK: ret i32 %j
}

define i32 @goo(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
entry:
  %0 = icmp slt i32 %a, %b
  %iftmp.0.0 = select i1 %0, i32 -1, i32 0
  %1 = and i32 %iftmp.0.0, %c
  %not = xor i32 %iftmp.0.0, -1
  %2 = and i32 %not, %d
  %3 = or i32 %1, %2
  ret i32 %3
; CHECK: %0 = icmp slt i32 %a, %b
; CHECK: %1 = select i1 %0, i32 %c, i32 %d
; CHECK: ret i32 %1
}
define i32 @poo(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
entry:
  %0 = icmp slt i32 %a, %b
  %iftmp.0.0 = select i1 %0, i32 -1, i32 0
  %1 = and i32 %iftmp.0.0, %c
  %iftmp = select i1 %0, i32 0, i32 -1
  %2 = and i32 %iftmp, %d
  %3 = or i32 %1, %2
  ret i32 %3
; CHECK: %0 = icmp slt i32 %a, %b
; CHECK: %1 = select i1 %0, i32 %c, i32 %d
; CHECK: ret i32 %1
}

define i32 @par(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
entry:
  %0 = icmp slt i32 %a, %b
  %iftmp.1.0 = select i1 %0, i32 -1, i32 0
  %1 = and i32 %iftmp.1.0, %c
  %not = xor i32 %iftmp.1.0, -1
  %2 = and i32 %not, %d
  %3 = or i32 %1, %2
  ret i32 %3
; CHECK: %0 = icmp slt i32 %a, %b
; CHECK: %1 = select i1 %0, i32 %c, i32 %d
; CHECK: ret i32 %1
}
