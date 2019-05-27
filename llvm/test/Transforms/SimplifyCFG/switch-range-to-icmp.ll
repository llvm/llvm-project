; RUN: opt %s -simplifycfg -S | FileCheck %s

declare i32 @f(i32)

define i32 @basic(i32 %x) {
; CHECK-LABEL: @basic
; CHECK: x.off = add i32 %x, -5
; CHECK: %switch = icmp ult i32 %x.off, 3
; CHECK: br i1 %switch, label %a, label %default

entry:
  switch i32 %x, label %default [
    i32 5, label %a
    i32 6, label %a
    i32 7, label %a
  ]
default:
  %0 = call i32 @f(i32 0)
  ret i32 %0
a:
  %1 = call i32 @f(i32 1)
  ret i32 %1
}


define i32 @unreachable(i32 %x) {
; CHECK-LABEL: @unreachable
; CHECK: x.off = add i32 %x, -5
; CHECK: %switch = icmp ult i32 %x.off, 3
; CHECK: br i1 %switch, label %a, label %b

entry:
  switch i32 %x, label %unreachable [
    i32 5, label %a
    i32 6, label %a
    i32 7, label %a
    i32 10, label %b
    i32 20, label %b
    i32 30, label %b
    i32 40, label %b
  ]
unreachable:
  unreachable
a:
  %0 = call i32 @f(i32 0)
  ret i32 %0
b:
  %1 = call i32 @f(i32 1)
  ret i32 %1
}


define i32 @unreachable2(i32 %x) {
; CHECK-LABEL: @unreachable2
; CHECK: x.off = add i32 %x, -5
; CHECK: %switch = icmp ult i32 %x.off, 3
; CHECK: br i1 %switch, label %a, label %b

entry:
  ; Note: folding the most popular case destination into the default
  ; would prevent switch-to-icmp here.
  switch i32 %x, label %unreachable [
    i32 5, label %a
    i32 6, label %a
    i32 7, label %a
    i32 10, label %b
    i32 20, label %b
  ]
unreachable:
  unreachable
a:
  %0 = call i32 @f(i32 0)
  ret i32 %0
b:
  %1 = call i32 @f(i32 1)
  ret i32 %1
}
