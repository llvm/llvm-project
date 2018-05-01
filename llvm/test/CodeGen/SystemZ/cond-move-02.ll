; Test LOCHI and LOCGHI.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -verify-machineinstrs | FileCheck %s

define i32 @f1(i32 %x) {
; CHECK-LABEL: f1:
; CHECK: chi %r2, 0
; CHECK: lhi %r2, 0
; CHECK: lochilh %r2, 42
; CHECK: br %r14
  %cond = icmp ne i32 %x, 0
  %res = select i1 %cond, i32 42, i32 0
  ret i32 %res
}

define i32 @f2(i32 %x, i32 %y) {
; CHECK-LABEL: f2:
; CHECK: chi %r2, 0
; CHECK: lochilh %r3, 42
; CHECK: br %r14
  %cond = icmp ne i32 %x, 0
  %res = select i1 %cond, i32 42, i32 %y
  ret i32 %res
}

define i32 @f3(i32 %x, i32 %y) {
; CHECK-LABEL: f3:
; CHECK: chi %r2, 0
; CHECK: lochie %r3, 42
; CHECK: br %r14
  %cond = icmp ne i32 %x, 0
  %res = select i1 %cond, i32 %y, i32 42
  ret i32 %res
}

define i64 @f4(i64 %x) {
; CHECK-LABEL: f4:
; CHECK: cghi %r2, 0
; CHECK: lghi %r2, 0
; CHECK: locghilh %r2, 42
; CHECK: br %r14
  %cond = icmp ne i64 %x, 0
  %res = select i1 %cond, i64 42, i64 0
  ret i64 %res
}

define i64 @f5(i64 %x, i64 %y) {
; CHECK-LABEL: f5:
; CHECK: cghi %r2, 0
; CHECK: locghilh %r3, 42
; CHECK: br %r14
  %cond = icmp ne i64 %x, 0
  %res = select i1 %cond, i64 42, i64 %y
  ret i64 %res
}

define i64 @f6(i64 %x, i64 %y) {
; CHECK-LABEL: f6:
; CHECK: cghi %r2, 0
; CHECK: locghie %r3, 42
; CHECK: br %r14
  %cond = icmp ne i64 %x, 0
  %res = select i1 %cond, i64 %y, i64 42
  ret i64 %res
}

; Check that we also get LOCHI as a result of early if-conversion.
define i32 @f7(i32 %x, i32 %y) {
; CHECK-LABEL: f7:
; CHECK: chi %r2, 0
; CHECK: lochie %r3, 42
; CHECK: br %r14
entry:
  %cond = icmp ne i32 %x, 0
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i32 [ %y, %if.then ], [ 42, %entry ]
  ret i32 %res
}

; ... and the same for LOCGHI.
define i64 @f8(i64 %x, i64 %y) {
; CHECK-LABEL: f8:
; CHECK: cghi %r2, 0
; CHECK: locghie %r3, 42
; CHECK: br %r14
entry:
  %cond = icmp ne i64 %x, 0
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i64 [ %y, %if.then ], [ 42, %entry ]
  ret i64 %res
}

; Check that inverting the condition works as well.
define i32 @f9(i32 %x, i32 %y) {
; CHECK-LABEL: f9:
; CHECK: chi %r2, 0
; CHECK: lochilh %r3, 42
; CHECK: br %r14
entry:
  %cond = icmp ne i32 %x, 0
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i32 [ 42, %if.then ], [ %y, %entry ]
  ret i32 %res
}

; ... and the same for LOCGHI.
define i64 @f10(i64 %x, i64 %y) {
; CHECK-LABEL: f10:
; CHECK: cghi %r2, 0
; CHECK: locghilh %r3, 42
; CHECK: br %r14
entry:
  %cond = icmp ne i64 %x, 0
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i64 [ 42, %if.then ], [ %y, %entry ]
  ret i64 %res
}

