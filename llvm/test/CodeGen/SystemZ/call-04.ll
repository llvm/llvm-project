; Test conditional sibling calls.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @fun_a()
declare void @fun_b()
declare void @fun_c(i32)

@var = global i32 1;

; Check a conditional sibling call.
define void @f1(i32 %val1, i32 %val2) {
; CHECK-LABEL: f1:
; CHECK: cr %r2, %r3
; CHECK: jgl fun_a@PLT
; CHECK: br %r14
  %cond = icmp slt i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call when there are two possibilities.
define void @f2(i32 %val1, i32 %val2) {
; CHECK-LABEL: f2:
; CHECK: cr %r2, %r3
; CHECK: jghe fun_b@PLT
; CHECK: jg fun_a@PLT
  %cond = icmp slt i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  tail call void @fun_b()
  ret void
}

; Check a conditional sibling call with an argument - not supported.
define void @f3(i32 %val1, i32 %val2) {
; CHECK-LABEL: f3:
; CHECK: crjhe %r2, %r3
; CHECK: jg fun_c@PLT
; CHECK: br %r14
  %cond = icmp slt i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_c(i32 1)
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - unsigned compare.
define void @f4(i32 %val1, i32 %val2) {
; CHECK-LABEL: f4:
; CHECK: clr %r2, %r3
; CHECK: jgl fun_a@PLT
; CHECK: br %r14
  %cond = icmp ult i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - 64-bit compare.
define void @f5(i64 %val1, i64 %val2) {
; CHECK-LABEL: f5:
; CHECK: cgr %r2, %r3
; CHECK: jgl fun_a@PLT
; CHECK: br %r14
  %cond = icmp slt i64 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - unsigned 64-bit compare.
define void @f6(i64 %val1, i64 %val2) {
; CHECK-LABEL: f6:
; CHECK: clgr %r2, %r3
; CHECK: jgl fun_a@PLT
; CHECK: br %r14
  %cond = icmp ult i64 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - less-equal compare.
define void @f7(i32 %val1, i32 %val2) {
; CHECK-LABEL: f7:
; CHECK: cr %r2, %r3
; CHECK: jgle fun_a@PLT
; CHECK: br %r14
  %cond = icmp sle i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - high compare.
define void @f8(i32 %val1, i32 %val2) {
; CHECK-LABEL: f8:
; CHECK: cr %r2, %r3
; CHECK: jgh fun_a@PLT
; CHECK: br %r14
  %cond = icmp sgt i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - high-equal compare.
define void @f9(i32 %val1, i32 %val2) {
; CHECK-LABEL: f9:
; CHECK: cr %r2, %r3
; CHECK: jghe fun_a@PLT
; CHECK: br %r14
  %cond = icmp sge i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - equal compare.
define void @f10(i32 %val1, i32 %val2) {
; CHECK-LABEL: f10:
; CHECK: cr %r2, %r3
; CHECK: jge fun_a@PLT
; CHECK: br %r14
  %cond = icmp eq i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - unequal compare.
define void @f11(i32 %val1, i32 %val2) {
; CHECK-LABEL: f11:
; CHECK: cr %r2, %r3
; CHECK: jglh fun_a@PLT
; CHECK: br %r14
  %cond = icmp ne i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate slt.
define void @f12(i32 %val1) {
; CHECK-LABEL: f12:
; CHECK: chi %r2, 4
; CHECK: jgle fun_a@PLT
; CHECK: br %r14
  %cond = icmp slt i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate sle.
define void @f13(i32 %val1) {
; CHECK-LABEL: f13:
; CHECK: chi %r2, 5
; CHECK: jgle fun_a@PLT
; CHECK: br %r14
  %cond = icmp sle i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate sgt.
define void @f14(i32 %val1) {
; CHECK-LABEL: f14:
; CHECK: chi %r2, 6
; CHECK: jghe fun_a@PLT
; CHECK: br %r14
  %cond = icmp sgt i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate sge.
define void @f15(i32 %val1) {
; CHECK-LABEL: f15:
; CHECK: chi %r2, 5
; CHECK: jghe fun_a@PLT
; CHECK: br %r14
  %cond = icmp sge i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate eq.
define void @f16(i32 %val1) {
; CHECK-LABEL: f16:
; CHECK: chi %r2, 5
; CHECK: jge fun_a@PLT
; CHECK: br %r14
  %cond = icmp eq i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate ne.
define void @f17(i32 %val1) {
; CHECK-LABEL: f17:
; CHECK: chi %r2, 5
; CHECK: jglh fun_a@PLT
; CHECK: br %r14
  %cond = icmp ne i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate ult.
define void @f18(i32 %val1) {
; CHECK-LABEL: f18:
; CHECK: clfi %r2, 4
; CHECK: jgle fun_a@PLT
; CHECK: br %r14
  %cond = icmp ult i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate 64-bit slt.
define void @f19(i64 %val1) {
; CHECK-LABEL: f19:
; CHECK: cghi %r2, 4
; CHECK: jgle fun_a@PLT
; CHECK: br %r14
  %cond = icmp slt i64 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate 64-bit ult.
define void @f20(i64 %val1) {
; CHECK-LABEL: f20:
; CHECK: clgfi %r2, 4
; CHECK: jgle fun_a@PLT
; CHECK: br %r14
  %cond = icmp ult i64 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void @fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}
