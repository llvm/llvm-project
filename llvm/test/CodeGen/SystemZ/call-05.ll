; Test conditional sibling calls.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s


@var = global i32 1;
@fun_a = global void()* null;
@fun_b = global void()* null;
@fun_c = global void(i32)* null;

; Check a conditional sibling call.
define void @f1(i32 %val1, i32 %val2) {
; CHECK-LABEL: f1:
; CHECK: crbl %r2, %r3, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp slt i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call when there are two possibilities.
define void @f2(i32 %val1, i32 %val2) {
; CHECK-LABEL: f2:
; CHECK: crbl %r2, %r3, 0(%r1)
; CHECK: br %r1
  %fun_a = load volatile void() *, void()** @fun_a;
  %fun_b = load volatile void() *, void()** @fun_b;
  %cond = icmp slt i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  tail call void %fun_b()
  ret void
}

; Check a conditional sibling call with an argument - not supported.
define void @f3(i32 %val1, i32 %val2) {
; CHECK-LABEL: f3:
; CHECK: crjhe %r2, %r3
; CHECK: br %r1
; CHECK: br %r14
  %fun_c = load volatile void(i32) *, void(i32)** @fun_c;
  %cond = icmp slt i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_c(i32 1)
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - unsigned compare.
define void @f4(i32 %val1, i32 %val2) {
; CHECK-LABEL: f4:
; CHECK: clrbl %r2, %r3, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp ult i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - 64-bit compare.
define void @f5(i64 %val1, i64 %val2) {
; CHECK-LABEL: f5:
; CHECK: cgrbl %r2, %r3, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp slt i64 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - unsigned 64-bit compare.
define void @f6(i64 %val1, i64 %val2) {
; CHECK-LABEL: f6:
; CHECK: clgrbl %r2, %r3, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp ult i64 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - less-equal compare.
define void @f7(i32 %val1, i32 %val2) {
; CHECK-LABEL: f7:
; CHECK: crble %r2, %r3, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp sle i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - high compare.
define void @f8(i32 %val1, i32 %val2) {
; CHECK-LABEL: f8:
; CHECK: crbh %r2, %r3, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp sgt i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - high-equal compare.
define void @f9(i32 %val1, i32 %val2) {
; CHECK-LABEL: f9:
; CHECK: crbhe %r2, %r3, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp sge i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - equal compare.
define void @f10(i32 %val1, i32 %val2) {
; CHECK-LABEL: f10:
; CHECK: crbe %r2, %r3, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp eq i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - unequal compare.
define void @f11(i32 %val1, i32 %val2) {
; CHECK-LABEL: f11:
; CHECK: crblh %r2, %r3, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp ne i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate slt.
define void @f12(i32 %val1) {
; CHECK-LABEL: f12:
; CHECK: cible %r2, 4, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp slt i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate sle.
define void @f13(i32 %val1) {
; CHECK-LABEL: f13:
; CHECK: cible %r2, 5, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp sle i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate sgt.
define void @f14(i32 %val1) {
; CHECK-LABEL: f14:
; CHECK: cibhe %r2, 6, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp sgt i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate sge.
define void @f15(i32 %val1) {
; CHECK-LABEL: f15:
; CHECK: cibhe %r2, 5, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp sge i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate eq.
define void @f16(i32 %val1) {
; CHECK-LABEL: f16:
; CHECK: cibe %r2, 5, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp eq i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate ne.
define void @f17(i32 %val1) {
; CHECK-LABEL: f17:
; CHECK: ciblh %r2, 5, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp ne i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate ult.
define void @f18(i32 %val1) {
; CHECK-LABEL: f18:
; CHECK: clible %r2, 4, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp ult i32 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate 64-bit slt.
define void @f19(i64 %val1) {
; CHECK-LABEL: f19:
; CHECK: cgible %r2, 4, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp slt i64 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - immediate 64-bit ult.
define void @f20(i64 %val1) {
; CHECK-LABEL: f20:
; CHECK: clgible %r2, 4, 0(%r1)
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = icmp ult i64 %val1, 5;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call to an argument - will fail due to
; intervening lgr.
define void @f21(i32 %val1, i32 %val2, void()* %fun) {
; CHECK-LABEL: f21:
; CHECK: crjhe %r2, %r3
; CHECK: lgr %r1, %r4
; CHECK: br %r1
; CHECK: br %r14
  %cond = icmp slt i32 %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - float olt compare.
define void @f22(float %val1, float %val2) {
; CHECK-LABEL: f22:
; CHECK: cebr %f0, %f2
; CHECK: blr %r1
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = fcmp olt float %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - float ult compare.
define void @f23(float %val1, float %val2) {
; CHECK-LABEL: f23:
; CHECK: cebr %f0, %f2
; CHECK: bnher %r1
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = fcmp ult float %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - float ord compare.
define void @f24(float %val1, float %val2) {
; CHECK-LABEL: f24:
; CHECK: cebr %f0, %f2
; CHECK: bnor %r1
; CHECK: br %r14
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = fcmp ord float %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}

; Check a conditional sibling call - float uno compare.
define void @f25(float %val1, float %val2) {
; CHECK-LABEL: f25:
; CHECK: cebr %f0, %f2
; CHECK: jo
; CHECK: br %r14
; CHECK: br %r1
  %fun_a = load volatile void() *, void()** @fun_a;
  %cond = fcmp uno float %val1, %val2;
  br i1 %cond, label %a, label %b;

a:
  tail call void %fun_a()
  ret void

b:
  store i32 1, i32 *@var;
  ret void
}
