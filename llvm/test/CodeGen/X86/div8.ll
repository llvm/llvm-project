; RUN: llc < %s | FileCheck %s
; ModuleID = '8div.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.6"

define signext i8 @test_div(i8 %dividend, i8 %divisor) nounwind ssp {
entry:
  %dividend.addr = alloca i8, align 2
  %divisor.addr = alloca i8, align 1
  %quotient = alloca i8, align 1
  store i8 %dividend, ptr %dividend.addr, align 2
  store i8 %divisor, ptr %divisor.addr, align 1
  %tmp = load i8, ptr %dividend.addr, align 2
  %tmp1 = load i8, ptr %divisor.addr, align 1
; Insist on i8->i32 zero extension, even though divb demands only i16:
; CHECK: movzbl {{.*}}%eax
; CHECK: divb
  %div = udiv i8 %tmp, %tmp1
  store i8 %div, ptr %quotient, align 1
  %tmp4 = load i8, ptr %quotient, align 1
  ret i8 %tmp4
}
