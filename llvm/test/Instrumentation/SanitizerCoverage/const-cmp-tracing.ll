; Test -sanitizer-coverage-trace-compares=1
; RUN: opt < %s -sancov -sanitizer-coverage-level=1 -sanitizer-coverage-trace-compares=1  -S -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-compares=1  -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define i32 @foo(i32 %a, i32 %b) #0 {
entry:

; compare (non-const, non-const)
  %cmp = icmp slt i32 %a, %b
; CHECK: call void @__sanitizer_cov_trace_cmp4
; CHECK-NEXT: icmp slt i32 %a, %b

; compare (const, non-const)
  icmp slt i32 %a, 1
; CHECK: call void @__sanitizer_cov_trace_const_cmp4(i32 zeroext 1, i32 zeroext %a)
; CHECK-NEXT: icmp slt i32 %a, 1

; compare (non-const, const)
  icmp slt i32 1, %a
; CHECK: call void @__sanitizer_cov_trace_const_cmp4(i32 zeroext 1, i32 zeroext %a)
; CHECK-NEXT: icmp slt i32 1, %a

; compare (const, const) - should not be instrumented
  icmp slt i32 1, 0
; CHECK-NOT: call void @__sanitizer_cov_trace
; CHECK: icmp slt i32 1, 0

; compare variables of byte size
  %x = trunc i32 %a to i8

  icmp slt i8 %x, 1
; CHECK: call void @__sanitizer_cov_trace_const_cmp1(i8 zeroext 1, i8 zeroext %x)
; CHECK-NEXT: icmp slt i8 %x, 1

  icmp slt i8 1, %x
; CHECK: call void @__sanitizer_cov_trace_const_cmp1(i8 zeroext 1, i8 zeroext %x)
; CHECK-NEXT: icmp slt i8 1, %x

; compare variables of word size
  %y = trunc i32 %a to i16

  icmp slt i16 %y, 1
; CHECK: call void @__sanitizer_cov_trace_const_cmp2(i16 zeroext 1, i16 zeroext %y)
; CHECK-NEXT: icmp slt i16 %y, 1

  icmp slt i16 1, %y
; CHECK: call void @__sanitizer_cov_trace_const_cmp2(i16 zeroext 1, i16 zeroext %y)
; CHECK-NEXT: icmp slt i16 1, %y

; compare variables of qword size
  %z = zext i32 %a to i64

  icmp slt i64 %z, 1
; CHECK: call void @__sanitizer_cov_trace_const_cmp8(i64 1, i64 %z)
; CHECK-NEXT: icmp slt i64 %z, 1

  icmp slt i64 1, %z
; CHECK: call void @__sanitizer_cov_trace_const_cmp8(i64 1, i64 %z)
; CHECK-NEXT: icmp slt i64 1, %z

  %conv = zext i1 %cmp to i32
  ret i32 %conv
}
