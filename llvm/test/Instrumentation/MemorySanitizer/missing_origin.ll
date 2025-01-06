; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S -passes=msan 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Test that result origin is directy propagated from the argument,
; and is not affected by all the literal undef operands.
; https://github.com/google/sanitizers/issues/559

define <4 x i32> @Shuffle(<4 x i32> %x) nounwind uwtable sanitize_memory {
entry:
  %y = shufflevector <4 x i32> %x, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  ret <4 x i32> %y
}

; CHECK-LABEL: @Shuffle(
; CHECK: [[A:%.*]] = load i32, ptr @__msan_param_origin_tls
; CHECK: store i32 [[A]], ptr @__msan_retval_origin_tls
; CHECK: ret <4 x i32>


; Regression test for origin propagation in "select i1, float, float".
; https://github.com/google/sanitizers/issues/581

define float @SelectFloat(i1 %b, float %x, float %y) nounwind uwtable sanitize_memory {
entry:
  %z = select i1 %b, float %x, float %y
  ret float %z
}

; CHECK-LABEL: @SelectFloat(
; CHECK-NOT: select {{.*}} i32 0, i32 0
; CHECK: ret float
