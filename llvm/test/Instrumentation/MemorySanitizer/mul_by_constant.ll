; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Check instrumentation mul when one of the operands is a constant.

define i64 @MulConst(i64 %x) sanitize_memory {
entry:
  %y = mul i64 %x, 42949672960000
  ret i64 %y
}

; 42949672960000 = 2**32 * 10000
; 36 trailing zero bits
; 68719476736 = 2**36

; CHECK-LABEL: @MulConst(
; CHECK: [[A:%.*]] = load {{.*}} @__msan_param_tls
; CHECK: [[B:%.*]] = mul i64 [[A]], 68719476736
; CHECK: store i64 [[B]], ptr @__msan_retval_tls


define i64 @MulZero(i64 %x) sanitize_memory {
entry:
  %y = mul i64 %x, 0
  ret i64 %y
}

; CHECK-LABEL: @MulZero(
; CHECK: [[A:%.*]] = load {{.*}} @__msan_param_tls
; CHECK: [[B:%.*]] = mul i64 [[A]], 0{{$}}
; CHECK: store i64 [[B]], ptr @__msan_retval_tls


define i64 @MulNeg(i64 %x) sanitize_memory {
entry:
  %y = mul i64 %x, -16
  ret i64 %y
}

; CHECK-LABEL: @MulNeg(
; CHECK: [[A:%.*]] = load {{.*}} @__msan_param_tls
; CHECK: [[B:%.*]] = mul i64 [[A]], 16
; CHECK: store i64 [[B]], ptr @__msan_retval_tls


define i64 @MulNeg2(i64 %x) sanitize_memory {
entry:
  %y = mul i64 %x, -48
  ret i64 %y
}

; CHECK-LABEL: @MulNeg2(
; CHECK: [[A:%.*]] = load {{.*}} @__msan_param_tls
; CHECK: [[B:%.*]] = mul i64 [[A]], 16
; CHECK: store i64 [[B]], ptr @__msan_retval_tls


define i64 @MulOdd(i64 %x) sanitize_memory {
entry:
  %y = mul i64 %x, 12345
  ret i64 %y
}

; CHECK-LABEL: @MulOdd(
; CHECK: [[A:%.*]] = load {{.*}} @__msan_param_tls
; CHECK: [[B:%.*]] = mul i64 [[A]], 1
; CHECK: store i64 [[B]], ptr @__msan_retval_tls


define i64 @MulLarge(i64 %x) sanitize_memory {
entry:
  %y = mul i64 %x, -9223372036854775808
  ret i64 %y
}

; -9223372036854775808 = 0x7000000000000000

; CHECK-LABEL: @MulLarge(
; CHECK: [[A:%.*]] = load {{.*}} @__msan_param_tls
; CHECK: [[B:%.*]] = mul i64 [[A]], -9223372036854775808
; CHECK: store i64 [[B]], ptr @__msan_retval_tls

define <4 x i32> @MulVectorConst(<4 x i32> %x) sanitize_memory {
entry:
  %y = mul <4 x i32> %x, <i32 3072, i32 0, i32 -16, i32 -48>
  ret <4 x i32> %y
}

; CHECK-LABEL: @MulVectorConst(
; CHECK: [[A:%.*]] = load {{.*}} @__msan_param_tls
; CHECK: [[B:%.*]] = mul <4 x i32> [[A]], <i32 1024, i32 0, i32 16, i32 16>
; CHECK: store <4 x i32> [[B]], ptr @__msan_retval_tls


; The constant in multiplication does not have to be a literal integer constant.
@X = linkonce_odr global ptr null
define i64 @MulNonIntegerConst(i64 %a) sanitize_memory {
  %mul = mul i64 %a, ptrtoint (ptr @X to i64)
  ret i64 %mul
}

; CHECK-LABEL: @MulNonIntegerConst(
; CHECK: [[A:%.*]] = load {{.*}} @__msan_param_tls
; CHECK: [[B:%.*]] = mul i64 [[A]], 1
; CHECK: store i64 [[B]], {{.*}}@__msan_retval_tls

define <2 x i64> @MulNonIntegerVectorConst(<2 x i64> %a) sanitize_memory {
  %mul = mul <2 x i64> %a, <i64 3072, i64 ptrtoint (ptr @X to i64)>
  ret <2 x i64> %mul
}

; CHECK-LABEL: @MulNonIntegerVectorConst(
; CHECK: [[A:%.*]] = load {{.*}} @__msan_param_tls
; CHECK: [[B:%.*]] = mul <2 x i64> [[A]], <i64 1024, i64 1>
; CHECK: store <2 x i64> [[B]], {{.*}}@__msan_retval_tls
