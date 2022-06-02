; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S          \
; RUN: -passes=msan 2>&1 | FileCheck -check-prefix=CHECK                       \
; RUN: %s --allow-empty

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define [2 x i32] @InsertValue(i32 %x, i32 %y) sanitize_memory {
entry:
  %a = insertvalue [2 x i32] undef, i32 %x, 0
  %b = insertvalue [2 x i32] %a, i32 %y, 1
  ret [2 x i32] %b
}

; CHECK-LABEL: @InsertValue(
; CHECK-DAG: [[Sx:%.*]] = load i32, ptr @__msan_param_tls
; CHECK-DAG: [[Sy:%.*]] = load i32, ptr {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK: [[A:%.*]] = insertvalue [2 x i32] [i32 -1, i32 -1], i32 [[Sx]], 0
; CHECK: [[B:%.*]] = insertvalue [2 x i32] [[A]], i32 [[Sy]], 1
; CHECK: store [2 x i32] [[B]], ptr {{.*}}@__msan_retval_tls
; CHECK: ret [2 x i32]


define [2 x double] @InsertValueDouble(double %x, double %y) sanitize_memory {
entry:
  %a = insertvalue [2 x double] undef, double %x, 0
  %b = insertvalue [2 x double] %a, double %y, 1
  ret [2 x double] %b
}

; CHECK-LABEL: @InsertValueDouble(
; CHECK-DAG: [[Sx:%.*]] = load i64, ptr @__msan_param_tls
; CHECK-DAG: [[Sy:%.*]] = load i64, ptr {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK: [[A:%.*]] = insertvalue [2 x i64] [i64 -1, i64 -1], i64 [[Sx]], 0
; CHECK: [[B:%.*]] = insertvalue [2 x i64] [[A]], i64 [[Sy]], 1
; CHECK: store [2 x i64] [[B]], ptr {{.*}}@__msan_retval_tls
; CHECK: ret [2 x double]


define i32 @ExtractValue([2 x i32] %a) sanitize_memory {
entry:
  %x = extractvalue [2 x i32] %a, 1
  ret i32 %x
}

; CHECK-LABEL: @ExtractValue(
; CHECK: [[Sa:%.*]] = load [2 x i32], ptr @__msan_param_tls
; CHECK: [[Sx:%.*]] = extractvalue [2 x i32] [[Sa]], 1
; CHECK: store i32 [[Sx]], ptr @__msan_retval_tls
; CHECK: ret i32


; Regression test for PR20493.

%MyStruct = type { i32, i32, [3 x i32] }

define i32 @ArrayInStruct(%MyStruct %s) sanitize_memory {
  %x = extractvalue %MyStruct %s, 2, 1
  ret i32 %x
}

; CHECK-LABEL: @ArrayInStruct(
; CHECK: [[Ss:%.*]] = load { i32, i32, [3 x i32] }, ptr @__msan_param_tls
; CHECK: [[Sx:%.*]] = extractvalue { i32, i32, [3 x i32] } [[Ss]], 2, 1
; CHECK: store i32 [[Sx]], ptr @__msan_retval_tls
; CHECK: ret i32


define i32 @ArrayOfStructs([3 x { i32, i32 }] %a) sanitize_memory {
  %x = extractvalue [3 x { i32, i32 }] %a, 2, 1
  ret i32 %x
}

; CHECK-LABEL: @ArrayOfStructs(
; CHECK: [[Ss:%.*]] = load [3 x { i32, i32 }], ptr @__msan_param_tls
; CHECK: [[Sx:%.*]] = extractvalue [3 x { i32, i32 }] [[Ss]], 2, 1
; CHECK: store i32 [[Sx]], ptr @__msan_retval_tls
; CHECK: ret i32


define <8 x i16> @ArrayOfVectors([3 x <8 x i16>] %a) sanitize_memory {
  %x = extractvalue [3 x <8 x i16>] %a, 1
  ret <8 x i16> %x
}

; CHECK-LABEL: @ArrayOfVectors(
; CHECK: [[Ss:%.*]] = load [3 x <8 x i16>], ptr @__msan_param_tls
; CHECK: [[Sx:%.*]] = extractvalue [3 x <8 x i16>] [[Ss]], 1
; CHECK: store <8 x i16> [[Sx]], ptr @__msan_retval_tls
; CHECK: ret <8 x i16>
