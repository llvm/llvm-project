; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=2 -S          \
; RUN: -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=2 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; Check origin handling of array types.

define void @foo([2 x i64] %v, [2 x i64]* %p) sanitize_memory {
entry:
  store [2 x i64] %v, [2 x i64]* %p, align 8
  ret void
}

; CHECK-LABEL: @foo
; CHECK: [[PARAM:%[01-9a-z]+]] = load {{.*}} @__msan_param_tls
; CHECK: [[ORIGIN:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls

; CHECK: [[TMP1:%[01-9a-z]+]] = ptrtoint
; CHECK: [[TMP2:%[01-9a-z]+]] = xor i64 [[TMP1]]
; CHECK: [[TMP3:%[01-9a-z]+]] = inttoptr i64 [[TMP2]] to [2 x i64]*
; CHECK: store [2 x i64] [[PARAM]], [2 x i64]* [[TMP3]]

; CHECK: {{.*}} call i32 @__msan_chain_origin(i32 {{.*}}[[ORIGIN]])
