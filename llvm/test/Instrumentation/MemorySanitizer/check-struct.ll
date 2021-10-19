; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S -passes='module(msan-module),function(msan)' 2>&1 | \
; RUN:   FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @main
define { i32, i8 } @main() sanitize_memory {
; CHECK: [[P:%.*]] = inttoptr i64 0 to { i32, i8 }*
  %p = inttoptr i64 0 to { i32, i8 } *
; CHECK: [[O:%.*]] = load { i32, i8 }, { i32, i8 }* [[P]]
  %o = load { i32, i8 }, { i32, i8 } *%p
; CHECK: [[FIELD0:%.+]] = extractvalue { i32, i8 } %_msld, 0
; CHECK: [[F0_POISONED:%.+]] = icmp ne i32 [[FIELD0]]
; CHECK: [[FIELD1:%.+]] = extractvalue { i32, i8 } %_msld, 1
; CHECK: [[F1_POISONED:%.+]] = icmp ne i8 [[FIELD1]]
; CHECK: [[F1_OR:%.+]] = or i1 [[F0_POISONED]], [[F1_POISONED]]
; CHECK-NOT: icmp ne i1 {{.*}}, false
; CHECK: br i1 [[F1_OR]]
; CHECK: call void @__msan_warning
; CHECK: ret { i32, i8 } [[O]]
  ret { i32, i8 } %o
}
