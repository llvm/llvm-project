; RUN: opt < %s -S -passes=msan 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @ptrtoaddr_propagate(
; CHECK:       %[[SHADOW:.*]] = load i64, ptr @__msan_param_tls
; CHECK-NOT:   __msan_warning
; CHECK:       store i64 %[[SHADOW]], ptr @__msan_retval_tls
; CHECK:       ret i64
define i64 @ptrtoaddr_propagate(ptr %p) sanitize_memory {
  %addr = ptrtoaddr ptr %p to i64
  ret i64 %addr
}
