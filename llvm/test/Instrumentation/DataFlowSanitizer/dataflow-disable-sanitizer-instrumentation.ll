
; This test checks that we are not instrumenting sanitizer code.
; RUN: opt < %s -passes='module(msan)' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function with sanitize_memory is instrumented.
; Function Attrs: nounwind uwtable
define void @instr_sa(ptr %a) sanitize_memory {
entry:
  %tmp1 = load i32, ptr %a, align 4
  %tmp2 = add i32 %tmp1,  1
  store i32 %tmp2, ptr %a, align 4
  ret void
}

; CHECK-LABEL: @instr_sa
; CHECK: %0 = load i64, ptr @__msan_param_tls


; Function with disable_sanitizer_instrumentation is not instrumented.
; Function Attrs: nounwind uwtable
define void @noinstr_dsi(ptr %a) disable_sanitizer_instrumentation {
entry:
  %tmp1 = load i32, ptr %a, align 4
  %tmp2 = add i32 %tmp1,  1
  store i32 %tmp2, ptr %a, align 4
  ret void
}

; CHECK-LABEL: @noinstr_dsi
; CHECK-NOT: %0 = load i64, ptr @__msan_param_tls


; disable_sanitizer_instrumentation takes precedence over sanitize_memory.
; Function Attrs: nounwind uwtable
define void @noinstr_dsi_sa(ptr %a) disable_sanitizer_instrumentation sanitize_memory {
entry:
  %tmp1 = load i32, ptr %a, align 4
  %tmp2 = add i32 %tmp1,  1
  store i32 %tmp2, ptr %a, align 4
  ret void
}

; CHECK-LABEL: @noinstr_dsi_sa
; CHECK-NOT: %0 = load i64, ptr @__msan_param_tls
