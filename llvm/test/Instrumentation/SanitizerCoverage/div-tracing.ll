; Test -sanitizer-coverage-trace-divs=1
; RUN: opt < %s -sancov -sanitizer-coverage-level=1 -sanitizer-coverage-trace-divs=1  -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @div_a_b(i32 %a, i32 %b) local_unnamed_addr {
entry:
  %div = sdiv i32 %a, %b
  ret i32 %div
}

; CHECK-LABEL: @div_a_b
; CHECK: call void @__sanitizer_cov_trace_div4(i32 %b)
; CHECK: ret


define i32 @div_a_10(i32 %a) local_unnamed_addr {
entry:
  %div = sdiv i32 %a, 10
  ret i32 %div
}

; CHECK-LABEL: @div_a_10
; CHECK-NOT: __sanitizer_cov_trace_div
; CHECK: ret

define i64 @div_a_b_64(i64 %a, i64 %b) local_unnamed_addr {
entry:
  %div = udiv i64 %a, %b
  ret i64 %div
}

; CHECK-LABEL: @div_a_b_64
; CHECK: call void @__sanitizer_cov_trace_div8(i64 %b)
; CHECK: ret

