; Test -sanitizer-coverage-trace-compares=1 API declarations on a non-x86_64 arch
; RUN: opt < %s -sancov -sanitizer-coverage-level=1 -sanitizer-coverage-trace-compares=1  -S | FileCheck %s

target triple = "i386-unknown-linux-gnu"
define i32 @foo() #0 {
entry:
  ret i32 0
}

; CHECK: declare void @__sanitizer_cov_trace_pc_indir(i64)
; CHECK: declare void @__sanitizer_cov_trace_cmp1(i8, i8)
; CHECK: declare void @__sanitizer_cov_trace_cmp2(i16, i16)
; CHECK: declare void @__sanitizer_cov_trace_cmp4(i32, i32)
; CHECK: declare void @__sanitizer_cov_trace_cmp8(i64, i64)
; CHECK: declare void @__sanitizer_cov_trace_div4(i32)
; CHECK: declare void @__sanitizer_cov_trace_div8(i64)
; CHECK: declare void @__sanitizer_cov_trace_gep(i64)
; CHECK: declare void @__sanitizer_cov_trace_switch(i64, i64*)
; CHECK: declare void @__sanitizer_cov_trace_pc()
; CHECK: declare void @__sanitizer_cov_trace_pc_guard(i32*)
; CHECK: declare void @__sanitizer_cov_trace_pc_guard_init(i32*, i32*)
; CHECK-NOT: declare
