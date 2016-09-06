; Test -sanitizer-coverage-experimental-tracing
; RUN: opt < %s -sancov -sanitizer-coverage-level=2 -sanitizer-coverage-experimental-tracing  -S | FileCheck %s --check-prefix=CHECK1
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-experimental-tracing  -S | FileCheck %s --check-prefix=CHECK3
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc  -S | FileCheck %s --check-prefix=CHECK_PC

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo(i32* %a) sanitize_address {
entry:
  %tobool = icmp eq i32* %a, null
  br i1 %tobool, label %if.end, label %if.then

  if.then:                                          ; preds = %entry
  store i32 0, i32* %a, align 4
  br label %if.end

  if.end:                                           ; preds = %entry, %if.then
  ret void
}

; CHECK1-LABEL: define void @foo
; CHECK1: call void @__sanitizer_cov_trace_func_enter
; CHECK1: call void @__sanitizer_cov_trace_basic_block
; CHECK1-NOT: call void @__sanitizer_cov_trace_basic_block
; CHECK1: ret void

; CHECK3-LABEL: define void @foo
; CHECK3: call void @__sanitizer_cov_trace_func_enter
; CHECK3: call void @__sanitizer_cov_trace_basic_block
; CHECK3: call void @__sanitizer_cov_trace_basic_block
; CHECK3-NOT: call void @__sanitizer_cov_trace_basic_block
; CHECK3: ret void

; CHECK_PC-LABEL: define void @foo
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC-NOT: call void @__sanitizer_cov_trace_pc
; CHECK_PC: ret void
; CHECK_PC-NOT: call void @__sanitizer_cov_module_init
