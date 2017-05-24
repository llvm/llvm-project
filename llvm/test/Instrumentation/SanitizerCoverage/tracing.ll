; Test -sanitizer-coverage-experimental-tracing
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc  -S | FileCheck %s --check-prefix=CHECK_PC
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard  -S | FileCheck %s --check-prefix=CHECK_PC_GUARD
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard  -S -mtriple=x86_64-apple-macosx | FileCheck %s --check-prefix=CHECK_PC_GUARD_DARWIN

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

; CHECK_PC-LABEL: define void @foo
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC-NOT: call void @__sanitizer_cov_trace_pc
; CHECK_PC: ret void
; CHECK_PC-NOT: call void @__sanitizer_cov_module_init

; CHECK_PC_GUARD-LABEL: define void @foo
; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD-NOT: call void @__sanitizer_cov_trace_pc
; CHECK_PC_GUARD: ret void
; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard_init(i32* bitcast (i32** @__start___sancov_guards to i32*), i32* bitcast (i32** @__stop___sancov_guards to i32*))

; CHECK_PC_GUARD_DARWIN-LABEL: define void @foo
; CHECK_PC_GUARD_DARWIN: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD_DARWIN: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD_DARWIN: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD_DARWIN: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD_DARWIN-NOT: call void @__sanitizer_cov_trace_pc
; CHECK_PC_GUARD_DARWIN: ret void
; CHECK_PC_GUARD_DARWIN: call void @__sanitizer_cov_trace_pc_guard_init(i32* bitcast (i32** @"\01section$start$__DATA$__sancov_guards" to i32*), i32* bitcast (i32** @"\01section$end$__DATA$__sancov_guards" to i32*))
