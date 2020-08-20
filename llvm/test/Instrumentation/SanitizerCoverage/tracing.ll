; Test -sanitizer-coverage-experimental-tracing
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc  -S -enable-new-pm=0 | FileCheck %s --check-prefix=CHECK_PC
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard  -S -enable-new-pm=0 | FileCheck %s --check-prefix=CHECK_PC_GUARD
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard  -S -mtriple=x86_64-apple-macosx -enable-new-pm=0 | FileCheck %s --check-prefix=CHECK_PC_GUARD_DARWIN

; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc  -S | FileCheck %s --check-prefix=CHECK_PC
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard  -S | FileCheck %s --check-prefix=CHECK_PC_GUARD
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard  -S -mtriple=x86_64-apple-macosx | FileCheck %s --check-prefix=CHECK_PC_GUARD_DARWIN

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

define available_externally void @external_bar(i32* %a) sanitize_address {
entry:
  ret void
}

declare void @longjmp(i8*) noreturn

; We expect three coverage points here for each BB.
define void @cond_longjmp(i1 %cond, i8* %jmp_buf) sanitize_address {
entry:
  br i1 %cond, label %lj, label %done
done:
  ret void
lj:
  call void @longjmp(i8* %jmp_buf)
  unreachable
}


; CHECK_PC-LABEL: define void @foo
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC-NOT: call void @__sanitizer_cov_trace_pc
; CHECK_PC: ret void
; CHECK_PC-NOT: call void @__sanitizer_cov_module_init
; CHECK_PC-LABEL: @cond_longjmp
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC: ret void
; CHECK_PC: call void @__sanitizer_cov_trace_pc
; CHECK_PC: call void @longjmp
; CHECK_PC: unreachable

; CHECK_PC_GUARD: section "__sancov_guards", comdat($foo), align 4
; CHECK_PC_GUARD-LABEL: define void @foo
; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD-NOT: call void @__sanitizer_cov_trace_pc
; CHECK_PC_GUARD: ret void
; CHECK_PC_GUARD-LABEL: @external_bar
; CHECK_PC_GUARD-NOT: call void @__sanitizer_cov_trace_pc
; CHECK_PC_GUARD: ret void
; CHECK_PC_GUARD-LABEL: @cond_longjmp
; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD: ret void
; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD: call void @longjmp
; CHECK_PC_GUARD: unreachable

; CHECK_PC_GUARD: call void @__sanitizer_cov_trace_pc_guard_init(i32* @__start___sancov_guards, i32* @__stop___sancov_guards)

; CHECK_PC_GUARD_DARWIN-LABEL: define void @foo
; CHECK_PC_GUARD_DARWIN: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD_DARWIN: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD_DARWIN: call void @__sanitizer_cov_trace_pc_guard
; CHECK_PC_GUARD_DARWIN-NOT: call void @__sanitizer_cov_trace_pc
; CHECK_PC_GUARD_DARWIN: ret void
; CHECK_PC_GUARD_DARWIN: call void @__sanitizer_cov_trace_pc_guard_init(i32* @"\01section$start$__DATA$__sancov_guards", i32* @"\01section$end$__DATA$__sancov_guards")
