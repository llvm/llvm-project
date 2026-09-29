; Checks that a function with no-return in the entry block is not instrumented.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard -S | FileCheck %s
; CHECK-NOT: call void @__sanitizer_cov_trace_pc_guard

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z3foov() noinline nounwind optnone uwtable {
entry:
  call void @abort() noreturn nounwind
  unreachable

return:                                           ; No predecessors!
  ret void
}

declare dso_local void @abort() noreturn nounwind
