; Tests that we don't insert __sanitizer_cov_trace_pc_guard_init or some such
; when there is no instrumentation.
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard  -S -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard  -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = global i32 0, align 4

; CHECK-NOT: call void
