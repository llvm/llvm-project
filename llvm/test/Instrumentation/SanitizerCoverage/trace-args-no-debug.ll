; Test that trace-args works WITHOUT debug info (-g). A function with no
; DISubprogram carries no #dbg_value records, so the source-argument map stays
; empty and the pass falls back to tracing each IR argument directly (no
; source-level struct-field offsets). This keeps trace-args usable on userspace
; / optimized code built without -g, not only debug kernels. opt runs the
; verifier, so a successful run also proves the emitted IR is well-formed.

; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-args -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @no_debug(ptr %p, i32 %x) {
entry:
  ret void
}
; CHECK-LABEL: define void @no_debug(ptr %p, i32 %x)
; pointer arg 0 is traced directly (no spill, no field offsets):
; CHECK-DAG: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @no_debug to i64), i32 0, i32 8, ptr %p, ptr null, i32 0)
; scalar arg 1 is spilled to a stack slot then traced:
; CHECK-DAG: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @no_debug to i64), i32 1, i32 4, ptr %{{.*}}, ptr null, i32 0)
