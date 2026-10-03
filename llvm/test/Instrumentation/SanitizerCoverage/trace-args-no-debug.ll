; Argument tracing without debug info: there is nothing to map the IR arguments
; back to, so they are reported positionally and without field offset tables.
; This keeps trace-args usable on code built without -g; the ABI's view of the
; arguments is then what a consumer sees.
;
; opt runs the verifier, so a passing run also proves the emitted IR is
; well-formed.
;
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-args -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i64, i64, i64 }

define void @no_debug(ptr %p, i32 %x) {
entry:
  ret void
}
; CHECK-LABEL: define void @no_debug(
; With no type to describe its pointee, a pointer is reported as its own value.
; CHECK: %[[P:[0-9]+]] = ptrtoint ptr %p to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @no_debug to i64), i32 0, i32 8, i64 %[[P]], ptr null, i32 0)
; CHECK: %[[X:[0-9]+]] = zext i32 %x to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @no_debug to i64), i32 1, i32 4, i64 %[[X]], ptr null, i32 0)

; A struct-return pointer is ABI-inserted and has no source-level counterpart,
; so it is skipped here as well and %x keeps index 0.
define void @no_debug_sret(ptr sret(%struct.S) %0, i32 %x) {
entry:
  ret void
}
; CHECK-LABEL: define void @no_debug_sret(
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @no_debug_sret to i64), i32 0, i32 4, i64 %{{[0-9]+}}, ptr null, i32 0)
; CHECK-NOT: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @no_debug_sret to i64), i32 1
