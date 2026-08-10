; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-pc-guard -sanitizer-coverage-inline-8bit-counters -S | FileCheck %s

; Module ctors should have stable names across modules, not something like
; @sancov.module_ctor.3 that may cause duplicate ctors after linked together.

; CHECK: define internal void @sancov.module_ctor_trace_pc_guard() #[[#]] comdat {
; CHECK: define internal void @sancov.module_ctor_8bit_counters() #[[#]] comdat {

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo() {
  ret void
}
