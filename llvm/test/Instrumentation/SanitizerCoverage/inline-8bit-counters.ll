; Test -sanitizer-coverage-inline-8bit-counters=1
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-inline-8bit-counters=1  -S | FileCheck %s

; CHECK:      @__sancov_gen_ = private global [1 x i8] zeroinitializer, section "__sancov_cntrs", comdat($foo), align 1
; CHECK:      @__start___sancov_cntrs = extern_weak hidden global i8
; CHECK-NEXT: @__stop___sancov_cntrs = extern_weak hidden global i8

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo() {
entry:
; CHECK:  %0 = load i8, ptr @__sancov_gen_, align 1, !nosanitize
; CHECK:  %1 = add i8 %0, 1
; CHECK:  store i8 %1, ptr @__sancov_gen_, align 1, !nosanitize
  ret void
}
; CHECK: call void @__sanitizer_cov_8bit_counters_init(ptr @__start___sancov_cntrs, ptr @__stop___sancov_cntrs)
