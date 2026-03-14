; Test -sanitizer-coverage-trace-compares=1
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-compares=1  -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define i32 @foo(i32 %a, i32 %b) #0 {
entry:
  %cmp = icmp slt i32 %a, %b
; CHECK: call void @__sanitizer_cov_trace_cmp4
; CHECK-NEXT: icmp slt i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}
