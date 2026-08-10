target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"
; RUN: llc -verify-machineinstrs -mcpu=a2 -enable-misched -enable-aa-sched-mi < %s | FileCheck %s

define i8 @test1(ptr noalias %a, ptr noalias %b, ptr noalias %c) nounwind {
entry:
  %q = load i8, ptr %b
  call void @llvm.prefetch(ptr %a, i32 0, i32 3, i32 1)
  %r = load i8, ptr %c
  %s = add i8 %q, %r
  ret i8 %s
}

declare void @llvm.prefetch(ptr, i32, i32, i32)

; Test that we've moved the second load to before the dcbt to better
; hide its latency.
; CHECK: @test1
; CHECK: lbz
; CHECK: lbz
; CHECK: dcbt

