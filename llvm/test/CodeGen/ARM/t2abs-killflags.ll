; RUN: llc %s -o - -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7s-unknown-unknown"

; Test that we don't crash the machine verifier when expanding t2ABS.
; It was applying the kill flag to its source, even if the original t2ABS didn't
; kill the operand prior to pseudo expansion.

; Function Attrs: noreturn
declare void @foo(i32, i32) #0

; CHECK-LABEL: @test
; CHECK: rsbmi
define void @test(i32 %arg) {
  %cmp58 = icmp slt i32 %arg, 0
  %sub62 = sub nsw i32 0, %arg
  %l.1 = select i1 %cmp58, i32 %sub62, i32 %arg
  call void @foo(i32 %l.1, i32 %arg) #0
  unreachable
}

attributes #0 = { noreturn }
