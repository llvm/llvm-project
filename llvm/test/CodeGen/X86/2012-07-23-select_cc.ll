; RUN: llc < %s -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; PR 13428

declare void @use(double)

define void @test() {
entry:
  call void @use(double 1.000000e+00)
  %A = icmp eq i64 undef, 2
  %B = zext i1 %A to i32
  %C = sitofp i32 %B to double
  call void @use(double %C)
  call void @use(double 0.000000e+00)
  unreachable
}
