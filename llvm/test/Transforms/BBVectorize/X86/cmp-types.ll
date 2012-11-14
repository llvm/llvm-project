target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -bb-vectorize -S | FileCheck %s

%"struct.btSoftBody" = type { float, float, float*, i8 }

define void @test1(%"struct.btSoftBody"* %n1, %"struct.btSoftBody"* %n2) uwtable align 2 {
entry:
  %tobool15 = icmp ne %"struct.btSoftBody"* %n1, null
  %cond16 = zext i1 %tobool15 to i32
  %tobool21 = icmp ne %"struct.btSoftBody"* %n2, null
  %cond22 = zext i1 %tobool21 to i32
  ret void
; CHECK: @test1
}

