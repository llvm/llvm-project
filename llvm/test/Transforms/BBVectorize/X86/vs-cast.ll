target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -bb-vectorize -S | FileCheck %s

define void @main() nounwind uwtable {
entry:
  %0 = bitcast <2 x i64> undef to i128
  %1 = bitcast <2 x i64> undef to i128
  ret void
; CHECK: @main
}

