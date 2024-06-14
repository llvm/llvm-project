; RUN: llc < %s

; Make sure this doesn't crash

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-pc-win32"

define void @foo() {
  %1 = shufflevector <3 x i8> undef, <3 x i8> undef, <2 x i32> <i32 0, i32 1>
  %2 = shufflevector <2 x i8> %1, <2 x i8> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %3 = shufflevector <4 x i8> undef, <4 x i8> %2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  store <4 x i8> %3, ptr undef
  ret void
}
