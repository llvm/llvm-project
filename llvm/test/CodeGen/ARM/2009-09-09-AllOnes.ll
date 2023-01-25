; RUN: llc -mattr=+neon < %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv7-elf"

define void @foo() {
entry:
  %0 = insertelement <4 x i32> undef, i32 -1, i32 3
  store <4 x i32> %0, ptr undef, align 16
  unreachable
}
