; RUN: llc < %s
; PR13111

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "armv7-none-linux-gnueabi"

define void @test_hi_char8() noinline {
entry:
  %0 = load <4 x i8>, ptr undef, align 1
  store <4 x i8> %0, ptr null, align 4
  ret void
}
