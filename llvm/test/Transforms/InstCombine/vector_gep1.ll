; RUN: opt -instcombine %s -disable-output
; RUN: opt -instsimplify %s -disable-output
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@G1 = global i8 zeroinitializer

define <2 x i1> @test(<2 x i8*> %a, <2 x i8*> %b) {
   %A = icmp eq <2 x i8*> %a, %b
   ret <2 x i1> %A
}

define <2 x i1> @test2(<2 x i8*> %a) {
  %A = inttoptr <2 x i32> <i32 1, i32 2> to <2 x i8*>
  %B = icmp ult <2 x i8*> %A, zeroinitializer
  ret <2 x i1> %B
}

define <2 x i1> @test3(<2 x i8*> %a) {
  %g = getelementptr <2 x i8*> %a, <2 x i32> <i32 1, i32 0>
  %B = icmp ult <2 x i8*> %g, zeroinitializer
  ret <2 x i1> %B
}

define <1 x i1> @test4(<1 x i8*> %a) {
  %g = getelementptr <1 x i8*> %a, <1 x i32> <i32 1>
  %B = icmp ult <1 x i8*> %g, zeroinitializer
  ret <1 x i1> %B
}

define <2 x i1> @test5(<2 x i8*> %a) {
  %w = getelementptr <2 x i8*> %a, <2 x i32> zeroinitializer
  %e = getelementptr <2 x i8*> %w, <2 x i32> <i32 5, i32 9>
  %g = getelementptr <2 x i8*> %e, <2 x i32> <i32 1, i32 0>
  %B = icmp ult <2 x i8*> %g, zeroinitializer
  ret <2 x i1> %B
}
