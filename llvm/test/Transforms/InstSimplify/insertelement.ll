; RUN: opt -S -instsimplify < %s | FileCheck %s

define <4 x i32> @test1(<4 x i32> %A) {
  %I = insertelement <4 x i32> %A, i32 5, i64 4294967296
  ; CHECK: ret <4 x i32> undef
  ret <4 x i32> %I
}

define <4 x i32> @test2(<4 x i32> %A) {
  %I = insertelement <4 x i32> %A, i32 5, i64 4
  ; CHECK: ret <4 x i32> undef
  ret <4 x i32> %I
}

define <4 x i32> @test3(<4 x i32> %A) {
  %I = insertelement <4 x i32> %A, i32 5, i64 1
  ; CHECK: ret <4 x i32> %I
  ret <4 x i32> %I
}

define <4 x i32> @test4(<4 x i32> %A) {
  %I = insertelement <4 x i32> %A, i32 5, i128 100
  ; CHECK: ret <4 x i32> undef
  ret <4 x i32> %I
}

define <4 x i32> @test5(<4 x i32> %A) {
  %I = insertelement <4 x i32> %A, i32 5, i64 undef
  ; CHECK: ret <4 x i32> undef
  ret <4 x i32> %I
}
