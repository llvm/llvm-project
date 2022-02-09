; RUN: opt < %s -passes='print<cost-model>' 2>&1 -disable-output -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define <4 x i32> @test1(<4 x i32> %arg) {
  ; CHECK: cost of 1 {{.*}} call <4 x i32> @llvm.ctpop.v4i32
  %ctpop = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %arg)
  ret <4 x i32> %ctpop
}

declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>)
