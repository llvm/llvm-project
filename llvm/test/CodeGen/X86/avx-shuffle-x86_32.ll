; RUN: llc < %s -mtriple=i686-pc-win32 -mcpu=corei7-avx -mattr=+avx | FileCheck %s

define <4 x i64> @test1(<4 x i64> %a) nounwind {
 %b = shufflevector <4 x i64> %a, <4 x i64> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
 ret <4 x i64>%b
 ; CHECK: test1:
 ; CHECK-NOT: vinsertf128
 }
