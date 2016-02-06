; RUN: llc -mcpu=pwr7 -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s
define <8 x i16> @test(<4 x i32> %a) {
entry:
  %0 = tail call <8 x i16> @llvm.ppc.altivec.vpkswss(<4 x i32> %a, <4 x i32> %a)
  ret <8 x i16> %0
}
; CHECK: vpkswss 2,

declare <8 x i16> @llvm.ppc.altivec.vpkswss(<4 x i32>, <4 x i32>)
