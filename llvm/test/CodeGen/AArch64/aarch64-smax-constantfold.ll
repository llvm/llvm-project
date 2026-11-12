; RUN: llc -mtriple=aarch64-none-linux-gnu < %s -o -| FileCheck %s

; Function Attrs: nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.smax.v4i16(<4 x i16>, <4 x i16>)

; CHECK-LABEL: test
define <4 x i16> @test() {
entry:
; CHECK: movi	v{{[0-9]+}}.2d, #0000000000000000
  %0 = tail call <4 x i16> @llvm.aarch64.neon.smax.v4i16(<4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>, <4 x i16> zeroinitializer)
  ret <4 x i16> %0
}
