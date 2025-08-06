; RUN: llc -mtriple=aarch64-- -O2 -mattr=+neon < %s | FileCheck %s

; CHECK-LABEL: test_avgceil_u
; CHECK: uhadd v0.8b, v0.8b, v1.8b
define <8 x i8> @test_avgceil_u(<8 x i16> %a, <8 x i16> %b) {
  %ta = trunc <8 x i16> %a to <8 x i8>
  %tb = trunc <8 x i16> %b to <8 x i8>
  %res = call <8 x i8> @llvm.aarch64.neon.uhadd.v8i8(<8 x i8> %ta, <8 x i8> %tb)
  ret <8 x i8> %res
}

; CHECK-LABEL: test_avgceil_s
; CHECK: shadd v0.8b, v0.8b, v1.8b
define <8 x i8> @test_avgceil_s(<8 x i16> %a, <8 x i16> %b) {
  %ta = trunc <8 x i16> %a to <8 x i8>
  %tb = trunc <8 x i16> %b to <8 x i8>
  %res = call <8 x i8> @llvm.aarch64.neon.shadd.v8i8(<8 x i8> %ta, <8 x i8> %tb)
  ret <8 x i8> %res
}

; CHECK-LABEL: test_avgfloor_u
; CHECK: urhadd v0.8b, v0.8b, v1.8b
define <8 x i8> @test_avgfloor_u(<8 x i16> %a, <8 x i16> %b) {
  %ta = trunc <8 x i16> %a to <8 x i8>
  %tb = trunc <8 x i16> %b to <8 x i8>
  %res = call <8 x i8> @llvm.aarch64.neon.urhadd.v8i8(<8 x i8> %ta, <8 x i8> %tb)
  ret <8 x i8> %res
}

; CHECK-LABEL: test_avgfloor_s
; CHECK: srhadd v0.8b, v0.8b, v1.8b
define <8 x i8> @test_avgfloor_s(<8 x i16> %a, <8 x i16> %b) {
  %ta = trunc <8 x i16> %a to <8 x i8>
  %tb = trunc <8 x i16> %b to <8 x i8>
  %res = call <8 x i8> @llvm.aarch64.neon.srhadd.v8i8(<8 x i8> %ta, <8 x i8> %tb)
  ret <8 x i8> %res
}

declare <8 x i8> @llvm.aarch64.neon.uhadd.v8i8(<8 x i8>, <8 x i8>)
declare <8 x i8> @llvm.aarch64.neon.shadd.v8i8(<8 x i8>, <8 x i8>)
declare <8 x i8> @llvm.aarch64.neon.urhadd.v8i8(<8 x i8>, <8 x i8>)
declare <8 x i8> @llvm.aarch64.neon.srhadd.v8i8(<8 x i8>, <8 x i8>)

