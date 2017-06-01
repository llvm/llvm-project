; RUN: llc < %s -mtriple=i386-apple-darwin10 -mcpu=corei7-avx -mattr=+avx | FileCheck %s

;; Check that FP_TO_SINT and FP_TO_UINT generate convert with truncate

; CHECK-LABEL: test1:
; CHECK: vcvttpd2dq
; CHECK: ret
; CHECK-LABEL: test2:
; CHECK: vcvttpd2dq
; CHECK: ret

define <4 x i8> @test1(<4 x double> %d) {
  %c = fptoui <4 x double> %d to <4 x i8>
  ret <4 x i8> %c
}
define <4 x i8> @test2(<4 x double> %d) {
  %c = fptosi <4 x double> %d to <4 x i8>
  ret <4 x i8> %c
}
