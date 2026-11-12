; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <4 x float> @foo(<4 x i16> %a) nounwind {
; CHECK-LABEL: foo:
; CHECK: ushll.4s	v0, v0, #0
; CHECK-NEXT: ucvtf.4s	v0, v0
; CHECK-NEXT: ret
  %vcvt.i = uitofp <4 x i16> %a to <4 x float>
  ret <4 x float> %vcvt.i
}

define <4 x float> @bar(<4 x i16> %a) nounwind {
; CHECK-LABEL: bar:
; CHECK: sshll.4s	v0, v0, #0
; CHECK-NEXT: scvtf.4s	v0, v0
; CHECK-NEXT: ret
  %vcvt.i = sitofp <4 x i16> %a to <4 x float>
  ret <4 x float> %vcvt.i
}
