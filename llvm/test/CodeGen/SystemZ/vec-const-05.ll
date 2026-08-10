; Test vector byte masks, v4f32 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test an all-zeros vector.
define <4 x float> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <4 x float> zeroinitializer
}

; Test an all-ones vector.
define <4 x float> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgbm %v24, 65535
; CHECK: br %r14
  ret <4 x float> <float 0xffffffffe0000000, float 0xffffffffe0000000,
                   float 0xffffffffe0000000, float 0xffffffffe0000000>
}

; Test a mixed vector (mask 0xc731).
define <4 x float> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgbm %v24, 50993
; CHECK: br %r14
  ret <4 x float> <float 0xffffe00000000000, float 0x381fffffe0000000,
                   float 0x379fffe000000000, float 0x371fe00000000000>
}

; Test that undefs are treated as zero (mask 0xc031).
define <4 x float> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgbm %v24, 49201
; CHECK: br %r14
  ret <4 x float> <float 0xffffe00000000000, float undef,
                   float 0x379fffe000000000, float 0x371fe00000000000>
}

; Test that we don't use VGBM if one of the bytes is not 0 or 0xff.
define <4 x float> @f5() {
; CHECK-LABEL: f5:
; CHECK-NOT: vgbm
; CHECK: br %r14
  ret <4 x float> <float 0xffffe00000000000, float 0x381fffffc0000000,
                   float 0x379fffe000000000, float 0x371fe00000000000>
}

; Test an all-zeros v2f32 that gets promoted to v4f32.
define <2 x float> @f6() {
; CHECK-LABEL: f6:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <2 x float> zeroinitializer
}

; Test a mixed v2f32 that gets promoted to v4f32 (mask 0xc700).
define <2 x float> @f7() {
; CHECK-LABEL: f7:
; CHECK: vgbm %v24, 50944
; CHECK: br %r14
  ret <2 x float> <float 0xffffe00000000000, float 0x381fffffe0000000>
}
