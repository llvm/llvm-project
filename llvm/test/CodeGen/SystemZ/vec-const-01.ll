; Test vector byte masks, v16i8 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test an all-zeros vector.
define <16 x i8> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <16 x i8> zeroinitializer
}

; Test an all-ones vector.
define <16 x i8> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgbm %v24, 65535
; CHECK: br %r14
  ret <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1,
                 i8 -1, i8 -1, i8 -1, i8 -1,
                 i8 -1, i8 -1, i8 -1, i8 -1,
                 i8 -1, i8 -1, i8 -1, i8 -1>
}

; Test a mixed vector (mask 0x8c75).
define <16 x i8> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgbm %v24, 35957
; CHECK: br %r14
  ret <16 x i8> <i8 -1, i8 0, i8 0, i8 0,
                 i8 -1, i8 -1, i8 0, i8 0,
                 i8 0, i8 -1, i8 -1, i8 -1,
                 i8 0, i8 -1, i8 0, i8 -1>
}

; Test that undefs are treated as zero.
define <16 x i8> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgbm %v24, 35957
; CHECK: br %r14
  ret <16 x i8> <i8 -1, i8 undef, i8 undef, i8 undef,
                 i8 -1, i8 -1, i8 undef, i8 undef,
                 i8 undef, i8 -1, i8 -1, i8 -1,
                 i8 undef, i8 -1, i8 undef, i8 -1>
}

; Test that we don't use VGBM if one of the bytes is not 0 or 0xff.
define <16 x i8> @f5() {
; CHECK-LABEL: f5:
; CHECK-NOT: vgbm
; CHECK: br %r14
  ret <16 x i8> <i8 -1, i8 0, i8 0, i8 0,
                 i8 -1, i8 -1, i8 0, i8 1,
                 i8 0, i8 -1, i8 -1, i8 -1,
                 i8 0, i8 -1, i8 0, i8 -1>
}

; Test an all-zeros v2i8 that gets promoted to v16i8.
define <2 x i8> @f6() {
; CHECK-LABEL: f6:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <2 x i8> zeroinitializer
}

; Test a mixed v2i8 that gets promoted to v16i8 (mask 0x8000).
define <2 x i8> @f7() {
; CHECK-LABEL: f7:
; CHECK: vgbm %v24, 32768
; CHECK: br %r14
  ret <2 x i8> <i8 255, i8 0>
}

; Test an all-zeros v4i8 that gets promoted to v16i8.
define <4 x i8> @f8() {
; CHECK-LABEL: f8:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <4 x i8> zeroinitializer
}

; Test a mixed v4i8 that gets promoted to v16i8 (mask 0x9000).
define <4 x i8> @f9() {
; CHECK-LABEL: f9:
; CHECK: vgbm %v24, 36864
; CHECK: br %r14
  ret <4 x i8> <i8 255, i8 0, i8 0, i8 255>
}

; Test an all-zeros v8i8 that gets promoted to v16i8.
define <8 x i8> @f10() {
; CHECK-LABEL: f10:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <8 x i8> zeroinitializer
}

; Test a mixed v8i8 that gets promoted to v16i8 (mask 0xE500).
define <8 x i8> @f11() {
; CHECK-LABEL: f11:
; CHECK: vgbm %v24, 58624
; CHECK: br %r14
  ret <8 x i8> <i8 255, i8 255, i8 255, i8 0, i8 0, i8 255, i8 0, i8 255>
}
