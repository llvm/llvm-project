; Test vector byte masks, v4i32 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test an all-zeros vector.
define <4 x i32> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <4 x i32> zeroinitializer
}

; Test an all-ones vector.
define <4 x i32> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgbm %v24, 65535
; CHECK: br %r14
  ret <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
}

; Test a mixed vector (mask 0x8c76).
define <4 x i32> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgbm %v24, 35958
; CHECK: br %r14
  ret <4 x i32> <i32 4278190080, i32 4294901760, i32 16777215, i32 16776960>
}

; Test that undefs are treated as zero (mask 0x8076).
define <4 x i32> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgbm %v24, 32886
; CHECK: br %r14
  ret <4 x i32> <i32 4278190080, i32 undef, i32 16777215, i32 16776960>
}

; Test that we don't use VGBM if one of the bytes is not 0 or 0xff.
define <4 x i32> @f5() {
; CHECK-LABEL: f5:
; CHECK-NOT: vgbm
; CHECK: br %r14
  ret <4 x i32> <i32 4278190080, i32 1, i32 16777215, i32 16776960>
}

; Test an all-zeros v2i32 that gets promoted to v4i32.
define <2 x i32> @f6() {
; CHECK-LABEL: f6:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <2 x i32> zeroinitializer
}

; Test a mixed v2i32 that gets promoted to v4i32 (mask 0xae00).
define <2 x i32> @f7() {
; CHECK-LABEL: f7:
; CHECK: vgbm %v24, 44544
; CHECK: br %r14
  ret <2 x i32> <i32 4278255360, i32 -256>
}
