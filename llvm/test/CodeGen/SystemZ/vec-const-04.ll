; Test vector byte masks, v2i64 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test an all-zeros vector.
define <2 x i64> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <2 x i64> zeroinitializer
}

; Test an all-ones vector.
define <2 x i64> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgbm %v24, 65535
; CHECK: br %r14
  ret <2 x i64> <i64 -1, i64 -1>
}

; Test a mixed vector (mask 0x8c76).
define <2 x i64> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgbm %v24, 35958
; CHECK: br %r14
  ret <2 x i64> <i64 18374686483966525440, i64 72057589759737600>
}

; Test that undefs are treated as zero (mask 0x8c00).
define <2 x i64> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgbm %v24, 35840
; CHECK: br %r14
  ret <2 x i64> <i64 18374686483966525440, i64 undef>
}

; Test that we don't use VGBM if one of the bytes is not 0 or 0xff.
define <2 x i64> @f5() {
; CHECK-LABEL: f5:
; CHECK-NOT: vgbm
; CHECK: br %r14
  ret <2 x i64> <i64 18374686483966525441, i64 72057589759737600>
}
