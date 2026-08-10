; Test vector replicates that use VECTOR GENERATE MASK, v8i16 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a word-granularity replicate with the lowest value that cannot use
; VREPIF.
define <8 x i16> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgmf %v24, 16, 16
; CHECK: br %r14
  ret <8 x i16> <i16 0, i16 32768, i16 0, i16 32768,
                 i16 0, i16 32768, i16 0, i16 32768>
}

; Test a word-granularity replicate that has the lower 17 bits set.
define <8 x i16> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgmf %v24, 15, 31
; CHECK: br %r14
  ret <8 x i16> <i16 1, i16 -1, i16 1, i16 -1,
                 i16 1, i16 -1, i16 1, i16 -1>
}

; Test a word-granularity replicate that has the upper 15 bits set.
define <8 x i16> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgmf %v24, 0, 14
; CHECK: br %r14
  ret <8 x i16> <i16 -2, i16 0, i16 -2, i16 0,
                 i16 -2, i16 0, i16 -2, i16 0>
}

; Test a word-granularity replicate that has middle bits set.
define <8 x i16> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgmf %v24, 12, 17
; CHECK: br %r14
  ret <8 x i16> <i16 15, i16 49152, i16 15, i16 49152,
                 i16 15, i16 49152, i16 15, i16 49152>
}

; Test a word-granularity replicate with a wrap-around mask.
define <8 x i16> @f5() {
; CHECK-LABEL: f5:
; CHECK: vgmf %v24, 17, 15
; CHECK: br %r14
  ret <8 x i16> <i16 -1, i16 32767, i16 -1, i16 32767,
                 i16 -1, i16 32767, i16 -1, i16 32767>
}

; Test a doubleword-granularity replicate with the lowest value that cannot
; use VREPIG.
define <8 x i16> @f6() {
; CHECK-LABEL: f6:
; CHECK: vgmg %v24, 48, 48
; CHECK: br %r14
  ret <8 x i16> <i16 0, i16 0, i16 0, i16 32768,
                 i16 0, i16 0, i16 0, i16 32768>
}

; Test a doubleword-granularity replicate that has the lower 22 bits set.
define <8 x i16> @f7() {
; CHECK-LABEL: f7:
; CHECK: vgmg %v24, 42, 63
; CHECK: br %r14
  ret <8 x i16> <i16 0, i16 0, i16 63, i16 -1,
                 i16 0, i16 0, i16 63, i16 -1>
}

; Test a doubleword-granularity replicate that has the upper 45 bits set.
define <8 x i16> @f8() {
; CHECK-LABEL: f8:
; CHECK: vgmg %v24, 0, 44
; CHECK: br %r14
  ret <8 x i16> <i16 -1, i16 -1, i16 -8, i16 0,
                 i16 -1, i16 -1, i16 -8, i16 0>
}

; Test a doubleword-granularity replicate that has middle bits set.
define <8 x i16> @f9() {
; CHECK-LABEL: f9:
; CHECK: vgmg %v24, 31, 42
; CHECK: br %r14
  ret <8 x i16> <i16 0, i16 1, i16 -32, i16 0,
                 i16 0, i16 1, i16 -32, i16 0>
}

; Test a doubleword-granularity replicate with a wrap-around mask.
define <8 x i16> @f10() {
; CHECK-LABEL: f10:
; CHECK: vgmg %v24, 18, 0
; CHECK: br %r14
  ret <8 x i16> <i16 32768, i16 16383, i16 -1, i16 -1,
                 i16 32768, i16 16383, i16 -1, i16 -1>
}

; Retest f1 with arbitrary undefs instead of 0s.
define <8 x i16> @f11() {
; CHECK-LABEL: f11:
; CHECK: vgmf %v24, 16, 16
; CHECK: br %r14
  ret <8 x i16> <i16 undef, i16 32768, i16 0, i16 32768,
                 i16 0, i16 32768, i16 undef, i16 32768>
}

; ...likewise f9.
define <8 x i16> @f12() {
; CHECK-LABEL: f12:
; CHECK: vgmg %v24, 31, 42
; CHECK: br %r14
  ret <8 x i16> <i16 undef, i16 1, i16 -32, i16 0,
                 i16 0, i16 1, i16 -32, i16 undef>
}
