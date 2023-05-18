; Test vector replicates that use VECTOR GENERATE MASK, v16i8 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a word-granularity replicate with the lowest value that cannot use
; VREPIF.
define <16 x i8> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgmf %v24, 16, 16
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 0, i8 128, i8 0,
                 i8 0, i8 0, i8 128, i8 0,
                 i8 0, i8 0, i8 128, i8 0,
                 i8 0, i8 0, i8 128, i8 0>
}

; Test a word-granularity replicate that has the lower 17 bits set.
define <16 x i8> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgmf %v24, 15, 31
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 1, i8 255, i8 255,
                 i8 0, i8 1, i8 255, i8 255,
                 i8 0, i8 1, i8 255, i8 255,
                 i8 0, i8 1, i8 255, i8 255>
}

; Test a word-granularity replicate that has the upper 15 bits set.
define <16 x i8> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgmf %v24, 0, 14
; CHECK: br %r14
  ret <16 x i8> <i8 255, i8 254, i8 0, i8 0,
                 i8 255, i8 254, i8 0, i8 0,
                 i8 255, i8 254, i8 0, i8 0,
                 i8 255, i8 254, i8 0, i8 0>
}

; Test a word-granularity replicate that has middle bits set.
define <16 x i8> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgmf %v24, 12, 17
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 15, i8 192, i8 0,
                 i8 0, i8 15, i8 192, i8 0,
                 i8 0, i8 15, i8 192, i8 0,
                 i8 0, i8 15, i8 192, i8 0>
}

; Test a word-granularity replicate with a wrap-around mask.
define <16 x i8> @f5() {
; CHECK-LABEL: f5:
; CHECK: vgmf %v24, 17, 15
; CHECK: br %r14
  ret <16 x i8> <i8 255, i8 255, i8 127, i8 255,
                 i8 255, i8 255, i8 127, i8 255,
                 i8 255, i8 255, i8 127, i8 255,
                 i8 255, i8 255, i8 127, i8 255>
}

; Test a doubleword-granularity replicate with the lowest value that cannot
; use VREPIG.
define <16 x i8> @f6() {
; CHECK-LABEL: f6:
; CHECK: vgmg %v24, 48, 48
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 0, i8 0, i8 0,
                 i8 0, i8 0, i8 128, i8 0,
                 i8 0, i8 0, i8 0, i8 0,
                 i8 0, i8 0, i8 128, i8 0>
}

; Test a doubleword-granularity replicate that has the lower 22 bits set.
define <16 x i8> @f7() {
; CHECK-LABEL: f7:
; CHECK: vgmg %v24, 42, 63
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 0, i8 0, i8 0,
                 i8 0, i8 63, i8 255, i8 255,
                 i8 0, i8 0, i8 0, i8 0,
                 i8 0, i8 63, i8 255, i8 255>
}

; Test a doubleword-granularity replicate that has the upper 45 bits set.
define <16 x i8> @f8() {
; CHECK-LABEL: f8:
; CHECK: vgmg %v24, 0, 44
; CHECK: br %r14
  ret <16 x i8> <i8 255, i8 255, i8 255, i8 255,
                 i8 255, i8 248, i8 0, i8 0,
                 i8 255, i8 255, i8 255, i8 255,
                 i8 255, i8 248, i8 0, i8 0>
}

; Test a doubleword-granularity replicate that has middle bits set.
define <16 x i8> @f9() {
; CHECK-LABEL: f9:
; CHECK: vgmg %v24, 31, 42
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 0, i8 0, i8 1,
                 i8 255, i8 224, i8 0, i8 0,
                 i8 0, i8 0, i8 0, i8 1,
                 i8 255, i8 224, i8 0, i8 0>
}

; Test a doubleword-granularity replicate with a wrap-around mask.
define <16 x i8> @f10() {
; CHECK-LABEL: f10:
; CHECK: vgmg %v24, 18, 0
; CHECK: br %r14
  ret <16 x i8> <i8 128, i8 0, i8 63, i8 255,
                 i8 255, i8 255, i8 255, i8 255,
                 i8 128, i8 0, i8 63, i8 255,
                 i8 255, i8 255, i8 255, i8 255>
}

; Retest f1 with arbitrary undefs instead of 0s.
define <16 x i8> @f11() {
; CHECK-LABEL: f11:
; CHECK: vgmf %v24, 16, 16
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 undef, i8 128, i8 0,
                 i8 0, i8 0, i8 128, i8 undef,
                 i8 undef, i8 0, i8 128, i8 0,
                 i8 undef, i8 undef, i8 128, i8 0>
}

; Try a case where we want consistent undefs to be treated as 0.
define <16 x i8> @f12() {
; CHECK-LABEL: f12:
; CHECK: vgmf %v24, 15, 23
; CHECK: br %r14
  ret <16 x i8> <i8 undef, i8 1, i8 255, i8 0,
                 i8 undef, i8 1, i8 255, i8 0,
                 i8 undef, i8 1, i8 255, i8 0,
                 i8 undef, i8 1, i8 255, i8 0>
}

; ...and again with the lower bits of the replicated constant.
define <16 x i8> @f13() {
; CHECK-LABEL: f13:
; CHECK: vgmf %v24, 15, 22
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 1, i8 254, i8 undef,
                 i8 0, i8 1, i8 254, i8 undef,
                 i8 0, i8 1, i8 254, i8 undef,
                 i8 0, i8 1, i8 254, i8 undef>
}

; Try a case where we want consistent undefs to be treated as -1.
define <16 x i8> @f14() {
; CHECK-LABEL: f14:
; CHECK: vgmf %v24, 28, 8
; CHECK: br %r14
  ret <16 x i8> <i8 undef, i8 128, i8 0, i8 15,
                 i8 undef, i8 128, i8 0, i8 15,
                 i8 undef, i8 128, i8 0, i8 15,
                 i8 undef, i8 128, i8 0, i8 15>
}

; ...and again with the lower bits of the replicated constant.
define <16 x i8> @f15() {
; CHECK-LABEL: f15:
; CHECK: vgmf %v24, 18, 3
; CHECK: br %r14
  ret <16 x i8> <i8 240, i8 0, i8 63, i8 undef,
                 i8 240, i8 0, i8 63, i8 undef,
                 i8 240, i8 0, i8 63, i8 undef,
                 i8 240, i8 0, i8 63, i8 undef>
}

; Repeat f9 with arbitrary undefs.
define <16 x i8> @f16() {
; CHECK-LABEL: f16:
; CHECK: vgmg %v24, 31, 42
; CHECK: br %r14
  ret <16 x i8> <i8 undef, i8 0, i8 undef, i8 1,
                 i8 255, i8 undef, i8 0, i8 0,
                 i8 0, i8 0, i8 0, i8 1,
                 i8 undef, i8 224, i8 undef, i8 undef>
}

; Try a case where we want some consistent undefs to be treated as 0
; and some to be treated as 255.
define <16 x i8> @f17() {
; CHECK-LABEL: f17:
; CHECK: vgmg %v24, 23, 35
; CHECK: br %r14
  ret <16 x i8> <i8 0, i8 undef, i8 1, i8 undef,
                 i8 240, i8 undef, i8 0, i8 0,
                 i8 0, i8 undef, i8 1, i8 undef,
                 i8 240, i8 undef, i8 0, i8 0>
}
