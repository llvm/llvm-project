; Test vector replicates that use VECTOR GENERATE MASK, v4f32 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a word-granularity replicate with the lowest value that cannot use
; VREPIF.
define <4 x float> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgmf %v24, 16, 16
; CHECK: br %r14
  ret <4 x float> <float 0x3790000000000000, float 0x3790000000000000,
                   float 0x3790000000000000, float 0x3790000000000000>
}

; Test a word-granularity replicate that has the lower 17 bits set.
define <4 x float> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgmf %v24, 15, 31
; CHECK: br %r14
  ret <4 x float> <float 0x37affff000000000, float 0x37affff000000000,
                   float 0x37affff000000000, float 0x37affff000000000>
}

; Test a word-granularity replicate that has the upper 15 bits set.
define <4 x float> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgmf %v24, 0, 14
; CHECK: br %r14
  ret <4 x float> <float 0xffffc00000000000, float 0xffffc00000000000,
                   float 0xffffc00000000000, float 0xffffc00000000000>
}

; Test a word-granularity replicate that has middle bits set.
define <4 x float> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgmf %v24, 2, 8
; CHECK: br %r14
  ret <4 x float> <float 0x3ff0000000000000, float 0x3ff0000000000000,
                   float 0x3ff0000000000000, float 0x3ff0000000000000>
}

; Test a word-granularity replicate with a wrap-around mask.
define <4 x float> @f5() {
; CHECK-LABEL: f5:
; CHECK: vgmf %v24, 9, 1
; CHECK: br %r14
  ret <4 x float> <float 0xc00fffffe0000000, float 0xc00fffffe0000000,
                   float 0xc00fffffe0000000, float 0xc00fffffe0000000>
}

; Test a doubleword-granularity replicate with the lowest value that cannot
; use VREPIG.
define <4 x float> @f6() {
; CHECK-LABEL: f6:
; CHECK: vgmg %v24, 48, 48
; CHECK: br %r14
  ret <4 x float> <float 0.0, float 0x3790000000000000,
                   float 0.0, float 0x3790000000000000>
}

; Test a doubleword-granularity replicate that has the lower 22 bits set.
define <4 x float> @f7() {
; CHECK-LABEL: f7:
; CHECK: vgmg %v24, 42, 63
; CHECK: br %r14
  ret <4 x float> <float 0.0, float 0x37ffffff80000000,
                   float 0.0, float 0x37ffffff80000000>
}

; Test a doubleword-granularity replicate that has the upper 45 bits set.
define <4 x float> @f8() {
; CHECK-LABEL: f8:
; CHECK: vgmg %v24, 0, 44
; CHECK: br %r14
  ret <4 x float> <float 0xffffffffe0000000, float 0xffff000000000000,
                   float 0xffffffffe0000000, float 0xffff000000000000>
}

; Test a doubleword-granularity replicate that has middle bits set.
define <4 x float> @f9() {
; CHECK-LABEL: f9:
; CHECK: vgmg %v24, 34, 41
; CHECK: br %r14
  ret <4 x float> <float 0.0, float 0x3ff8000000000000,
                   float 0.0, float 0x3ff8000000000000>
}

; Test a doubleword-granularity replicate with a wrap-around mask.
define <4 x float> @f10() {
; CHECK-LABEL: f10:
; CHECK: vgmg %v24, 32, 0
; CHECK: br %r14
  ret <4 x float> <float 0x8000000000000000, float 0xffffffffe0000000,
                   float 0x8000000000000000, float 0xffffffffe0000000>
}
