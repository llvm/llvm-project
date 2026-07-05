; Test vector replicates, v4f32 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a byte-granularity replicate with the lowest useful value.
define <4 x float> @f1() {
; CHECK-LABEL: f1:
; CHECK: vrepib %v24, 1
; CHECK: br %r14
  ret <4 x float> <float 0x3820202020000000, float 0x3820202020000000,
                   float 0x3820202020000000, float 0x3820202020000000>
}

; Test a byte-granularity replicate with an arbitrary value.
define <4 x float> @f2() {
; CHECK-LABEL: f2:
; CHECK: vrepib %v24, -55
; CHECK: br %r14
  ret <4 x float> <float 0xc139393920000000, float 0xc139393920000000,
                   float 0xc139393920000000, float 0xc139393920000000>
}

; Test a byte-granularity replicate with the highest useful value.
define <4 x float> @f3() {
; CHECK-LABEL: f3:
; CHECK: vrepib %v24, -2
; CHECK: br %r14
  ret <4 x float> <float 0xc7dfdfdfc0000000, float 0xc7dfdfdfc0000000,
                   float 0xc7dfdfdfc0000000, float 0xc7dfdfdfc0000000>
}

; Test a halfword-granularity replicate with the lowest useful value.
define <4 x float> @f4() {
; CHECK-LABEL: f4:
; CHECK: vrepih %v24, 1
; CHECK: br %r14
  ret <4 x float> <float 0x37a0001000000000, float 0x37a0001000000000,
                   float 0x37a0001000000000, float 0x37a0001000000000>
}

; Test a halfword-granularity replicate with an arbitrary value.
define <4 x float> @f5() {
; CHECK-LABEL: f5:
; CHECK: vrepih %v24, 25650
; CHECK: br %r14
  ret <4 x float> <float 0x44864c8640000000, float 0x44864c8640000000,
                   float 0x44864c8640000000, float 0x44864c8640000000>
}

; Test a halfword-granularity replicate with the highest useful value.
define <4 x float> @f6() {
; CHECK-LABEL: f6:
; CHECK: vrepih %v24, -2
; CHECK: br %r14
  ret <4 x float> <float 0xffffdfffc0000000, float 0xffffdfffc0000000,
                   float 0xffffdfffc0000000, float 0xffffdfffc0000000>
}

; Test a word-granularity replicate with the lowest useful positive value.
define <4 x float> @f7() {
; CHECK-LABEL: f7:
; CHECK: vrepif %v24, 1
; CHECK: br %r14
  ret <4 x float> <float 0x36a0000000000000, float 0x36a0000000000000,
                   float 0x36a0000000000000, float 0x36a0000000000000>
}

; Test a word-granularity replicate with the highest in-range value.
define <4 x float> @f8() {
; CHECK-LABEL: f8:
; CHECK: vrepif %v24, 32767
; CHECK: br %r14
  ret <4 x float> <float 0x378fffc000000000, float 0x378fffc000000000,
                   float 0x378fffc000000000, float 0x378fffc000000000>
}

; Test a word-granularity replicate with the next highest value.
; This cannot use VREPIF.
define <4 x float> @f9() {
; CHECK-LABEL: f9:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <4 x float> <float 0x3790000000000000, float 0x3790000000000000,
                   float 0x3790000000000000, float 0x3790000000000000>
}

; Test a word-granularity replicate with the lowest in-range value.
define <4 x float> @f10() {
; CHECK-LABEL: f10:
; CHECK: vrepif %v24, -32768
; CHECK: br %r14
  ret <4 x float> <float 0xfffff00000000000, float 0xfffff00000000000,
                   float 0xfffff00000000000, float 0xfffff00000000000>
}

; Test a word-granularity replicate with the next lowest value.
; This cannot use VREPIF.
define <4 x float> @f11() {
; CHECK-LABEL: f11:
; CHECK-NOT: vrepif
; CHECK: br %r14
  ret <4 x float> <float 0xffffefffe0000000, float 0xffffefffe0000000,
                   float 0xffffefffe0000000, float 0xffffefffe0000000>
}

; Test a word-granularity replicate with the highest useful negative value.
define <4 x float> @f12() {
; CHECK-LABEL: f12:
; CHECK: vrepif %v24, -2
; CHECK: br %r14
  ret <4 x float> <float 0xffffffffc0000000, float 0xffffffffc0000000,
                   float 0xffffffffc0000000, float 0xffffffffc0000000>
}

; Test a doubleword-granularity replicate with the lowest useful positive
; value.
define <4 x float> @f13() {
; CHECK-LABEL: f13:
; CHECK: vrepig %v24, 1
; CHECK: br %r14
  ret <4 x float> <float 0.0, float 0x36a0000000000000,
                   float 0.0, float 0x36a0000000000000>
}

; Test a doubleword-granularity replicate with the highest in-range value.
define <4 x float> @f14() {
; CHECK-LABEL: f14:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <4 x float> <float 0.0, float 0x378fffc000000000,
                   float 0.0, float 0x378fffc000000000>
}

; Test a doubleword-granularity replicate with the next highest value.
; This cannot use VREPIG.
define <4 x float> @f15() {
; CHECK-LABEL: f15:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <4 x float> <float 0.0, float 0x3790000000000000,
                   float 0.0, float 0x3790000000000000>
}

; Test a doubleword-granularity replicate with the lowest in-range value.
define <4 x float> @f16() {
; CHECK-LABEL: f16:
; CHECK: vrepig %v24, -32768
; CHECK: br %r14
  ret <4 x float> <float 0xffffffffe0000000, float 0xfffff00000000000,
                   float 0xffffffffe0000000, float 0xfffff00000000000>
}

; Test a doubleword-granularity replicate with the next lowest value.
; This cannot use VREPIG.
define <4 x float> @f17() {
; CHECK-LABEL: f17:
; CHECK-NOT: vrepig
; CHECK: br %r14
  ret <4 x float> <float 0xffffffffe0000000, float 0xffffefffe0000000,
                   float 0xffffffffe0000000, float 0xffffefffe0000000>
}

; Test a doubleword-granularity replicate with the highest useful negative
; value.
define <4 x float> @f18() {
; CHECK-LABEL: f18:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <4 x float> <float 0xffffffffe0000000, float 0xffffffffc0000000,
                   float 0xffffffffe0000000, float 0xffffffffc0000000>
}

; Repeat f14 with undefs optimistically treated as 0, 32767.
define <4 x float> @f19() {
; CHECK-LABEL: f19:
; CHECK: vrepig %v24, 32767
; CHECK: br %r14
  ret <4 x float> <float undef, float undef,
                   float 0.0, float 0x378fffc000000000>
}

; Repeat f18 with undefs optimistically treated as -2, -1.
define <4 x float> @f20() {
; CHECK-LABEL: f20:
; CHECK: vrepig %v24, -2
; CHECK: br %r14
  ret <4 x float> <float 0xffffffffe0000000, float undef,
                   float undef, float 0xffffffffc0000000>
}
