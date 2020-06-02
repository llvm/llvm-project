; RUN: llc < %s -mtriple=aarch64-- | FileCheck %s

; Check that building up a vector w/ only one non-zero lane initializes
; efficiently.

define <8 x i8> @v8i8z(i8 %t, i8 %s) nounwind {
  %v = insertelement <8 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 undef>, i8 %s, i32 7
  ret <8 x i8> %v

; CHECK-LABEL: v8i8z
; CHECK: movi d[[R:[0-9]+]], #0
; CHECK: mov  v[[R]].b[7], w{{[0-9]+}}
}

define <16 x i8> @v16i8z(i8 %t, i8 %s) nounwind {
  %v = insertelement <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 undef>, i8 %s, i32 15
  ret <16 x i8> %v

; CHECK-LABEL: v16i8z:
; CHECK: movi v[[R:[0-9]+]].2d, #0
; CHECK: mov  v[[R]].b[15], w{{[0-9]+}}
}

define <4 x i16> @v4i16z(i16 %t, i16 %s) nounwind {
  %v = insertelement <4 x i16> <i16 0, i16 0, i16 0, i16 undef>, i16 %s, i32 3
  ret <4 x i16> %v

; CHECK-LABEL: v4i16z:
; CHECK: movi d[[R:[0-9]+]], #0
; CHECK: mov  v[[R]].h[3], w{{[0-9]+}}
}

define <8 x i16> @v8i16z(i16 %t, i16 %s) nounwind {
  %v = insertelement <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 undef>, i16 %s, i32 7
  ret <8 x i16> %v

; CHECK-LABEL: v8i16z:
; CHECK: movi v[[R:[0-9]+]].2d, #0
; CHECK: mov  v[[R]].h[7], w{{[0-9]+}}
}

define <2 x i32> @v2i32z(i32 %t, i32 %s) nounwind {
  %v = insertelement <2 x i32> <i32 0, i32 undef>, i32 %s, i32 1
  ret <2 x i32> %v

; CHECK-LABEL: v2i32z:
; CHECK: movi d[[R:[0-9]+]], #0
; CHECK: mov  v[[R]].s[1], w{{[0-9]+}}
}

define <4 x i32> @v4i32z(i32 %t, i32 %s) nounwind {
  %v = insertelement <4 x i32> <i32 0, i32 0, i32 0, i32 undef>, i32 %s, i32 3
  ret <4 x i32> %v

; CHECK-LABEL: v4i32z:
; CHECK: movi v[[R:[0-9]+]].2d, #0
; CHECK: mov  v[[R]].s[3], w{{[0-9]+}}
}

define <2 x i64> @v2i64z(i64 %t, i64 %s) nounwind {
  %v = insertelement <2 x i64> <i64 0, i64 undef>, i64 %s, i32 1
  ret <2 x i64> %v

; CHECK-LABEL: v2i64z:
; CHECK: movi v[[R:[0-9]+]].2d, #0
; CHECK: mov  v[[R]].d[1], x{{[0-9]+}}
}

define <2 x float> @v2f32z(float %t, float %s) nounwind {
  %v = insertelement <2 x float> <float 0.0, float undef>, float %s, i32 1
  ret <2 x float> %v

; CHECK-LABEL: v2f32z:
; CHECK: movi d[[R:[0-9]+]], #0
; CHECK: mov  v[[R]].s[1], v{{[0-9]+}}.s[0]
}

define <4 x float> @v4f32z(float %t, float %s) nounwind {
  %v = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float undef>, float %s, i32 3
  ret <4 x float> %v

; CHECK-LABEL: v4f32z:
; CHECK: movi v[[R:[0-9]+]].2d, #0
; CHECK: mov  v[[R]].s[3], v{{[0-9]+}}.s[0]
}

define <2 x double> @v2f64z(double %t, double %s) nounwind {
  %v = insertelement <2 x double> <double 0.0, double undef>, double %s, i32 1
  ret <2 x double> %v

; CHECK-LABEL: v2f64z:
; CHECK: movi v[[R:[0-9]+]].2d, #0
; CHECK: mov  v[[R]].d[1], v{{[0-9]+}}.d[0]
}

; Check that building up a vector w/ only one non-ones lane initializes
; efficiently.

define <8 x i8> @v8i8m(i8 %t, i8 %s) nounwind {
  %v = insertelement <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 undef>, i8 %s, i32 7
  ret <8 x i8> %v

; CHECK-LABEL: v8i8m
; CHECK: movi d{{[0-9]+}}, #0xffffffffffffffff
; CHECK: mov  v[[R]].b[7], w{{[0-9]+}}
}

define <16 x i8> @v16i8m(i8 %t, i8 %s) nounwind {
  %v = insertelement <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 undef>, i8 %s, i32 15
  ret <16 x i8> %v

; CHECK-LABEL: v16i8m
; CHECK: movi v[[R:[0-9]+]].2d, #0xffffffffffffffff
; CHECK: mov  v[[R]].b[15], w{{[0-9]+}}
}

define <4 x i16> @v4i16m(i16 %t, i16 %s) nounwind {
  %v = insertelement <4 x i16> <i16 -1, i16 -1, i16 -1, i16 undef>, i16 %s, i32 3
  ret <4 x i16> %v

; CHECK-LABEL: v4i16m
; CHECK: movi d{{[0-9]+}}, #0xffffffffffffffff
; CHECK: mov  v[[R]].h[3], w{{[0-9]+}}
}

define <8 x i16> @v8i16m(i16 %t, i16 %s) nounwind {
  %v = insertelement <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 undef>, i16 %s, i32 7
  ret <8 x i16> %v

; CHECK-LABEL: v8i16m
; CHECK: movi v[[R:[0-9]+]].2d, #0xffffffffffffffff
; CHECK: mov  v[[R]].h[7], w{{[0-9]+}}
}

define <2 x i32> @v2i32m(i32 %t, i32 %s) nounwind {
  %v = insertelement <2 x i32> <i32 -1, i32 undef>, i32 %s, i32 1
  ret <2 x i32> %v

; CHECK-LABEL: v2i32m
; CHECK: movi d{{[0-9]+}}, #0xffffffffffffffff
; CHECK: mov  v[[R]].s[1], w{{[0-9]+}}
}

define <4 x i32> @v4i32m(i32 %t, i32 %s) nounwind {
  %v = insertelement <4 x i32> <i32 -1, i32 -1, i32 -1, i32 undef>, i32 %s, i32 3
  ret <4 x i32> %v

; CHECK-LABEL: v4i32m
; CHECK: movi v[[R:[0-9]+]].2d, #0xffffffffffffffff
; CHECK: mov  v[[R]].s[3], w{{[0-9]+}}
}

define <2 x i64> @v2i64m(i64 %t, i64 %s) nounwind {
  %v = insertelement <2 x i64> <i64 -1, i64 undef>, i64 %s, i32 1
  ret <2 x i64> %v

; CHECK-LABEL: v2i64m
; CHECK: movi v[[R:[0-9]+]].2d, #0xffffffffffffffff
; CHECK: mov  v[[R]].d[1], x{{[0-9]+}}
}

; Check that building up a vector w/ some constants initializes efficiently.

define void @v8i8st(<8 x i8>* %p, i8 %s) nounwind {
  %v = insertelement <8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 undef>, i8 %s, i32 7
  store <8 x i8> %v, <8 x i8>* %p, align 8
  ret void

; CHECK-LABEL: v8i8st:
; CHECK: movi v[[R:[0-9]+]].8b, #1
; CHECK: mov  v[[R]].b[7], w{{[0-9]+}}
; CHECK: str  d[[R]], [x{{[0-9]+}}]
}

define void @v16i8st(<16 x i8>* %p, i8 %s) nounwind {
  %v = insertelement <16 x i8> <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 undef>, i8 %s, i32 15
  store <16 x i8> %v, <16 x i8>* %p, align 16
  ret void

; CHECK-LABEL: v16i8st:
; CHECK: movi v[[R:[0-9]+]].16b, #128
; CHECK: mov  v[[R]].b[15], w{{[0-9]+}}
; CHECK: str  q[[R]], [x{{[0-9]+}}]
}

define void @v4i16st(<4 x i16>* %p, i16 %s) nounwind {
  %v = insertelement <4 x i16> <i16 21760, i16 21760, i16 21760, i16 undef>, i16 %s, i32 3
  store <4 x i16> %v, <4 x i16>* %p, align 8
  ret void

; CHECK-LABEL: v4i16st:
; CHECK: movi v[[R:[0-9]+]].4h, #85, lsl #8
; CHECK: mov  v[[R]].h[3], w{{[0-9]+}}
; CHECK: str  d[[R]], [x{{[0-9]+}}]
}

define void @v8i16st(<8 x i16>* %p, i16 %s) nounwind {
  %v = insertelement <8 x i16> <i16 -21761, i16 -21761, i16 -21761, i16 -21761, i16 -21761, i16 -21761, i16 -21761, i16 undef>, i16 %s, i32 7
  store <8 x i16> %v, <8 x i16>* %p, align 16
  ret void

; CHECK-LABEL: v8i16st:
; CHECK: mvni v[[R:[0-9]+]].8h, #85, lsl #8
; CHECK: mov  v[[R]].h[7], w{{[0-9]+}}
; CHECK: str  q[[R]], [x{{[0-9]+}}]
}

define void @v2i32st(<2 x i32>* %p, i32 %s) nounwind {
  %v = insertelement <2 x i32> <i32 983040, i32 undef>, i32 %s, i32 1
  store <2 x i32> %v, <2 x i32>* %p, align 8
  ret void

; CHECK-LABEL: v2i32st:
; CHECK: movi v[[R:[0-9]+]].2s, #15, lsl #16
; CHECK: mov  v[[R]].s[1], w{{[0-9]+}}
; CHECK: str  d[[R]], [x{{[0-9]+}}]
}

define void @v4i32st(<4 x i32>* %p, i32 %s) nounwind {
  %v = insertelement <4 x i32> <i32 16318463, i32 16318463, i32 16318463, i32 undef>, i32 %s, i32 3
  store <4 x i32> %v, <4 x i32>* %p, align 16
  ret void

; CHECK-LABEL: v4i32st:
; CHECK: movi v[[R:[0-9]+]].4s, #248, msl #16
; CHECK: mov  v[[R]].s[3], w{{[0-9]+}}
; CHECK: str  q[[R]], [x{{[0-9]+}}]
}

define void @v2i64st(<2 x i64>* %p, i64 %s) nounwind {
  %v = insertelement <2 x i64> <i64 13835058055282163712, i64 undef>, i64 %s, i32 1
  store <2 x i64> %v, <2 x i64>* %p, align 16
  ret void

; CHECK-LABEL: v2i64st:
; CHECK: fmov v[[R:[0-9]+]].2d, #-2.0
; CHECK: mov  v[[R]].d[1], x{{[0-9]+}}
; CHECK: str  q[[R]], [x{{[0-9]+}}]
}

define void @v2f32st(<2 x float>* %p, float %s) nounwind {
  %v = insertelement <2 x float> <float 2.0, float undef>, float %s, i32 1
  store <2 x float> %v, <2 x float>* %p, align 8
  ret void

; CHECK-LABEL: v2f32st:
; CHECK: movi v[[R:[0-9]+]].2s, #64, lsl #24
; CHECK: mov  v[[R]].s[1], v{{[0-9]+}}.s[0]
; CHECK: str  d[[R]], [x{{[0-9]+}}]
}

define void @v4f32st(<4 x float>* %p, float %s) nounwind {
  %v = insertelement <4 x float> <float -2.0, float -2.0, float -2.0, float undef>, float %s, i32 3
  store <4 x float> %v, <4 x float>* %p, align 16
  ret void

; CHECK-LABEL: v4f32st:
; CHECK: movi v[[R:[0-9]+]].4s, #192, lsl #24
; CHECK: mov  v[[R]].s[3], v{{[0-9]+}}.s[0]
; CHECK: str  q[[R]], [x{{[0-9]+}}]
}

define void @v2f64st(<2 x double>* %p, double %s) nounwind {
  %v = insertelement <2 x double> <double 2.0, double undef>, double %s, i32 1
  store <2 x double> %v, <2 x double>* %p, align 16
  ret void

; CHECK-LABEL: v2f64st:
; CHECK: fmov v[[R:[0-9]+]].2d, #2.0
; CHECK: mov  v[[R]].d[1], v{{[0-9]+}}.d[0]
; CHECK: str  q[[R]], [x{{[0-9]+}}]
}

; In this test the illegal type has a preferred alignment greater than the
; stack alignment, that gets reduced to the alignment of a broken down
; legal type.
define <32 x i8> @test_lanex_32xi8(<32 x i8> %a, i32 %x) {
; CHECK-LABEL: test_lanex_32xi8
; CHECK:       stp q0, q1, [sp, #-32]!
; CHECK:       ldp q0, q1, [sp], #32
  %b = insertelement <32 x i8> %a, i8 30, i32 %x
  ret <32 x i8> %b
}

