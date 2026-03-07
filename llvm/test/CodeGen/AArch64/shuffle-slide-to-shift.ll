; RUN: llc -mtriple=aarch64 < %s | FileCheck %s


; repro  of gh issue

; Left slide: {v[0]..v[7]} => {v[1]..v[7], 0}
define <8 x i8> @slide_left_1(<8 x i8> %v) {
; CHECK-LABEL: slide_left_1:
; CHECK:       ushr d0, d0, #8
  %r = shufflevector <8 x i8> %v, <8 x i8> zeroinitializer,
       <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  ret <8 x i8> %r
}

; Right slide: {v[0]..v[7]} => {0, v[0]..v[6]}
define <8 x i8> @slide_right_1(<8 x i8> %v) {
; CHECK-LABEL: slide_right_1:
; CHECK:       shl d0, d0, #8
  %r = shufflevector <8 x i8> zeroinitializer, <8 x i8> %v,
       <8 x i32> <i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  ret <8 x i8> %r
}

; 64 bit test (with poison)

; Left slide with poison (from issue's Alive2 proof)
define <8 x i8> @slide_left_poison(<8 x i8> %v) {
; CHECK-LABEL: slide_left_poison:
; CHECK:       ushr d0, d0, #8
  %r = shufflevector <8 x i8> %v, <8 x i8> <i8 0, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>,
       <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  ret <8 x i8> %r
}

; Right slide with poison (from issue's Alive2 proof)
define <8 x i8> @slide_right_poison(<8 x i8> %v) {
; CHECK-LABEL: slide_right_poison:
; CHECK:       shl d0, d0, #8
  %r = shufflevector <8 x i8> <i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 0>, <8 x i8> %v,
       <8 x i32> <i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  ret <8 x i8> %r
}

; should also optimize 32-bit vectors

; Left slide 32-bit: <4 x i8>
define <4 x i8> @slide_left_v4i8(<4 x i8> %v) {
; CHECK-LABEL: slide_left_v4i8:
; CHECK:       ushr d0, d0, #16
  %r = shufflevector <4 x i8> %v, <4 x i8> zeroinitializer,
       <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i8> %r
}

; Right slide 32-bit: <4 x i8>
define <4 x i8> @slide_right_v4i8(<4 x i8> %v) {
; CHECK-LABEL: slide_right_v4i8:
; CHECK:       shl d0, d0, #16
  %r = shufflevector <4 x i8> zeroinitializer, <4 x i8> %v,
       <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x i8> %r
}

; should NOT optimize 128-bit vectors 

; Left slide 128-bit: <16 x i8> - must use ext
define <16 x i8> @slide_left_v16i8(<16 x i8> %v) {
; CHECK-LABEL: slide_left_v16i8:
; CHECK:       movi
; CHECK:       ext
  %r = shufflevector <16 x i8> %v, <16 x i8> zeroinitializer,
       <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8,
                   i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>
  ret <16 x i8> %r
}

; Right slide 128-bit: <16 x i8> - must use ext
define <16 x i8> @slide_right_v16i8(<16 x i8> %v) {
; CHECK-LABEL: slide_right_v16i8:
; CHECK:       movi
; CHECK:       ext
  %r = shufflevector <16 x i8> zeroinitializer, <16 x i8> %v,
       <16 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22,
                   i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
  ret <16 x i8> %r
}

; Slide by max (N-1 elements)

; Left slide by 7 (max for v8i8)
define <8 x i8> @slide_left_max(<8 x i8> %v) {
; CHECK-LABEL: slide_left_max:
; CHECK:       ushr d0, d0, #56
  %r = shufflevector <8 x i8> %v, <8 x i8> zeroinitializer,
       <8 x i32> <i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  ret <8 x i8> %r
}

; Right slide by 7 (max for v8i8): [v0..v7] => [0,0,0,0,0,0,0,v0]
define <8 x i8> @slide_right_max(<8 x i8> %v) {
; CHECK-LABEL: slide_right_max:
; CHECK:       shl d0, d0, #56
  %r = shufflevector <8 x i8> zeroinitializer, <8 x i8> %v,
       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  ret <8 x i8> %r
}