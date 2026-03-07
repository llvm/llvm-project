; RUN: llc -mtriple=aarch64 < %s | FileCheck %s

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

; === Tests (with poison) ===

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