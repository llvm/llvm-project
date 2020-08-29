; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <4 x i16> @shuffle1(<4 x i16> %v) {
; CHECK-LABEL: shuffle1:
; CHECK:         dup v0.2s, v0.s[0]
; CHECK-NEXT:    ret
entry:
  %res = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 0, i32 undef, i32 0, i32 1>
  ret <4 x i16> %res
}

define <4 x i16> @shuffle2(<4 x i16> %v) {
; CHECK-LABEL: shuffle2:
; CHECK:         dup v0.2s, v0.s[1]
; CHECK-NEXT:    ret
entry:
  %res = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 3>
  ret <4 x i16> %res
}

define <8 x i16> @shuffle3(<8 x i16> %v) {
; CHECK-LABEL: shuffle3:
; CHECK:         dup v0.2d, v0.d[0]
; CHECK-NEXT:    ret
entry:
  %res = shufflevector <8 x i16> %v, <8 x i16> undef, <8 x i32> <i32 undef, i32 undef, i32 2, i32 3,
                                                                 i32 undef, i32 1, i32 undef, i32 3>
  ret <8 x i16> %res
}

define <4 x i32> @shuffle4(<4 x i32> %v) {
; CHECK-LABEL: shuffle4:
; CHECK:         dup v0.2d, v0.d[0]
; CHECK-NEXT:    ret
entry:
  %res = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  ret <4 x i32> %res
}

define <16 x i8> @shuffle5(<16 x i8> %v) {
; CHECK-LABEL: shuffle5:
; CHECK:         dup v0.4s, v0.s[2]
; CHECK-NEXT:    ret
entry:
  %res = shufflevector <16 x i8> %v, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11,
                                                                  i32 8, i32 9, i32 10, i32 11,
                                                                  i32 8, i32 9, i32 10, i32 11,
                                                                  i32 8, i32 9, i32 10, i32 11>
  ret <16 x i8> %res
}

define <16 x i8> @shuffle6(<16 x i8> %v) {
; CHECK-LABEL: shuffle6:
; CHECK:         dup v0.2d, v0.d[1]
; CHECK-NEXT:    ret
entry:
  %res = shufflevector <16 x i8> %v, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11,
                                                                  i32 12, i32 13, i32 14, i32 15,
                                                                  i32 8, i32 9, i32 10, i32 11,
                                                                  i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %res
}

define <8 x i8> @shuffle7(<8 x i8> %v) {
; CHECK-LABEL: shuffle7:
; CHECK:         dup v0.2s, v0.s[1]
; CHECK-NEXT:    ret
entry:
  %res = shufflevector <8 x i8> %v, <8 x i8> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 undef,
                                                               i32 undef, i32 5, i32 6, i32 undef>
  ret <8 x i8> %res
}

define <8 x i8> @shuffle8(<8 x i8> %v) {
; CHECK-LABEL: shuffle8:
; CHECK:         dup v0.4h, v0.h[3]
; CHECK-NEXT:    ret
entry:
  %res = shufflevector <8 x i8> %v, <8 x i8> undef, <8 x i32> <i32 6, i32 7, i32 6, i32 undef,
                                                               i32 undef, i32 7, i32 6, i32 undef>
  ret <8 x i8> %res
}

; No blocks
define <8 x i8> @shuffle_not1(<16 x i8> %v) {
; CHECK-LABEL: shuffle_not1:
; CHECK:         ext v0.16b, v0.16b, v0.16b, #2
  %res = shufflevector <16 x i8> %v, <16 x i8> undef, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  ret <8 x i8> %res
}

; Block is not a proper lane
define <4 x i32> @shuffle_not2(<4 x i32> %v) {
; CHECK-LABEL: shuffle_not2:
; CHECK-NOT:     dup
; CHECK:         ext
; CHECK:         ret
entry:
  %res = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 1, i32 2, i32 1, i32 2>
  ret <4 x i32> %res
}

; Block size is equal to vector size
define <4 x i16> @shuffle_not3(<4 x i16> %v) {
; CHECK-LABEL: shuffle_not3:
; CHECK-NOT:     dup
; CHECK:         ret
entry:
  %res = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i16> %res
}

; Blocks mismatch
define <8 x i8> @shuffle_not4(<8 x i8> %v) {
; CHECK-LABEL: shuffle_not4:
; CHECK-NOT:     dup
; CHECK:         ret
entry:
  %res = shufflevector <8 x i8> %v, <8 x i8> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 undef,
                                                               i32 undef, i32 5, i32 5, i32 undef>
  ret <8 x i8> %res
}
