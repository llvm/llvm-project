; RUN: llc < %s -march=x86-64 -mattr=+avx2 | FileCheck %s

; CHECK: phaddw1:
; CHECK: vphaddw
define <16 x i16> @phaddw1(<16 x i16> %x, <16 x i16> %y) {
  %a = shufflevector <16 x i16> %x, <16 x i16> %y, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 16, i32 18, i32 20, i32 22, i32 8, i32 10, i32 12, i32 14, i32 24, i32 26, i32 28, i32 30>
  %b = shufflevector <16 x i16> %x, <16 x i16> %y, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 17, i32 19, i32 21, i32 23, i32 9, i32 11, i32 13, i32 15, i32 25, i32 27, i32 29, i32 31>
  %r = add <16 x i16> %a, %b
  ret <16 x i16> %r
}

; CHECK: phaddw2:
; CHECK: vphaddw
define <16 x i16> @phaddw2(<16 x i16> %x, <16 x i16> %y) {
  %a = shufflevector <16 x i16> %x, <16 x i16> %y, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 17, i32 19, i32 21, i32 23, i32 9, i32 11, i32 13, i32 15, i32 25, i32 27, i32 29, i32 31>
  %b = shufflevector <16 x i16> %y, <16 x i16> %x, <16 x i32> <i32 16, i32 18, i32 20, i32 22, i32 0, i32 2, i32 4, i32 6, i32 24, i32 26, i32 28, i32 30, i32 8, i32 10, i32 12, i32 14>
  %r = add <16 x i16> %a, %b
  ret <16 x i16> %r
}

; CHECK: phaddd1:
; CHECK: vphaddd
define <8 x i32> @phaddd1(<8 x i32> %x, <8 x i32> %y) {
  %a = shufflevector <8 x i32> %x, <8 x i32> %y, <8 x i32> <i32 0, i32 2, i32 8, i32 10, i32 4, i32 6, i32 12, i32 14>
  %b = shufflevector <8 x i32> %x, <8 x i32> %y, <8 x i32> <i32 1, i32 3, i32 9, i32 11, i32 5, i32 7, i32 13, i32 15>
  %r = add <8 x i32> %a, %b
  ret <8 x i32> %r
}

; CHECK: phaddd2:
; CHECK: vphaddd
define <8 x i32> @phaddd2(<8 x i32> %x, <8 x i32> %y) {
  %a = shufflevector <8 x i32> %x, <8 x i32> %y, <8 x i32> <i32 1, i32 2, i32 9, i32 10, i32 5, i32 6, i32 13, i32 14>
  %b = shufflevector <8 x i32> %y, <8 x i32> %x, <8 x i32> <i32 8, i32 11, i32 0, i32 3, i32 12, i32 15, i32 4, i32 7>
  %r = add <8 x i32> %a, %b
  ret <8 x i32> %r
}

; CHECK: phaddd3:
; CHECK: vphaddd
define <8 x i32> @phaddd3(<8 x i32> %x) {
  %a = shufflevector <8 x i32> %x, <8 x i32> undef, <8 x i32> <i32 undef, i32 2, i32 8, i32 10, i32 4, i32 6, i32 undef, i32 14>
  %b = shufflevector <8 x i32> %x, <8 x i32> undef, <8 x i32> <i32 1, i32 3, i32 9, i32 undef, i32 5, i32 7, i32 13, i32 15>
  %r = add <8 x i32> %a, %b
  ret <8 x i32> %r
}

; CHECK: phsubw1:
; CHECK: vphsubw
define <16 x i16> @phsubw1(<16 x i16> %x, <16 x i16> %y) {
  %a = shufflevector <16 x i16> %x, <16 x i16> %y, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 16, i32 18, i32 20, i32 22, i32 8, i32 10, i32 12, i32 14, i32 24, i32 26, i32 28, i32 30>
  %b = shufflevector <16 x i16> %x, <16 x i16> %y, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 17, i32 19, i32 21, i32 23, i32 9, i32 11, i32 13, i32 15, i32 25, i32 27, i32 29, i32 31>
  %r = sub <16 x i16> %a, %b
  ret <16 x i16> %r
}

; CHECK: phsubd1:
; CHECK: vphsubd
define <8 x i32> @phsubd1(<8 x i32> %x, <8 x i32> %y) {
  %a = shufflevector <8 x i32> %x, <8 x i32> %y, <8 x i32> <i32 0, i32 2, i32 8, i32 10, i32 4, i32 6, i32 12, i32 14>
  %b = shufflevector <8 x i32> %x, <8 x i32> %y, <8 x i32> <i32 1, i32 3, i32 9, i32 11, i32 5, i32 7, i32 13, i32 15>
  %r = sub <8 x i32> %a, %b
  ret <8 x i32> %r
}

; CHECK: phsubd2:
; CHECK: vphsubd
define <8 x i32> @phsubd2(<8 x i32> %x, <8 x i32> %y) {
  %a = shufflevector <8 x i32> %x, <8 x i32> %y, <8 x i32> <i32 0, i32 undef, i32 8, i32 undef, i32 4, i32 6, i32 12, i32 14>
  %b = shufflevector <8 x i32> %x, <8 x i32> %y, <8 x i32> <i32 1, i32 undef, i32 9, i32 11, i32 5, i32 7, i32 undef, i32 15>
  %r = sub <8 x i32> %a, %b
  ret <8 x i32> %r
}
