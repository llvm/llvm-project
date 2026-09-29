; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that the value produced is 0x0706050403020100.
; CHECK: r1:0 = CONST64(#506097522914230528)

define <8 x i8> @fred() {
  ret <8 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>
}
