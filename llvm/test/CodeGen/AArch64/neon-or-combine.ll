; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

; Check that the DAGCombiner does not crash with an assertion failure
; when performing a target specific combine to simplify a 'or' dag node
; according to the following rule:
;   (or (and B, A), (and C, ~A)) => (VBSL A, B, C)
; The assertion failure was caused by an invalid comparison between APInt
; values with different 'BitWidth'.

define <8 x i8> @test1(<8 x i8> %a, <8 x i8> %b)  {
  %tmp1 = and <8 x i8> %a, < i8 -1, i8 -1, i8 0, i8 0, i8 -1, i8 -1, i8 0, i8 0 >
  %tmp2 = and <8 x i8> %b, < i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0 >
  %tmp3 = or <8 x i8> %tmp1, %tmp2
  ret <8 x i8> %tmp3
}

; CHECK-LABEL: test1
; CHECK: ret

define <16 x i8> @test2(<16 x i8> %a, <16 x i8> %b) {
  %tmp1 = and <16 x i8> %a, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
  %tmp2 = and <16 x i8> %b, < i8 -1, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0 >
  %tmp3 = or <16 x i8> %tmp1, %tmp2
  ret <16 x i8> %tmp3
}

; CHECK-LABEL: test2
; CHECK: ret

