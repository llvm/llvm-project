; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+sse4.1 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 | FileCheck %s

; When commuting the operands of a SSE blend, make sure that the resulting blend
; mask can be encoded as a imm8.
; Before, when commuting the operands to the shuffle in function @test, the backend
; produced the following assembly:
;   pblendw $4294967103, %xmm1, %xmm0

define <4 x i32> @test(<4 x i32> %a, <4 x i32> %b) {
; CHECK: pblendw $63, %xmm1, %xmm0
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 3>
  ; add forces execution domain
  %sum = add <4 x i32> %shuffle, %shuffle
  ret <4 x i32> %sum
}
