; RUN: not llc -mtriple aarch64-none-linux-gnu -mattr=+neon -o %t.s -filetype=asm %s 2>&1 | FileCheck %s

; The 'y' constraint only applies to SVE vector registers (Z0-Z7)
; The test below ensures that we get an appropriate error should the
; constraint be used with a Neon register.

; Function Attrs: nounwind readnone
; CHECK: error: couldn't allocate input reg for constraint 'y'
define <4 x i32> @test_neon(<4 x i32> %in1, <4 x i32> %in2) {
  %1 = tail call <4 x i32> asm "add $0.4s, $1.4s, $2.4s", "=w,w,y"(<4 x i32> %in1, <4 x i32> %in2)
  ret <4 x i32> %1
}
