; RUN: llc < %s -enable-unsafe-fp-math -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s 


; rdar://13126763
; Expression "x + x*x" was mistakenly transformed into "x * 3.0f".

define float @test1(float %x) {
  %t1 = fmul fast float %x, %x
  %t2 = fadd fast float %t1, %x
  ret float %t2
; CHECK: test1
; CHECK: vaddss
}

; (x + x) + x => x * 3.0
define float @test2(float %x) {
  %t1 = fadd fast float %x, %x
  %t2 = fadd fast float %t1, %x
  ret float %t2
; CHECK: .long  1077936128
; CHECK: test2
; CHECK: vmulss LCPI1_0(%rip), %xmm0, %xmm0
}

; x + (x + x) => x * 3.0
define float @test3(float %x) {
  %t1 = fadd fast float %x, %x
  %t2 = fadd fast float %t1, %x
  ret float %t2
; CHECK: .long  1077936128
; CHECK: test3
; CHECK: vmulss LCPI2_0(%rip), %xmm0, %xmm0
}

; (y + x) + x != x * 3.0
define float @test4(float %x, float %y) {
  %t1 = fadd fast float %x, %y
  %t2 = fadd fast float %t1, %x
  ret float %t2
; CHECK: test4
; CHECK: vaddss
}

; rdar://13445387
; "x + x + x => 3.0 * x" should be disabled after legalization because 
; Instruction-Selection doesn't know how to handle "3.0"
; 
define float @test5() {
  %mul.i.i151 = fmul <4 x float> zeroinitializer, zeroinitializer
  %vecext.i8.i152 = extractelement <4 x float> %mul.i.i151, i32 1
  %vecext1.i9.i153 = extractelement <4 x float> %mul.i.i151, i32 0
  %add.i10.i154 = fadd float %vecext1.i9.i153, %vecext.i8.i152
  %vecext.i7.i155 = extractelement <4 x float> %mul.i.i151, i32 2
  %add.i.i156 = fadd float %vecext.i7.i155, %add.i10.i154
  ret float %add.i.i156
}
