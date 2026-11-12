; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_00
; CHECK: r0 = and(r0,r1)
; CHECK-NEXT: jumpr r31
define <4 x i8> @test_00(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = and <4 x i8> %a0, %a1
  ret <4 x i8> %v0
}

; CHECK-LABEL: test_01
; CHECK: r0 = or(r0,r1)
; CHECK-NEXT: jumpr r31
define <4 x i8> @test_01(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = or <4 x i8> %a0, %a1
  ret <4 x i8> %v0
}

; CHECK-LABEL: test_02
; CHECK: r0 = xor(r0,r1)
; CHECK-NEXT: jumpr r31
define <4 x i8> @test_02(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = xor <4 x i8> %a0, %a1
  ret <4 x i8> %v0
}

attributes #0 = { nounwind readnone }
