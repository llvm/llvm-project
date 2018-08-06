; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_00
; CHECK: [[P00:p[0-9]+]] = vcmpb.eq(r1:0,r3:2)
; CHECK: [[P01:p[0-9]+]] = not([[P00]])
; CHECK: r1:0 = mask([[P01]])
; CHECK: jumpr r31
define <8 x i8> @test_00(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp ne <8 x i8> %a0, %a1
  %v1 = sext <8 x i1> %v0 to <8 x i8>
  ret <8 x i8> %v1
}

; CHECK-LABEL: test_01
; CHECK: [[P00:p[0-9]+]] = vcmpb.gt(r1:0,r3:2)
; CHECK: [[P01:p[0-9]+]] = not([[P00]])
; CHECK: r1:0 = mask([[P01]])
; CHECK: jumpr r31
define <8 x i8> @test_01(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp sle <8 x i8> %a0, %a1
  %v1 = sext <8 x i1> %v0 to <8 x i8>
  ret <8 x i8> %v1
}

; CHECK-LABEL: test_02
; CHECK: [[P00:p[0-9]+]] = vcmpb.gtu(r1:0,r3:2)
; CHECK: [[P01:p[0-9]+]] = not([[P00]])
; CHECK: r1:0 = mask([[P01]])
; CHECK: jumpr r31
define <8 x i8> @test_02(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp ule <8 x i8> %a0, %a1
  %v1 = sext <8 x i1> %v0 to <8 x i8>
  ret <8 x i8> %v1
}

; CHECK-LABEL: test_10
; CHECK: [[P00:p[0-9]+]] = vcmph.eq(r1:0,r3:2)
; CHECK: [[P01:p[0-9]+]] = not([[P00]])
; CHECK: r1:0 = mask([[P01]])
; CHECK: jumpr r31
define <4 x i16> @test_10(<4 x i16> %a0, <4 x i16> %a1) #0 {
  %v0 = icmp ne <4 x i16> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i16>
  ret <4 x i16> %v1
}

; CHECK-LABEL: test_11
; CHECK: [[P00:p[0-9]+]] = vcmph.gt(r1:0,r3:2)
; CHECK: [[P01:p[0-9]+]] = not([[P00]])
; CHECK: r1:0 = mask([[P01]])
; CHECK: jumpr r31
define <4 x i16> @test_11(<4 x i16> %a0, <4 x i16> %a1) #0 {
  %v0 = icmp sle <4 x i16> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i16>
  ret <4 x i16> %v1
}

; CHECK-LABEL: test_12
; CHECK: [[P00:p[0-9]+]] = vcmph.gtu(r1:0,r3:2)
; CHECK: [[P01:p[0-9]+]] = not([[P00]])
; CHECK: r1:0 = mask([[P01]])
; CHECK: jumpr r31
define <4 x i16> @test_12(<4 x i16> %a0, <4 x i16> %a1) #0 {
  %v0 = icmp ule <4 x i16> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i16>
  ret <4 x i16> %v1
}

; CHECK-LABEL: test_20
; CHECK: [[P00:p[0-9]+]] = vcmpw.eq(r1:0,r3:2)
; CHECK: [[P01:p[0-9]+]] = not([[P00]])
; CHECK: r1:0 = mask([[P01]])
; CHECK: jumpr r31
define <2 x i32> @test_20(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp ne <2 x i32> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i32>
  ret <2 x i32> %v1
}

; CHECK-LABEL: test_21
; CHECK: [[P00:p[0-9]+]] = vcmpw.gt(r1:0,r3:2)
; CHECK: [[P01:p[0-9]+]] = not([[P00]])
; CHECK: r1:0 = mask([[P01]])
; CHECK: jumpr r31
define <2 x i32> @test_21(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp sle <2 x i32> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i32>
  ret <2 x i32> %v1
}

; CHECK-LABEL: test_22
; CHECK: [[P00:p[0-9]+]] = vcmpw.gtu(r1:0,r3:2)
; CHECK: [[P01:p[0-9]+]] = not([[P00]])
; CHECK: r1:0 = mask([[P01]])
; CHECK: jumpr r31
define <2 x i32> @test_22(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp ule <2 x i32> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i32>
  ret <2 x i32> %v1
}

attributes #0 = { nounwind readnone }
