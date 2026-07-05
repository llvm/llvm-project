; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; min

; CHECK-LABEL: test_00:
; CHECK: r1:0 = vminb(r1:0,r3:2)
define <8 x i8> @test_00(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp slt <8 x i8> %a0, %a1
  %v1 = select <8 x i1> %v0, <8 x i8> %a0, <8 x i8> %a1
  ret <8 x i8> %v1
}

; CHECK-LABEL: test_01:
; CHECK: r1:0 = vminb(r1:0,r3:2)
define <8 x i8> @test_01(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp sle <8 x i8> %a0, %a1
  %v1 = select <8 x i1> %v0, <8 x i8> %a0, <8 x i8> %a1
  ret <8 x i8> %v1
}

; CHECK-LABEL: test_02:
; CHECK: r1:0 = vminh(r1:0,r3:2)
define <4 x i16> @test_02(<4 x i16> %a0, <4 x i16> %a1) #0 {
  %v0 = icmp slt <4 x i16> %a0, %a1
  %v1 = select <4 x i1> %v0, <4 x i16> %a0, <4 x i16> %a1
  ret <4 x i16> %v1
}

; CHECK-LABEL: test_03:
; CHECK: r1:0 = vminh(r1:0,r3:2)
define <4 x i16> @test_03(<4 x i16> %a0, <4 x i16> %a1) #0 {
  %v0 = icmp sle <4 x i16> %a0, %a1
  %v1 = select <4 x i1> %v0, <4 x i16> %a0, <4 x i16> %a1
  ret <4 x i16> %v1
}

; CHECK-LABEL: test_04:
; CHECK: r1:0 = vminw(r1:0,r3:2)
define <2 x i32> @test_04(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp slt <2 x i32> %a0, %a1
  %v1 = select <2 x i1> %v0, <2 x i32> %a0, <2 x i32> %a1
  ret <2 x i32> %v1
}

; CHECK-LABEL: test_05:
; CHECK: r1:0 = vminw(r1:0,r3:2)
define <2 x i32> @test_05(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp sle <2 x i32> %a0, %a1
  %v1 = select <2 x i1> %v0, <2 x i32> %a0, <2 x i32> %a1
  ret <2 x i32> %v1
}

; minu

; CHECK-LABEL: test_06:
; CHECK: r1:0 = vminub(r1:0,r3:2)
define <8 x i8> @test_06(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp ult <8 x i8> %a0, %a1
  %v1 = select <8 x i1> %v0, <8 x i8> %a0, <8 x i8> %a1
  ret <8 x i8> %v1
}

; CHECK-LABEL: test_07:
; CHECK: r1:0 = vminub(r1:0,r3:2)
define <8 x i8> @test_07(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp ule <8 x i8> %a0, %a1
  %v1 = select <8 x i1> %v0, <8 x i8> %a0, <8 x i8> %a1
  ret <8 x i8> %v1
}

; CHECK-LABEL: test_08:
; CHECK: r1:0 = vminuh(r1:0,r3:2)
define <4 x i16> @test_08(<4 x i16> %a0, <4 x i16> %a1) #0 {
  %v0 = icmp ult <4 x i16> %a0, %a1
  %v1 = select <4 x i1> %v0, <4 x i16> %a0, <4 x i16> %a1
  ret <4 x i16> %v1
}

; CHECK-LABEL: test_09:
; CHECK: r1:0 = vminuh(r1:0,r3:2)
define <4 x i16> @test_09(<4 x i16> %a0, <4 x i16> %a1) #0 {
  %v0 = icmp ule <4 x i16> %a0, %a1
  %v1 = select <4 x i1> %v0, <4 x i16> %a0, <4 x i16> %a1
  ret <4 x i16> %v1
}

; CHECK-LABEL: test_0a:
; CHECK: r1:0 = vminuw(r1:0,r3:2)
define <2 x i32> @test_0a(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp ult <2 x i32> %a0, %a1
  %v1 = select <2 x i1> %v0, <2 x i32> %a0, <2 x i32> %a1
  ret <2 x i32> %v1
}

; CHECK-LABEL: test_0b:
; CHECK: r1:0 = vminuw(r1:0,r3:2)
define <2 x i32> @test_0b(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp ule <2 x i32> %a0, %a1
  %v1 = select <2 x i1> %v0, <2 x i32> %a0, <2 x i32> %a1
  ret <2 x i32> %v1
}

; max

; CHECK-LABEL: test_0c:
; CHECK: r1:0 = vmaxb(r1:0,r3:2)
define <8 x i8> @test_0c(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp sgt <8 x i8> %a0, %a1
  %v1 = select <8 x i1> %v0, <8 x i8> %a0, <8 x i8> %a1
  ret <8 x i8> %v1
}

; CHECK-LABEL: test_0d:
; CHECK: r1:0 = vmaxb(r1:0,r3:2)
define <8 x i8> @test_0d(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp sge <8 x i8> %a0, %a1
  %v1 = select <8 x i1> %v0, <8 x i8> %a0, <8 x i8> %a1
  ret <8 x i8> %v1
}

; CHECK-LABEL: test_0e:
; CHECK: r1:0 = vmaxh(r1:0,r3:2)
define <4 x i16> @test_0e(<4 x i16> %a0, <4 x i16> %a1) #0 {
  %v0 = icmp sgt <4 x i16> %a0, %a1
  %v1 = select <4 x i1> %v0, <4 x i16> %a0, <4 x i16> %a1
  ret <4 x i16> %v1
}

; CHECK-LABEL: test_0f:
; CHECK: r1:0 = vmaxh(r1:0,r3:2)
define <4 x i16> @test_0f(<4 x i16> %a0, <4 x i16> %a1) #0 {
  %v0 = icmp sge <4 x i16> %a0, %a1
  %v1 = select <4 x i1> %v0, <4 x i16> %a0, <4 x i16> %a1
  ret <4 x i16> %v1
}

; CHECK-LABEL: test_10:
; CHECK: r1:0 = vmaxw(r1:0,r3:2)
define <2 x i32> @test_10(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp sgt <2 x i32> %a0, %a1
  %v1 = select <2 x i1> %v0, <2 x i32> %a0, <2 x i32> %a1
  ret <2 x i32> %v1
}

; CHECK-LABEL: test_11:
; CHECK: r1:0 = vmaxw(r1:0,r3:2)
define <2 x i32> @test_11(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp sge <2 x i32> %a0, %a1
  %v1 = select <2 x i1> %v0, <2 x i32> %a0, <2 x i32> %a1
  ret <2 x i32> %v1
}

; maxu

; CHECK-LABEL: test_12:
; CHECK: r1:0 = vmaxub(r1:0,r3:2)
define <8 x i8> @test_12(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp ugt <8 x i8> %a0, %a1
  %v1 = select <8 x i1> %v0, <8 x i8> %a0, <8 x i8> %a1
  ret <8 x i8> %v1
}

; CHECK-LABEL: test_13:
; CHECK: r1:0 = vmaxub(r1:0,r3:2)
define <8 x i8> @test_13(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp uge <8 x i8> %a0, %a1
  %v1 = select <8 x i1> %v0, <8 x i8> %a0, <8 x i8> %a1
  ret <8 x i8> %v1
}

; CHECK-LABEL: test_14:
; CHECK: r1:0 = vmaxuh(r1:0,r3:2)
define <4 x i16> @test_14(<4 x i16> %a0, <4 x i16> %a1) #0 {
  %v0 = icmp ugt <4 x i16> %a0, %a1
  %v1 = select <4 x i1> %v0, <4 x i16> %a0, <4 x i16> %a1
  ret <4 x i16> %v1
}

; CHECK-LABEL: test_15:
; CHECK: r1:0 = vmaxuh(r1:0,r3:2)
define <4 x i16> @test_15(<4 x i16> %a0, <4 x i16> %a1) #0 {
  %v0 = icmp uge <4 x i16> %a0, %a1
  %v1 = select <4 x i1> %v0, <4 x i16> %a0, <4 x i16> %a1
  ret <4 x i16> %v1
}

; CHECK-LABEL: test_16:
; CHECK: r1:0 = vmaxuw(r1:0,r3:2)
define <2 x i32> @test_16(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp ugt <2 x i32> %a0, %a1
  %v1 = select <2 x i1> %v0, <2 x i32> %a0, <2 x i32> %a1
  ret <2 x i32> %v1
}

; CHECK-LABEL: test_17:
; CHECK: r1:0 = vmaxuw(r1:0,r3:2)
define <2 x i32> @test_17(<2 x i32> %a0, <2 x i32> %a1) #0 {
  %v0 = icmp uge <2 x i32> %a0, %a1
  %v1 = select <2 x i1> %v0, <2 x i32> %a0, <2 x i32> %a1
  ret <2 x i32> %v1
}

