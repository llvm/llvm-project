; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_00
; CHECK: [[L00:r[0-9:]+]] = vsxtbh(r0)
; CHECK: [[R00:r[0-9:]+]] = vsxtbh(r1)
; CHECK: [[P00:p[0-3]+]] = vcmph.eq([[L00]],[[R00]])
; CHECK-NOT: not([[P00]])
define <4 x i8> @test_00(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = icmp eq <4 x i8> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i8>
  ret <4 x i8> %v1
}

; CHECK-LABEL: test_01
; CHECK: [[L01:r[0-9:]+]] = vsxtbh(r0)
; CHECK: [[R01:r[0-9:]+]] = vsxtbh(r1)
; CHECK: [[P01:p[0-3]+]] = vcmph.eq([[L01]],[[R01]])
; CHECK: not([[P01]])
define <4 x i8> @test_01(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = icmp ne <4 x i8> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i8>
  ret <4 x i8> %v1
}

; CHECK-LABEL: test_02
; CHECK: [[L02:r[0-9:]+]] = vsxtbh(r0)
; CHECK: [[R02:r[0-9:]+]] = vsxtbh(r1)
; CHECK: [[P02:p[0-3]+]] = vcmph.gt([[R02]],[[L02]])
; CHECK-NOT: not([[P02]])
define <4 x i8> @test_02(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = icmp slt <4 x i8> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i8>
  ret <4 x i8> %v1
}

; CHECK-LABEL: test_03
; CHECK: [[L03:r[0-9:]+]] = vsxtbh(r0)
; CHECK: [[R03:r[0-9:]+]] = vsxtbh(r1)
; CHECK: [[P03:p[0-3]+]] = vcmph.gt([[L03]],[[R03]])
; CHECK: not([[P03]])
define <4 x i8> @test_03(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = icmp sle <4 x i8> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i8>
  ret <4 x i8> %v1
}

; CHECK-LABEL: test_04
; CHECK: [[L04:r[0-9:]+]] = vsxtbh(r0)
; CHECK: [[R04:r[0-9:]+]] = vsxtbh(r1)
; CHECK: [[P04:p[0-3]+]] = vcmph.gt([[L04]],[[R04]])
; CHECK-NOT: not([[P04]])
define <4 x i8> @test_04(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = icmp sgt <4 x i8> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i8>
  ret <4 x i8> %v1
}

; CHECK-LABEL: test_05
; CHECK: [[L05:r[0-9:]+]] = vsxtbh(r0)
; CHECK: [[R05:r[0-9:]+]] = vsxtbh(r1)
; CHECK: [[P05:p[0-3]+]] = vcmph.gt([[R05]],[[L05]])
; CHECK: not([[P05]])
define <4 x i8> @test_05(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = icmp sge <4 x i8> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i8>
  ret <4 x i8> %v1
}

; CHECK-LABEL: test_06
; CHECK: [[L06:r[0-9:]+]] = vsxtbh(r0)
; CHECK: [[R06:r[0-9:]+]] = vsxtbh(r1)
; CHECK: [[P06:p[0-3]+]] = vcmph.gtu([[R06]],[[L06]])
; CHECK-NOT: not([[P06]])
define <4 x i8> @test_06(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = icmp ult <4 x i8> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i8>
  ret <4 x i8> %v1
}

; CHECK-LABEL: test_07
; CHECK: [[L07:r[0-9:]+]] = vsxtbh(r0)
; CHECK: [[R07:r[0-9:]+]] = vsxtbh(r1)
; CHECK: [[P07:p[0-3]+]] = vcmph.gtu([[L07]],[[R07]])
; CHECK: not([[P07]])
define <4 x i8> @test_07(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = icmp ule <4 x i8> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i8>
  ret <4 x i8> %v1
}

; CHECK-LABEL: test_08
; CHECK: [[L08:r[0-9:]+]] = vsxtbh(r0)
; CHECK: [[R08:r[0-9:]+]] = vsxtbh(r1)
; CHECK: [[P08:p[0-3]+]] = vcmph.gtu([[L08]],[[R08]])
; CHECK-NOT: not([[P08]])
define <4 x i8> @test_08(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = icmp ugt <4 x i8> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i8>
  ret <4 x i8> %v1
}

; CHECK-LABEL: test_09
; CHECK: [[L09:r[0-9:]+]] = vsxtbh(r0)
; CHECK: [[R09:r[0-9:]+]] = vsxtbh(r1)
; CHECK: [[P09:p[0-3]+]] = vcmph.gtu([[R09]],[[L09]])
; CHECK: not([[P09]])
define <4 x i8> @test_09(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = icmp uge <4 x i8> %a0, %a1
  %v1 = sext <4 x i1> %v0 to <4 x i8>
  ret <4 x i8> %v1
}


; CHECK-LABEL: test_10
; CHECK: [[L10:r[0-9:]+]] = vsxthw(r0)
; CHECK: [[R10:r[0-9:]+]] = vsxthw(r1)
; CHECK: [[P10:p[0-3]+]] = vcmpw.eq([[L10]],[[R10]])
; CHECK-NOT: not([[P10]])
define <2 x i16> @test_10(<2 x i16> %a0, <2 x i16> %a1) #0 {
  %v0 = icmp eq <2 x i16> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i16>
  ret <2 x i16> %v1
}

; CHECK-LABEL: test_11
; CHECK: [[L11:r[0-9:]+]] = vsxthw(r0)
; CHECK: [[R11:r[0-9:]+]] = vsxthw(r1)
; CHECK: [[P11:p[0-3]+]] = vcmpw.eq([[L11]],[[R11]])
; CHECK: not([[P11]])
define <2 x i16> @test_11(<2 x i16> %a0, <2 x i16> %a1) #0 {
  %v0 = icmp ne <2 x i16> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i16>
  ret <2 x i16> %v1
}

; CHECK-LABEL: test_12
; CHECK: [[L12:r[0-9:]+]] = vsxthw(r0)
; CHECK: [[R12:r[0-9:]+]] = vsxthw(r1)
; CHECK: [[P12:p[0-3]+]] = vcmpw.gt([[R12]],[[L12]])
; CHECK-NOT: not([[P12]])
define <2 x i16> @test_12(<2 x i16> %a0, <2 x i16> %a1) #0 {
  %v0 = icmp slt <2 x i16> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i16>
  ret <2 x i16> %v1
}

; CHECK-LABEL: test_13
; CHECK: [[L13:r[0-9:]+]] = vsxthw(r0)
; CHECK: [[R13:r[0-9:]+]] = vsxthw(r1)
; CHECK: [[P13:p[0-3]+]] = vcmpw.gt([[L13]],[[R13]])
; CHECK: not([[P13]])
define <2 x i16> @test_13(<2 x i16> %a0, <2 x i16> %a1) #0 {
  %v0 = icmp sle <2 x i16> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i16>
  ret <2 x i16> %v1
}

; CHECK-LABEL: test_14
; CHECK: [[L14:r[0-9:]+]] = vsxthw(r0)
; CHECK: [[R14:r[0-9:]+]] = vsxthw(r1)
; CHECK: [[P14:p[0-3]+]] = vcmpw.gt([[L14]],[[R14]])
; CHECK-NOT: not([[P14]])
define <2 x i16> @test_14(<2 x i16> %a0, <2 x i16> %a1) #0 {
  %v0 = icmp sgt <2 x i16> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i16>
  ret <2 x i16> %v1
}

; CHECK-LABEL: test_15
; CHECK: [[L15:r[0-9:]+]] = vsxthw(r0)
; CHECK: [[R15:r[0-9:]+]] = vsxthw(r1)
; CHECK: [[P15:p[0-3]+]] = vcmpw.gt([[R15]],[[L15]])
; CHECK: not([[P15]])
define <2 x i16> @test_15(<2 x i16> %a0, <2 x i16> %a1) #0 {
  %v0 = icmp sge <2 x i16> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i16>
  ret <2 x i16> %v1
}

; CHECK-LABEL: test_16
; CHECK: [[L16:r[0-9:]+]] = vsxthw(r0)
; CHECK: [[R16:r[0-9:]+]] = vsxthw(r1)
; CHECK: [[P16:p[0-3]+]] = vcmpw.gtu([[R16]],[[L16]])
; CHECK-NOT: not([[P16]])
define <2 x i16> @test_16(<2 x i16> %a0, <2 x i16> %a1) #0 {
  %v0 = icmp ult <2 x i16> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i16>
  ret <2 x i16> %v1
}

; CHECK-LABEL: test_17
; CHECK: [[L17:r[0-9:]+]] = vsxthw(r0)
; CHECK: [[R17:r[0-9:]+]] = vsxthw(r1)
; CHECK: [[P17:p[0-3]+]] = vcmpw.gtu([[L17]],[[R17]])
; CHECK: not([[P17]])
define <2 x i16> @test_17(<2 x i16> %a0, <2 x i16> %a1) #0 {
  %v0 = icmp ule <2 x i16> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i16>
  ret <2 x i16> %v1
}

; CHECK-LABEL: test_18
; CHECK: [[L18:r[0-9:]+]] = vsxthw(r0)
; CHECK: [[R18:r[0-9:]+]] = vsxthw(r1)
; CHECK: [[P18:p[0-3]+]] = vcmpw.gtu([[L18]],[[R18]])
; CHECK-NOT: not([[P18]])
define <2 x i16> @test_18(<2 x i16> %a0, <2 x i16> %a1) #0 {
  %v0 = icmp ugt <2 x i16> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i16>
  ret <2 x i16> %v1
}

; CHECK-LABEL: test_19
; CHECK: [[L19:r[0-9:]+]] = vsxthw(r0)
; CHECK: [[R19:r[0-9:]+]] = vsxthw(r1)
; CHECK: [[P19:p[0-3]+]] = vcmpw.gtu([[R19]],[[L19]])
; CHECK: not([[P19]])
define <2 x i16> @test_19(<2 x i16> %a0, <2 x i16> %a1) #0 {
  %v0 = icmp uge <2 x i16> %a0, %a1
  %v1 = sext <2 x i1> %v0 to <2 x i16>
  ret <2 x i16> %v1
}

attributes #0 = { nounwind readnone }
