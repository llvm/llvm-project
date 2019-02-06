; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     -mcpu=pwr8 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN: FileCheck %s --check-prefix=CHECK-P8
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     -mcpu=pwr9 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN: FileCheck %s --check-prefix=CHECK-P9
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     -mcpu=pwr9 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN: FileCheck %s --check-prefix=CHECK-BE

define void @test8(<8 x double>* nocapture %Sink, <8 x i16>* nocapture readonly %SrcPtr) {
entry:
  %0 = load <8 x i16>, <8 x i16>* %SrcPtr, align 16
  %1 = uitofp <8 x i16> %0 to <8 x double>
  store <8 x double> %1, <8 x double>* %Sink, align 16
  ret void
; CHECK-P9-LABEL: @test8
; CHECK-P9: vperm
; CHECK-P9: xvcvuxddp
; CHECK-P9: vperm
; CHECK-P9: xvcvuxddp
; CHECK-P9: vperm
; CHECK-P9: xvcvuxddp
; CHECK-P9: vperm
; CHECK-P9: xvcvuxddp
; CHECK-P8-LABEL: @test8
; CHECK-P8: vperm
; CHECK-P8: vperm
; CHECK-P8: vperm
; CHECK-P8: vperm
; CHECK-P8: xvcvuxddp
; CHECK-P8: xvcvuxddp
; CHECK-P8: xvcvuxddp
; CHECK-P8: xvcvuxddp
}

define void @test4(<4 x double>* nocapture %Sink, <4 x i16>* nocapture readonly %SrcPtr) {
entry:
  %0 = load <4 x i16>, <4 x i16>* %SrcPtr, align 16
  %1 = uitofp <4 x i16> %0 to <4 x double>
  store <4 x double> %1, <4 x double>* %Sink, align 16
  ret void
; CHECK-P9-LABEL: @test4
; CHECK-P9: vperm
; CHECK-P9: xvcvuxddp
; CHECK-P9: vperm
; CHECK-P9: xvcvuxddp
; CHECK-P8-LABEL: @test4
; CHECK-P8: vperm
; CHECK-P8: vperm
; CHECK-P8: xvcvuxddp
; CHECK-P8: xvcvuxddp
}

define void @test2(<2 x double>* nocapture %Sink, <2 x i16>* nocapture readonly %SrcPtr) {
entry:
  %0 = load <2 x i16>, <2 x i16>* %SrcPtr, align 16
  %1 = uitofp <2 x i16> %0 to <2 x double>
  store <2 x double> %1, <2 x double>* %Sink, align 16
  ret void
; CHECK-P9-LABEL: .LCPI2_0:
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 30
; CHECK-P9-NEXT: .byte 13
; CHECK-P9-NEXT: .byte 12
; CHECK-P9-NEXT: .byte 11
; CHECK-P9-NEXT: .byte 10
; CHECK-P9-NEXT: .byte 9
; CHECK-P9-NEXT: .byte 8
; CHECK-P9-NEXT: .byte 29
; CHECK-P9-NEXT: .byte 28
; CHECK-P9-NEXT: .byte 5
; CHECK-P9-NEXT: .byte 4
; CHECK-P9-NEXT: .byte 3
; CHECK-P9-NEXT: .byte 2
; CHECK-P9-NEXT: .byte 1
; CHECK-P9-NEXT: .byte 0
; CHECK-P9: addi [[REG1:r[0-9]+]], {{r[0-9]+}}, .LCPI2_0@toc@l
; CHECK-P9: lxvx [[REG2:v[0-9]+]], 0, [[REG1]]
; CHECK-P9: vperm [[REG3:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}, [[REG2]]
; CHECK-P9: xvcvuxddp {{vs[0-9]+}}, [[REG3]]
; CHECK-P8-LABEL: @test2
; CHECK-P8: vperm [[REG1:v[0-9]+]]
; CHECK-P8: xvcvuxddp {{vs[0-9]+}}, [[REG1]]
; CHECK-BE-LABEL: .LCPI2_0:
; CHECK-BE-NEXT: .byte 16
; CHECK-BE-NEXT: .byte 17
; CHECK-BE-NEXT: .byte 18
; CHECK-BE-NEXT: .byte 19
; CHECK-BE-NEXT: .byte 20
; CHECK-BE-NEXT: .byte 21
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 1
; CHECK-BE-NEXT: .byte 24
; CHECK-BE-NEXT: .byte 25
; CHECK-BE-NEXT: .byte 26
; CHECK-BE-NEXT: .byte 27
; CHECK-BE-NEXT: .byte 28
; CHECK-BE-NEXT: .byte 29
; CHECK-BE-NEXT: .byte 2
; CHECK-BE-NEXT: .byte 3
; CHECK-BE: addi [[REG1:r[0-9]+]], {{r[0-9]+}}, .LCPI2_0@toc@l
; CHECK-BE: lxvx [[REG2:v[0-9]+]], 0, [[REG1]]
; CHECK-BE: vperm [[REG3:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}, [[REG2]]
; CHECK-BE: xvcvuxddp {{vs[0-9]+}}, [[REG3]]
}

define void @stest8(<8 x double>* nocapture %Sink, <8 x i16>* nocapture readonly %SrcPtr) {
entry:
  %0 = load <8 x i16>, <8 x i16>* %SrcPtr, align 16
  %1 = sitofp <8 x i16> %0 to <8 x double>
  store <8 x double> %1, <8 x double>* %Sink, align 16
  ret void
; CHECK-P9-LABEL: @stest8
; CHECK-P9: vperm
; CHECK-P9: vextsh2d
; CHECK-P9: xvcvsxddp
; CHECK-P9: vperm
; CHECK-P9: vextsh2d
; CHECK-P9: xvcvsxddp
; CHECK-P9: vperm
; CHECK-P9: vextsh2d
; CHECK-P9: xvcvsxddp
; CHECK-P9: vperm
; CHECK-P9: vextsh2d
; CHECK-P9: xvcvsxddp
}

define void @stest4(<4 x double>* nocapture %Sink, <4 x i16>* nocapture readonly %SrcPtr) {
entry:
  %0 = load <4 x i16>, <4 x i16>* %SrcPtr, align 16
  %1 = sitofp <4 x i16> %0 to <4 x double>
  store <4 x double> %1, <4 x double>* %Sink, align 16
  ret void
; CHECK-P9-LABEL: @stest4
; CHECK-P9: vperm
; CHECK-P9: vextsh2d
; CHECK-P9: xvcvsxddp
; CHECK-P9: vperm
; CHECK-P9: vextsh2d
; CHECK-P9: xvcvsxddp
}

define void @stest2(<2 x double>* nocapture %Sink, <2 x i16>* nocapture readonly %SrcPtr) {
entry:
  %0 = load <2 x i16>, <2 x i16>* %SrcPtr, align 16
  %1 = sitofp <2 x i16> %0 to <2 x double>
  store <2 x double> %1, <2 x double>* %Sink, align 16
  ret void
; CHECK-P9-LABEL: .LCPI5_0:
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 30
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 29
; CHECK-P9-NEXT: .byte 28
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 31
; CHECK-P9-NEXT: .byte 31
; CHECK-P9: vperm [[REG1:v[0-9]+]]
; CHECK-P9: vextsh2d [[REG2:v[0-9]+]], [[REG1]]
; CHECK-P9: xvcvsxddp {{vs[0-9]+}}, [[REG2]]
; CHECK-BE-LABEL: .LCPI5_0:
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 1
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 0
; CHECK-BE-NEXT: .byte 2
; CHECK-BE-NEXT: .byte 3
; CHECK-BE: addi [[REG1:r[0-9]+]], {{r[0-9]+}}, .LCPI5_0@toc@l
; CHECK-BE: lxvx [[REG2:v[0-9]+]], 0, [[REG1]]
; CHECK-BE: vperm [[REG3:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}, [[REG2]]
; CHECK-BE: vextsh2d [[REG4:v[0-9]+]], [[REG3]]
; CHECK-BE: xvcvsxddp {{vs[0-9]+}}, [[REG4]]
}
