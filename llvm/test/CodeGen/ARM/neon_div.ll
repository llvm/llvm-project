; RUN: llc -mtriple arm-eabi -mattr=+neon -disable-post-ra -pre-RA-sched source %s -o - | FileCheck %s
; RUN: llc -mtriple thumbv7-windows-itanium -mattr=+neon -disable-post-ra -pre-RA-sched source %s -o - | FileCheck %s

define <8 x i8> @sdivi8(ptr %A, ptr %B) nounwind {
  %tmp1 = load <8 x i8>, ptr %A
  %tmp2 = load <8 x i8>, ptr %B
  %tmp3 = sdiv <8 x i8> %tmp1, %tmp2
  ret <8 x i8> %tmp3
}

; CHECK-LABEL: sdivi8:
; CHECK: vrecpe.f32
; CHECK: vmovn.i32
; CHECK: vrecpe.f32
; CHECK: vmovn.i32
; CHECK: vmovn.i16

define <8 x i8> @udivi8(ptr %A, ptr %B) nounwind {
  %tmp1 = load <8 x i8>, ptr %A
  %tmp2 = load <8 x i8>, ptr %B
  %tmp3 = udiv <8 x i8> %tmp1, %tmp2
  ret <8 x i8> %tmp3
}

; CHECK-LABEL: udivi8:
; CHECK: vrecpe.f32
; CHECK: vrecps.f32
; CHECK: vmovn.i32
; CHECK: vrecpe.f32
; CHECK: vrecps.f32
; CHECK: vmovn.i32
; CHECK: vqmovun.s16

define <4 x i16> @sdivi16(ptr %A, ptr %B) nounwind {
  %tmp1 = load <4 x i16>, ptr %A
  %tmp2 = load <4 x i16>, ptr %B
  %tmp3 = sdiv <4 x i16> %tmp1, %tmp2
  ret <4 x i16> %tmp3
}

; CHECK-LABEL: sdivi16:
; CHECK: vrecpe.f32
; CHECK: vrecps.f32
; CHECK: vmovn.i32

define <4 x i16> @udivi16(ptr %A, ptr %B) nounwind {
  %tmp1 = load <4 x i16>, ptr %A
  %tmp2 = load <4 x i16>, ptr %B
  %tmp3 = udiv <4 x i16> %tmp1, %tmp2
  ret <4 x i16> %tmp3
}

; CHECK-LABEL: udivi16:
; CHECK: vrecpe.f32
; CHECK: vrecps.f32
; CHECK: vrecps.f32
; CHECK: vmovn.i32

