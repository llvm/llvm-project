; RUN: llc -fast-isel -mattr=+sse < %s | FileCheck %s
; <rdar://problem/10215997>
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.7"

define void @vectortest() nounwind ssp {
entry:
  %p1 = alloca <4 x float>, align 16
  %p2 = alloca <4 x float>, align 16
  %p3 = alloca <4 x float>, align 16
  %p4 = alloca <4 x float>, align 16
  %p5 = alloca <4 x float>, align 16
  store <4 x float> <float 0x3FF19999A0000000, float 0x3FF3333340000000, float 0x3FF4CCCCC0000000, float 0x3FF6666660000000>, ptr %p1, align 16
  store <4 x float> <float 0x4000CCCCC0000000, float 0x40019999A0000000, float 0x4002666660000000, float 0x4003333340000000>, ptr %p2, align 16
  store <4 x float> <float 0x4008CCCCC0000000, float 0x40099999A0000000, float 0x400A666660000000, float 0x400B333340000000>, ptr %p3, align 16
  store <4 x float> <float 0x4010666660000000, float 0x4010CCCCC0000000, float 0x4011333340000000, float 0x40119999A0000000>, ptr %p4, align 16
  store <4 x float> <float 0x4014666660000000, float 0x4014CCCCC0000000, float 0x4015333340000000, float 0x40159999A0000000>, ptr %p5, align 16
  %0 = load <4 x float>, ptr %p1, align 16
  %1 = load <4 x float>, ptr %p2, align 16
  %2 = load <4 x float>, ptr %p3, align 16
  %3 = load <4 x float>, ptr %p4, align 16
  %4 = load <4 x float>, ptr %p5, align 16
; CHECK:      movups {{%xmm[0-7]}}, (%esp)
; CHECK-NEXT: calll _dovectortest 
  call void @dovectortest(<4 x float> %0, <4 x float> %1, <4 x float> %2, <4 x float> %3, <4 x float> %4)
  ret void
}

declare void @dovectortest(<4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>)
