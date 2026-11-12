; RUN: llc -mtriple=arm-none-linux-gnu -mattr=+neon,+i8mm -float-abi=hard < %s -o -| FileCheck %s

define <4 x i32> @smmla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: smmla.v4i32.v16i8
; CHECK:        vsmmla.s8       q0, q1, q2
  %vmmla1.i = tail call <4 x i32> @llvm.arm.neon.smmla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) #3
  ret <4 x i32> %vmmla1.i
}

define <4 x i32> @ummla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: ummla.v4i32.v16i8
; CHECK:        vummla.u8       q0, q1, q2
  %vmmla1.i = tail call <4 x i32> @llvm.arm.neon.ummla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) #3
  ret <4 x i32> %vmmla1.i
}

define <4 x i32> @usmmla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: usmmla.v4i32.v16i8
; CHECK:        vusmmla.s8       q0, q1, q2
  %vusmmla1.i = tail call <4 x i32> @llvm.arm.neon.usmmla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) #3
  ret <4 x i32> %vusmmla1.i
}

define <2 x i32> @usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %b) {
entry:
; CHECK-LABEL: usdot.v2i32.v8i8
; CHECK:        vusdot.s8       d0, d1, d2
  %vusdot1.i = tail call <2 x i32> @llvm.arm.neon.usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %b) #3
  ret <2 x i32> %vusdot1.i
}

define <2 x i32> @usdot_lane.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %b) {
entry:
; CHECK-LABEL: usdot_lane.v2i32.v8i8
; CHECK:        vusdot.s8       d0, d1, d2[0]
  %0 = bitcast <8 x i8> %b to <2 x i32>
  %shuffle = shufflevector <2 x i32> %0, <2 x i32> undef, <2 x i32> zeroinitializer
  %1 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vusdot1.i = tail call <2 x i32> @llvm.arm.neon.usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %1) #3
  ret <2 x i32> %vusdot1.i
}

define <2 x i32> @sudot_lane.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %b) {
entry:
; CHECK-LABEL: sudot_lane.v2i32.v8i8
; CHECK:        vsudot.u8       d0, d1, d2[0]
  %0 = bitcast <8 x i8> %b to <2 x i32>
  %shuffle = shufflevector <2 x i32> %0, <2 x i32> undef, <2 x i32> zeroinitializer
  %1 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vusdot1.i = tail call <2 x i32> @llvm.arm.neon.usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %1, <8 x i8> %a) #3
  ret <2 x i32> %vusdot1.i
}

define <4 x i32> @usdotq_lane.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <8 x i8> %b) {
entry:
; CHECK-LABEL: usdotq_lane.v4i32.v16i8
; CHECK:        vusdot.s8       q0, q1, d4[0]
  %0 = bitcast <8 x i8> %b to <2 x i32>
  %shuffle = shufflevector <2 x i32> %0, <2 x i32> undef, <4 x i32> zeroinitializer
  %1 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vusdot1.i = tail call <4 x i32> @llvm.arm.neon.usdot.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %1) #3
  ret <4 x i32> %vusdot1.i
}

define <4 x i32> @sudotq_lane.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <8 x i8> %b) {
entry:
; CHECK-LABEL: sudotq_lane.v4i32.v16i8
; CHECK:        vsudot.u8       q0, q1, d4[0]
  %0 = bitcast <8 x i8> %b to <2 x i32>
  %shuffle = shufflevector <2 x i32> %0, <2 x i32> undef, <4 x i32> zeroinitializer
  %1 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vusdot1.i = tail call <4 x i32> @llvm.arm.neon.usdot.v4i32.v16i8(<4 x i32> %r, <16 x i8> %1, <16 x i8> %a) #3
  ret <4 x i32> %vusdot1.i
}

declare <4 x i32> @llvm.arm.neon.smmla.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) #2
declare <4 x i32> @llvm.arm.neon.ummla.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) #2
declare <4 x i32> @llvm.arm.neon.usmmla.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) #2
declare <2 x i32> @llvm.arm.neon.usdot.v2i32.v8i8(<2 x i32>, <8 x i8>, <8 x i8>) #2
declare <4 x i32> @llvm.arm.neon.usdot.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) #2
