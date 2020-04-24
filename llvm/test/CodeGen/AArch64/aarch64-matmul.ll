; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon,+i8mm < %s -o -| FileCheck %s

define <4 x i32> @smmla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: smmla.v4i32.v16i8
; CHECK: smmla   v0.4s, v1.16b, v2.16b
  %vmmla1.i = tail call <4 x i32> @llvm.aarch64.neon.smmla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b)
  ret <4 x i32> %vmmla1.i
}

define <4 x i32> @ummla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: ummla.v4i32.v16i8
; CHECK: ummla   v0.4s, v1.16b, v2.16b
  %vmmla1.i = tail call <4 x i32> @llvm.aarch64.neon.ummla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b)
  ret <4 x i32> %vmmla1.i
}

define <4 x i32> @usmmla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: usmmla.v4i32.v16i8
; CHECK: usmmla   v0.4s, v1.16b, v2.16b
  %vusmmla1.i = tail call <4 x i32> @llvm.aarch64.neon.usmmla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) #3
  ret <4 x i32> %vusmmla1.i
}

define <2 x i32> @usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %b) {
entry:
; CHECK-LABEL: usdot.v2i32.v8i8
; CHECK: usdot   v0.2s, v1.8b, v2.8b
  %vusdot1.i = tail call <2 x i32> @llvm.aarch64.neon.usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %b)
  ret <2 x i32> %vusdot1.i
}

define <2 x i32> @usdot_lane.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %b) {
entry:
; CHECK-LABEL: usdot_lane.v2i32.v8i8
; CHECK: usdot   v0.2s, v1.8b, v2.4b[0]
  %0 = bitcast <8 x i8> %b to <2 x i32>
  %shuffle = shufflevector <2 x i32> %0, <2 x i32> undef, <2 x i32> zeroinitializer
  %1 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vusdot1.i = tail call <2 x i32> @llvm.aarch64.neon.usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %1)
  ret <2 x i32> %vusdot1.i
}

define <2 x i32> @sudot_lane.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %b) {
entry:
; CHECK-LABEL: sudot_lane.v2i32.v8i8
; CHECK: sudot   v0.2s, v1.8b, v2.4b[0]
  %0 = bitcast <8 x i8> %b to <2 x i32>
  %shuffle = shufflevector <2 x i32> %0, <2 x i32> undef, <2 x i32> zeroinitializer
  %1 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vusdot1.i = tail call <2 x i32> @llvm.aarch64.neon.usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %1, <8 x i8> %a)
  ret <2 x i32> %vusdot1.i
}

define <2 x i32> @usdot_lane.v2i32.v16i8(<2 x i32> %r, <8 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: usdot_lane.v2i32.v16i8
; CHECK: usdot   v0.2s, v1.8b, v2.4b[0]
  %0 = bitcast <16 x i8> %b to <4 x i32>
  %shuffle = shufflevector <4 x i32> %0, <4 x i32> undef, <2 x i32> zeroinitializer
  %1 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vusdot1.i = tail call <2 x i32> @llvm.aarch64.neon.usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %1)
  ret <2 x i32> %vusdot1.i
}

define <2 x i32> @sudot_lane.v2i32.v16i8(<2 x i32> %r, <8 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: sudot_lane.v2i32.v16i8
; CHECK: sudot   v0.2s, v1.8b, v2.4b[0]
  %0 = bitcast <16 x i8> %b to <4 x i32>
  %shuffle = shufflevector <4 x i32> %0, <4 x i32> undef, <2 x i32> zeroinitializer
  %1 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vusdot1.i = tail call <2 x i32> @llvm.aarch64.neon.usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %1, <8 x i8> %a) #3
  ret <2 x i32> %vusdot1.i
}

define <4 x i32> @usdot.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: usdot.v4i32.v16i8
; CHECK: usdot   v0.4s, v1.16b, v2.16b
  %vusdot1.i = tail call <4 x i32> @llvm.aarch64.neon.usdot.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) #3
  ret <4 x i32> %vusdot1.i
}

define <4 x i32> @usdot_lane.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <8 x i8> %b) {
entry:
; CHECK-LABEL: usdot_lane.v4i32.v16i8
; CHECK: usdot   v0.4s, v1.16b, v2.4b[0]
  %0 = bitcast <8 x i8> %b to <2 x i32>
  %shuffle = shufflevector <2 x i32> %0, <2 x i32> undef, <4 x i32> zeroinitializer
  %1 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vusdot1.i = tail call <4 x i32> @llvm.aarch64.neon.usdot.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %1) #3
  ret <4 x i32> %vusdot1.i
}

define <4 x i32> @sudot_lane.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <8 x i8> %b) {
entry:
; CHECK-LABEL: sudot_lane.v4i32.v16i8
; CHECK: sudot   v0.4s, v1.16b, v2.4b[0]
  %0 = bitcast <8 x i8> %b to <2 x i32>
  %shuffle = shufflevector <2 x i32> %0, <2 x i32> undef, <4 x i32> zeroinitializer
  %1 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vusdot1.i = tail call <4 x i32> @llvm.aarch64.neon.usdot.v4i32.v16i8(<4 x i32> %r, <16 x i8> %1, <16 x i8> %a) #3
  ret <4 x i32> %vusdot1.i
}

define <4 x i32> @usdot_laneq.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: usdot_laneq.v4i32.v16i8
; CHECK: usdot   v0.4s, v1.16b, v2.4b[0]
  %0 = bitcast <16 x i8> %b to <4 x i32>
  %shuffle = shufflevector <4 x i32> %0, <4 x i32> undef, <4 x i32> zeroinitializer
  %1 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vusdot1.i = tail call <4 x i32> @llvm.aarch64.neon.usdot.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %1) #3
  ret <4 x i32> %vusdot1.i
}

define <4 x i32> @sudot_laneq.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: sudot_laneq.v4i32.v16i8
; CHECK: sudot   v0.4s, v1.16b, v2.4b[0]
  %0 = bitcast <16 x i8> %b to <4 x i32>
  %shuffle = shufflevector <4 x i32> %0, <4 x i32> undef, <4 x i32> zeroinitializer
  %1 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vusdot1.i = tail call <4 x i32> @llvm.aarch64.neon.usdot.v4i32.v16i8(<4 x i32> %r, <16 x i8> %1, <16 x i8> %a) #3
  ret <4 x i32> %vusdot1.i
}

declare <4 x i32> @llvm.aarch64.neon.smmla.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) #2
declare <4 x i32> @llvm.aarch64.neon.ummla.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) #2
declare <4 x i32> @llvm.aarch64.neon.usmmla.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) #2
declare <2 x i32> @llvm.aarch64.neon.usdot.v2i32.v8i8(<2 x i32>, <8 x i8>, <8 x i8>) #2
declare <4 x i32> @llvm.aarch64.neon.usdot.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) #2

