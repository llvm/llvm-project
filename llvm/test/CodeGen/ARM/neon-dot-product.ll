; RUN: llc -mtriple armv8a-none-linux-gnu -mattr=+dotprod -float-abi=hard < %s | FileCheck %s

declare <2 x i32> @llvm.arm.neon.udot.v2i32.v8i8(<2 x i32>, <8 x i8>, <8 x i8>)
declare <4 x i32> @llvm.arm.neon.udot.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>)
declare <2 x i32> @llvm.arm.neon.sdot.v2i32.v8i8(<2 x i32>, <8 x i8>, <8 x i8>)
declare <4 x i32> @llvm.arm.neon.sdot.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>)

define <2 x i32> @test_vdot_u32(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) #0 {
entry:
; CHECK-LABEL: test_vdot_u32:
; CHECK: vudot.u8        d0, d1, d2
  %vdot1.i = call <2 x i32> @llvm.arm.neon.udot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) #2
  ret <2 x i32> %vdot1.i
}

define <4 x i32> @test_vdotq_u32(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c) #0 {
entry:
; CHECK-LABEL: test_vdotq_u32:
; CHECK: vudot.u8        q0, q1, q2
  %vdot1.i = call <4 x i32> @llvm.arm.neon.udot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c) #2
  ret <4 x i32> %vdot1.i
}

define <2 x i32> @test_vdot_s32(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) #0 {
entry:
; CHECK-LABEL: test_vdot_s32:
; CHECK: vsdot.s8        d0, d1, d2
  %vdot1.i = call <2 x i32> @llvm.arm.neon.sdot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) #2
  ret <2 x i32> %vdot1.i
}

define <4 x i32> @test_vdotq_s32(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c) #0 {
entry:
; CHECK-LABEL: test_vdotq_s32:
; CHECK: vsdot.s8        q0, q1, q2
  %vdot1.i = call <4 x i32> @llvm.arm.neon.sdot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c) #2
  ret <4 x i32> %vdot1.i
}

define <2 x i32> @test_vdot_lane_u32(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) {
entry:
; CHECK-LABEL: test_vdot_lane_u32:
; CHECK: vudot.u8        d0, d1, d2[1]
  %.cast = bitcast <8 x i8> %c to <2 x i32>
  %shuffle = shufflevector <2 x i32> %.cast, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %.cast5 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vdot1.i = call <2 x i32> @llvm.arm.neon.udot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %.cast5) #2
  ret <2 x i32> %vdot1.i
}

define <4 x i32> @test_vdotq_lane_u32(<4 x i32> %a, <16 x i8> %b, <8 x i8> %c) {
entry:
; CHECK-LABEL: test_vdotq_lane_u32:
; CHECK: vudot.u8        q0, q1, d4[1]
  %.cast = bitcast <8 x i8> %c to <2 x i32>
  %shuffle = shufflevector <2 x i32> %.cast, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %.cast3 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vdot1.i = call <4 x i32> @llvm.arm.neon.udot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %.cast3) #2
  ret <4 x i32> %vdot1.i
}

define <2 x i32> @test_vdot_lane_s32(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c) {
entry:
; CHECK-LABEL: test_vdot_lane_s32:
; CHECK: vsdot.s8        d0, d1, d2[1]
  %.cast = bitcast <8 x i8> %c to <2 x i32>
  %shuffle = shufflevector <2 x i32> %.cast, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %.cast5 = bitcast <2 x i32> %shuffle to <8 x i8>
  %vdot1.i = call <2 x i32> @llvm.arm.neon.sdot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %.cast5) #2
  ret <2 x i32> %vdot1.i
}

define <4 x i32> @test_vdotq_lane_s32(<4 x i32> %a, <16 x i8> %b, <8 x i8> %c) {
entry:
; CHECK-LABEL: test_vdotq_lane_s32:
; CHECK: vsdot.s8        q0, q1, d4[1]
  %.cast = bitcast <8 x i8> %c to <2 x i32>
  %shuffle = shufflevector <2 x i32> %.cast, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %.cast3 = bitcast <4 x i32> %shuffle to <16 x i8>
  %vdot1.i = call <4 x i32> @llvm.arm.neon.sdot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %.cast3) #2
  ret <4 x i32> %vdot1.i
}
