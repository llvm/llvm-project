; RUN: llc < %s -mtriple=armv8 -mattr=+v8.1a | FileCheck %s

;-----------------------------------------------------------------------------
; RDMA Vector

declare <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16>, <8 x i16>)
declare <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32>, <4 x i32>)

declare <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16>, <8 x i16>)
declare <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32>, <4 x i32>)

declare <4 x i16> @llvm.ssub.sat.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16>, <8 x i16>)
declare <2 x i32> @llvm.ssub.sat.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32>, <4 x i32>)

define <4 x i16> @test_vqrdmlah_v4i16(<4 x i16> %acc, <4 x i16> %mhs, <4 x i16> %rhs) {
; CHECK-LABEL: test_vqrdmlah_v4i16:
   %prod = call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> %mhs,  <4 x i16> %rhs)
   %retval =  call <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16> %acc,  <4 x i16> %prod)
; CHECK: vqrdmlah.s16 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
   ret <4 x i16> %retval
}

define <8 x i16> @test_vqrdmlah_v8i16(<8 x i16> %acc, <8 x i16> %mhs, <8 x i16> %rhs) {
; CHECK-LABEL: test_vqrdmlah_v8i16:
   %prod = call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> %mhs, <8 x i16> %rhs)
   %retval =  call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> %acc, <8 x i16> %prod)
; CHECK: vqrdmlah.s16 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
   ret <8 x i16> %retval
}

define <2 x i32> @test_vqrdmlah_v2i32(<2 x i32> %acc, <2 x i32> %mhs, <2 x i32> %rhs) {
; CHECK-LABEL: test_vqrdmlah_v2i32:
   %prod = call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> %mhs, <2 x i32> %rhs)
   %retval =  call <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32> %acc, <2 x i32> %prod)
; CHECK: vqrdmlah.s32 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
   ret <2 x i32> %retval
}

define <4 x i32> @test_vqrdmlah_v4i32(<4 x i32> %acc, <4 x i32> %mhs, <4 x i32> %rhs) {
; CHECK-LABEL: test_vqrdmlah_v4i32:
   %prod = call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> %mhs, <4 x i32> %rhs)
   %retval =  call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> %acc, <4 x i32> %prod)
; CHECK: vqrdmlah.s32 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
   ret <4 x i32> %retval
}

define <4 x i16> @test_vqrdmlsh_v4i16(<4 x i16> %acc, <4 x i16> %mhs, <4 x i16> %rhs) {
; CHECK-LABEL: test_vqrdmlsh_v4i16:
   %prod = call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> %mhs,  <4 x i16> %rhs)
   %retval =  call <4 x i16> @llvm.ssub.sat.v4i16(<4 x i16> %acc, <4 x i16> %prod)
; CHECK: vqrdmlsh.s16 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
   ret <4 x i16> %retval
}

define <8 x i16> @test_vqrdmlsh_v8i16(<8 x i16> %acc, <8 x i16> %mhs, <8 x i16> %rhs) {
; CHECK-LABEL: test_vqrdmlsh_v8i16:
   %prod = call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> %mhs, <8 x i16> %rhs)
   %retval =  call <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16> %acc, <8 x i16> %prod)
; CHECK: vqrdmlsh.s16 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
   ret <8 x i16> %retval
}

define <2 x i32> @test_vqrdmlsh_v2i32(<2 x i32> %acc, <2 x i32> %mhs, <2 x i32> %rhs) {
; CHECK-LABEL: test_vqrdmlsh_v2i32:
   %prod = call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> %mhs, <2 x i32> %rhs)
   %retval =  call <2 x i32> @llvm.ssub.sat.v2i32(<2 x i32> %acc, <2 x i32> %prod)
; CHECK: vqrdmlsh.s32 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
   ret <2 x i32> %retval
}

define <4 x i32> @test_vqrdmlsh_v4i32(<4 x i32> %acc, <4 x i32> %mhs, <4 x i32> %rhs) {
; CHECK-LABEL: test_vqrdmlsh_v4i32:
   %prod = call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> %mhs, <4 x i32> %rhs)
   %retval =  call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> %acc, <4 x i32> %prod)
; CHECK: vqrdmlsh.s32 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
   ret <4 x i32> %retval
}

;-----------------------------------------------------------------------------
; RDMA Scalar

define <4 x i16> @test_vqrdmlah_lane_s16(<4 x i16> %acc, <4 x i16> %x, <4 x i16> %v) {
; CHECK-LABEL: test_vqrdmlah_lane_s16:
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %prod = call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> %x, <4 x i16> %shuffle)
  %retval =  call <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16> %acc, <4 x i16> %prod)
; CHECK: vqrdmlah.s16 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}[3]
  ret <4 x i16> %retval
}

define <8 x i16> @test_vqrdmlahq_lane_s16(<8 x i16> %acc, <8 x i16> %x, <4 x i16> %v) {
; CHECK-LABEL: test_vqrdmlahq_lane_s16:
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %prod = call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> %x, <8 x i16> %shuffle)
  %retval =  call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> %acc, <8 x i16> %prod)
; CHECK: vqrdmlah.s16 {{q[0-9]+}}, {{q[0-9]+}}, {{d[0-9]+}}[2]
  ret <8 x i16> %retval
}

define <2 x i32> @test_vqrdmlah_lane_s32(<2 x i32> %acc, <2 x i32> %x, <2 x i32> %v) {
; CHECK-LABEL: test_vqrdmlah_lane_s32:
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %prod = tail call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> %x, <2 x i32> %shuffle)
  %retval =  call <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32> %acc, <2 x i32> %prod)
; CHECK: vqrdmlah.s32 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}[1]
  ret <2 x i32> %retval
}

define <4 x i32> @test_vqrdmlahq_lane_s32(<4 x i32> %acc,<4 x i32> %x, <2 x i32> %v) {
; CHECK-LABEL: test_vqrdmlahq_lane_s32:
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> zeroinitializer
  %prod = tail call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> %x, <4 x i32> %shuffle)
  %retval =  call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> %acc, <4 x i32> %prod)
; CHECK: vqrdmlah.s32 {{q[0-9]+}}, {{q[0-9]+}}, {{d[0-9]+}}[0]
  ret <4 x i32> %retval
}

define <4 x i16> @test_vqrdmlsh_lane_s16(<4 x i16> %acc, <4 x i16> %x, <4 x i16> %v) {
; CHECK-LABEL: test_vqrdmlsh_lane_s16:
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %prod = call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> %x, <4 x i16> %shuffle)
  %retval =  call <4 x i16> @llvm.ssub.sat.v4i16(<4 x i16> %acc, <4 x i16> %prod)
; CHECK: vqrdmlsh.s16 {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}[3]
  ret <4 x i16> %retval
}

define <8 x i16> @test_vqrdmlshq_lane_s16(<8 x i16> %acc, <8 x i16> %x, <4 x i16> %v) {
; CHECK-LABEL: test_vqrdmlshq_lane_s16:
entry:
  %shuffle = shufflevector <4 x i16> %v, <4 x i16> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %prod = call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> %x, <8 x i16> %shuffle)
  %retval =  call <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16> %acc, <8 x i16> %prod)
; CHECK: vqrdmlsh.s16 {{q[0-9]+}}, {{q[0-9]+}}, {{d[0-9]+}}[2]
  ret <8 x i16> %retval
}

define <2 x i32> @test_vqrdmlsh_lane_s32(<2 x i32> %acc, <2 x i32> %x, <2 x i32> %v) {
; CHECK-LABEL: test_vqrdmlsh_lane_s32:
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  %prod = tail call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> %x, <2 x i32> %shuffle)
  %retval =  call <2 x i32> @llvm.ssub.sat.v2i32(<2 x i32> %acc, <2 x i32> %prod)
; CHECK: vqrdmlsh.s32  {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}[1]
  ret <2 x i32> %retval
}

define <4 x i32> @test_vqrdmlshq_lane_s32(<4 x i32> %acc,<4 x i32> %x, <2 x i32> %v) {
; CHECK-LABEL: test_vqrdmlshq_lane_s32:
entry:
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> undef, <4 x i32> zeroinitializer
  %prod = tail call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> %x, <4 x i32> %shuffle)
  %retval =  call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> %acc, <4 x i32> %prod)
; CHECK: vqrdmlsh.s32 {{q[0-9]+}}, {{q[0-9]+}}, {{d[0-9]+}}[0]
  ret <4 x i32> %retval
}
