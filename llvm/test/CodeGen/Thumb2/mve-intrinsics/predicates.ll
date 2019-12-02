; RUN: opt -instcombine %s | llc -mtriple=thumbv8.1m.main -mattr=+mve.fp -verify-machineinstrs -o - | FileCheck %s

declare <16 x i1> @llvm.arm.mve.vctp8(i32)
declare <8 x i1> @llvm.arm.mve.vctp16(i32)
declare <4 x i1> @llvm.arm.mve.vctp32(i32)
declare <4 x i1> @llvm.arm.mve.vctp64(i32)

declare i32 @llvm.arm.mve.pred.v2i.v4i1(<4 x i1>)
declare i32 @llvm.arm.mve.pred.v2i.v8i1(<8 x i1>)
declare i32 @llvm.arm.mve.pred.v2i.v16i1(<16 x i1>)

declare <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32)
declare <8 x i1> @llvm.arm.mve.pred.i2v.v8i1(i32)
declare <16 x i1> @llvm.arm.mve.pred.i2v.v16i1(i32)

define arm_aapcs_vfpcc zeroext i16 @test_vctp8q(i32 %a) {
; CHECK-LABEL: test_vctp8q:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vctp.8 r0
; CHECK-NEXT:    vmrs r0, p0
; CHECK-NEXT:    bx lr
entry:
  %0 = call <16 x i1> @llvm.arm.mve.vctp8(i32 %a)
  %1 = call i32 @llvm.arm.mve.pred.v2i.v16i1(<16 x i1> %0)
  %2 = trunc i32 %1 to i16
  ret i16 %2
}

define arm_aapcs_vfpcc zeroext i16 @test_vctp8q_m(i32 %a, i16 zeroext %p) {
; CHECK-LABEL: test_vctp8q_m:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmsr p0, r1
; CHECK-NEXT:    vpst
; CHECK-NEXT:    vctpt.8 r0
; CHECK-NEXT:    vmrs r0, p0
; CHECK-NEXT:    bx lr
entry:
  %0 = zext i16 %p to i32
  %1 = call <16 x i1> @llvm.arm.mve.pred.i2v.v16i1(i32 %0)
  %2 = call <16 x i1> @llvm.arm.mve.vctp8(i32 %a)
  %3 = and <16 x i1> %1, %2
  %4 = call i32 @llvm.arm.mve.pred.v2i.v16i1(<16 x i1> %3)
  %5 = trunc i32 %4 to i16
  ret i16 %5
}

define arm_aapcs_vfpcc zeroext i16 @test_vctp16q(i32 %a) {
; CHECK-LABEL: test_vctp16q:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vctp.16 r0
; CHECK-NEXT:    vmrs r0, p0
; CHECK-NEXT:    bx lr
entry:
  %0 = call <8 x i1> @llvm.arm.mve.vctp16(i32 %a)
  %1 = call i32 @llvm.arm.mve.pred.v2i.v8i1(<8 x i1> %0)
  %2 = trunc i32 %1 to i16
  ret i16 %2
}

define arm_aapcs_vfpcc zeroext i16 @test_vctp16q_m(i32 %a, i16 zeroext %p) {
; CHECK-LABEL: test_vctp16q_m:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmsr p0, r1
; CHECK-NEXT:    vpst
; CHECK-NEXT:    vctpt.16 r0
; CHECK-NEXT:    vmrs r0, p0
; CHECK-NEXT:    bx lr
entry:
  %0 = zext i16 %p to i32
  %1 = call <8 x i1> @llvm.arm.mve.pred.i2v.v8i1(i32 %0)
  %2 = call <8 x i1> @llvm.arm.mve.vctp16(i32 %a)
  %3 = and <8 x i1> %1, %2
  %4 = call i32 @llvm.arm.mve.pred.v2i.v8i1(<8 x i1> %3)
  %5 = trunc i32 %4 to i16
  ret i16 %5
}

define arm_aapcs_vfpcc zeroext i16 @test_vctp32q(i32 %a) {
; CHECK-LABEL: test_vctp32q:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vctp.32 r0
; CHECK-NEXT:    vmrs r0, p0
; CHECK-NEXT:    bx lr
entry:
  %0 = call <4 x i1> @llvm.arm.mve.vctp32(i32 %a)
  %1 = call i32 @llvm.arm.mve.pred.v2i.v4i1(<4 x i1> %0)
  %2 = trunc i32 %1 to i16
  ret i16 %2
}

define arm_aapcs_vfpcc zeroext i16 @test_vctp32q_m(i32 %a, i16 zeroext %p) {
; CHECK-LABEL: test_vctp32q_m:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmsr p0, r1
; CHECK-NEXT:    vpst
; CHECK-NEXT:    vctpt.32 r0
; CHECK-NEXT:    vmrs r0, p0
; CHECK-NEXT:    bx lr
entry:
  %0 = zext i16 %p to i32
  %1 = call <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32 %0)
  %2 = call <4 x i1> @llvm.arm.mve.vctp32(i32 %a)
  %3 = and <4 x i1> %1, %2
  %4 = call i32 @llvm.arm.mve.pred.v2i.v4i1(<4 x i1> %3)
  %5 = trunc i32 %4 to i16
  ret i16 %5
}

define arm_aapcs_vfpcc zeroext i16 @test_vctp64q(i32 %a) {
; CHECK-LABEL: test_vctp64q:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vctp.64 r0
; CHECK-NEXT:    vmrs r0, p0
; CHECK-NEXT:    bx lr
entry:
  %0 = call <4 x i1> @llvm.arm.mve.vctp64(i32 %a)
  %1 = call i32 @llvm.arm.mve.pred.v2i.v4i1(<4 x i1> %0)
  %2 = trunc i32 %1 to i16
  ret i16 %2
}

define arm_aapcs_vfpcc zeroext i16 @test_vctp64q_m(i32 %a, i16 zeroext %p) {
; CHECK-LABEL: test_vctp64q_m:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmsr p0, r1
; CHECK-NEXT:    vpst
; CHECK-NEXT:    vctpt.64 r0
; CHECK-NEXT:    vmrs r0, p0
; CHECK-NEXT:    bx lr
entry:
  %0 = zext i16 %p to i32
  %1 = call <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32 %0)
  %2 = call <4 x i1> @llvm.arm.mve.vctp64(i32 %a)
  %3 = and <4 x i1> %1, %2
  %4 = call i32 @llvm.arm.mve.pred.v2i.v4i1(<4 x i1> %3)
  %5 = trunc i32 %4 to i16
  ret i16 %5
}

define arm_aapcs_vfpcc <16 x i8> @test_vpselq_i8(<16 x i8> %a, <16 x i8> %b, i16 zeroext %p) #2 {
; CHECK-LABEL: test_vpselq_i8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmsr p0, r0
; CHECK-NEXT:    vpsel q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %0 = zext i16 %p to i32
  %1 = call <16 x i1> @llvm.arm.mve.pred.i2v.v16i1(i32 %0)
  %2 = select <16 x i1> %1, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %2
}

define arm_aapcs_vfpcc <8 x i16> @test_vpselq_i16(<8 x i16> %a, <8 x i16> %b, i16 zeroext %p) #2 {
; CHECK-LABEL: test_vpselq_i16:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmsr p0, r0
; CHECK-NEXT:    vpsel q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %0 = zext i16 %p to i32
  %1 = call <8 x i1> @llvm.arm.mve.pred.i2v.v8i1(i32 %0)
  %2 = select <8 x i1> %1, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %2
}

define arm_aapcs_vfpcc <8 x half> @test_vpselq_f16(<8 x half> %a, <8 x half> %b, i16 zeroext %p) #2 {
; CHECK-LABEL: test_vpselq_f16:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmsr p0, r0
; CHECK-NEXT:    vpsel q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %0 = zext i16 %p to i32
  %1 = call <8 x i1> @llvm.arm.mve.pred.i2v.v8i1(i32 %0)
  %2 = select <8 x i1> %1, <8 x half> %a, <8 x half> %b
  ret <8 x half> %2
}

define arm_aapcs_vfpcc <4 x i32> @test_vpselq_i32(<4 x i32> %a, <4 x i32> %b, i16 zeroext %p) #2 {
; CHECK-LABEL: test_vpselq_i32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmsr p0, r0
; CHECK-NEXT:    vpsel q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %0 = zext i16 %p to i32
  %1 = call <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32 %0)
  %2 = select <4 x i1> %1, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %2
}

define arm_aapcs_vfpcc <4 x float> @test_vpselq_f32(<4 x float> %a, <4 x float> %b, i16 zeroext %p) #2 {
; CHECK-LABEL: test_vpselq_f32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmsr p0, r0
; CHECK-NEXT:    vpsel q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %0 = zext i16 %p to i32
  %1 = call <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32 %0)
  %2 = select <4 x i1> %1, <4 x float> %a, <4 x float> %b
  ret <4 x float> %2
}

define arm_aapcs_vfpcc <2 x i64> @test_vpselq_i64(<2 x i64> %a, <2 x i64> %b, i16 zeroext %p) #2 {
; CHECK-LABEL: test_vpselq_i64:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmsr p0, r0
; CHECK-NEXT:    vpsel q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %0 = zext i16 %p to i32
  %1 = call <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32 %0)
  %2 = bitcast <2 x i64> %a to <4 x i32>
  %3 = bitcast <2 x i64> %b to <4 x i32>
  %4 = select <4 x i1> %1, <4 x i32> %2, <4 x i32> %3
  %5 = bitcast <4 x i32> %4 to <2 x i64>
  ret <2 x i64> %5
}
