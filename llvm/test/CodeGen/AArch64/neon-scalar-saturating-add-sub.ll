; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <1 x i8> @llvm.arm.neon.vqaddu.v1i8(<1 x i8>, <1 x i8>)
declare <1 x i8> @llvm.arm.neon.vqadds.v1i8(<1 x i8>, <1 x i8>)

define <1 x i8> @test_uqadd_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_uqadd_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.arm.neon.vqaddu.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: uqadd {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}
  ret <1 x i8> %tmp1
}

define <1 x i8> @test_sqadd_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_sqadd_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.arm.neon.vqadds.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: sqadd {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}
  ret <1 x i8> %tmp1
}

declare <1 x i8> @llvm.arm.neon.vqsubu.v1i8(<1 x i8>, <1 x i8>)
declare <1 x i8> @llvm.arm.neon.vqsubs.v1i8(<1 x i8>, <1 x i8>)

define <1 x i8> @test_uqsub_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_uqsub_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.arm.neon.vqsubu.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: uqsub {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}
  ret <1 x i8> %tmp1
}

define <1 x i8> @test_sqsub_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_sqsub_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.arm.neon.vqsubs.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: sqsub {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}
  ret <1 x i8> %tmp1
}

declare <1 x i16> @llvm.arm.neon.vqaddu.v1i16(<1 x i16>, <1 x i16>)
declare <1 x i16> @llvm.arm.neon.vqadds.v1i16(<1 x i16>, <1 x i16>)

define <1 x i16> @test_uqadd_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_uqadd_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.arm.neon.vqaddu.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: uqadd {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}
  ret <1 x i16> %tmp1
}

define <1 x i16> @test_sqadd_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_sqadd_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.arm.neon.vqadds.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: sqadd {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}
  ret <1 x i16> %tmp1
}

declare <1 x i16> @llvm.arm.neon.vqsubu.v1i16(<1 x i16>, <1 x i16>)
declare <1 x i16> @llvm.arm.neon.vqsubs.v1i16(<1 x i16>, <1 x i16>)

define <1 x i16> @test_uqsub_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_uqsub_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.arm.neon.vqsubu.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: uqsub {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}
  ret <1 x i16> %tmp1
}

define <1 x i16> @test_sqsub_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_sqsub_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.arm.neon.vqsubs.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: sqsub {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}
  ret <1 x i16> %tmp1
}

declare <1 x i32> @llvm.arm.neon.vqaddu.v1i32(<1 x i32>, <1 x i32>)
declare <1 x i32> @llvm.arm.neon.vqadds.v1i32(<1 x i32>, <1 x i32>)

define <1 x i32> @test_uqadd_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_uqadd_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.arm.neon.vqaddu.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: uqadd {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}
  ret <1 x i32> %tmp1
}

define <1 x i32> @test_sqadd_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_sqadd_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.arm.neon.vqadds.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: sqadd {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}
  ret <1 x i32> %tmp1
}

declare <1 x i32> @llvm.arm.neon.vqsubu.v1i32(<1 x i32>, <1 x i32>)
declare <1 x i32> @llvm.arm.neon.vqsubs.v1i32(<1 x i32>, <1 x i32>)

define <1 x i32> @test_uqsub_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_uqsub_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.arm.neon.vqsubu.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: uqsub {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}
  ret <1 x i32> %tmp1
}


define <1 x i32> @test_sqsub_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_sqsub_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.arm.neon.vqsubs.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: sqsub {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}
  ret <1 x i32> %tmp1
}

declare <1 x i64> @llvm.arm.neon.vqaddu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.arm.neon.vqadds.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_uqadd_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uqadd_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqaddu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: uqadd {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sqadd_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sqadd_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqadds.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: sqadd {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

declare <1 x i64> @llvm.arm.neon.vqsubu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.arm.neon.vqsubs.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_uqsub_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uqsub_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqsubu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: uqsub {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sqsub_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sqsub_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqsubs.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: sqsub {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

define i8 @test_vuqaddb_s8(i8 %a, i8 %b) {
; CHECK: test_vuqaddb_s8
; CHECK: suqadd {{b[0-9]+}}, {{b[0-9]+}}
entry:
  %vuqadd.i = insertelement <1 x i8> undef, i8 %a, i32 0
  %vuqadd1.i = insertelement <1 x i8> undef, i8 %b, i32 0
  %vuqadd2.i = call <1 x i8> @llvm.aarch64.neon.vuqadd.v1i8(<1 x i8> %vuqadd.i, <1 x i8> %vuqadd1.i)
  %0 = extractelement <1 x i8> %vuqadd2.i, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.aarch64.neon.vsqadd.v1i8(<1 x i8>, <1 x i8>)

define i16 @test_vuqaddh_s16(i16 %a, i16 %b) {
; CHECK: test_vuqaddh_s16
; CHECK: suqadd {{h[0-9]+}}, {{h[0-9]+}}
entry:
  %vuqadd.i = insertelement <1 x i16> undef, i16 %a, i32 0
  %vuqadd1.i = insertelement <1 x i16> undef, i16 %b, i32 0
  %vuqadd2.i = call <1 x i16> @llvm.aarch64.neon.vuqadd.v1i16(<1 x i16> %vuqadd.i, <1 x i16> %vuqadd1.i)
  %0 = extractelement <1 x i16> %vuqadd2.i, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.aarch64.neon.vsqadd.v1i16(<1 x i16>, <1 x i16>)

define i32 @test_vuqadds_s32(i32 %a, i32 %b) {
; CHECK: test_vuqadds_s32
; CHECK: suqadd {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vuqadd.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %vuqadd1.i = insertelement <1 x i32> undef, i32 %b, i32 0
  %vuqadd2.i = call <1 x i32> @llvm.aarch64.neon.vuqadd.v1i32(<1 x i32> %vuqadd.i, <1 x i32> %vuqadd1.i)
  %0 = extractelement <1 x i32> %vuqadd2.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vsqadd.v1i32(<1 x i32>, <1 x i32>)

define i64 @test_vuqaddd_s64(i64 %a, i64 %b) {
; CHECK: test_vuqaddd_s64
; CHECK: suqadd {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vuqadd.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vuqadd1.i = insertelement <1 x i64> undef, i64 %b, i32 0
  %vuqadd2.i = call <1 x i64> @llvm.aarch64.neon.vuqadd.v1i64(<1 x i64> %vuqadd.i, <1 x i64> %vuqadd1.i)
  %0 = extractelement <1 x i64> %vuqadd2.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vsqadd.v1i64(<1 x i64>, <1 x i64>)

define i8 @test_vsqaddb_u8(i8 %a, i8 %b) {
; CHECK: test_vsqaddb_u8
; CHECK: usqadd {{b[0-9]+}}, {{b[0-9]+}}
entry:
  %vsqadd.i = insertelement <1 x i8> undef, i8 %a, i32 0
  %vsqadd1.i = insertelement <1 x i8> undef, i8 %b, i32 0
  %vsqadd2.i = call <1 x i8> @llvm.aarch64.neon.vsqadd.v1i8(<1 x i8> %vsqadd.i, <1 x i8> %vsqadd1.i)
  %0 = extractelement <1 x i8> %vsqadd2.i, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.aarch64.neon.vuqadd.v1i8(<1 x i8>, <1 x i8>)

define i16 @test_vsqaddh_u16(i16 %a, i16 %b) {
; CHECK: test_vsqaddh_u16
; CHECK: usqadd {{h[0-9]+}}, {{h[0-9]+}}
entry:
  %vsqadd.i = insertelement <1 x i16> undef, i16 %a, i32 0
  %vsqadd1.i = insertelement <1 x i16> undef, i16 %b, i32 0
  %vsqadd2.i = call <1 x i16> @llvm.aarch64.neon.vsqadd.v1i16(<1 x i16> %vsqadd.i, <1 x i16> %vsqadd1.i)
  %0 = extractelement <1 x i16> %vsqadd2.i, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.aarch64.neon.vuqadd.v1i16(<1 x i16>, <1 x i16>)

define i32 @test_vsqadds_u32(i32 %a, i32 %b) {
; CHECK: test_vsqadds_u32
; CHECK: usqadd {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vsqadd.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %vsqadd1.i = insertelement <1 x i32> undef, i32 %b, i32 0
  %vsqadd2.i = call <1 x i32> @llvm.aarch64.neon.vsqadd.v1i32(<1 x i32> %vsqadd.i, <1 x i32> %vsqadd1.i)
  %0 = extractelement <1 x i32> %vsqadd2.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vuqadd.v1i32(<1 x i32>, <1 x i32>)

define i64 @test_vsqaddd_u64(i64 %a, i64 %b) {
; CHECK: test_vsqaddd_u64
; CHECK: usqadd {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vsqadd.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vsqadd1.i = insertelement <1 x i64> undef, i64 %b, i32 0
  %vsqadd2.i = call <1 x i64> @llvm.aarch64.neon.vsqadd.v1i64(<1 x i64> %vsqadd.i, <1 x i64> %vsqadd1.i)
  %0 = extractelement <1 x i64> %vsqadd2.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vuqadd.v1i64(<1 x i64>, <1 x i64>)
