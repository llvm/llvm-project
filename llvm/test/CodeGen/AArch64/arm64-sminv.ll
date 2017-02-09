; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -asm-verbose=false | FileCheck %s

define signext i8 @test_vminv_s8(<8 x i8> %a1) {
; CHECK: test_vminv_s8
; CHECK: sminv.8b b[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.b w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vminv.i = tail call i32 @llvm.aarch64.neon.sminv.i32.v8i8(<8 x i8> %a1)
  %0 = trunc i32 %vminv.i to i8
  ret i8 %0
}

define signext i16 @test_vminv_s16(<4 x i16> %a1) {
; CHECK: test_vminv_s16
; CHECK: sminv.4h h[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.h w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vminv.i = tail call i32 @llvm.aarch64.neon.sminv.i32.v4i16(<4 x i16> %a1)
  %0 = trunc i32 %vminv.i to i16
  ret i16 %0
}

define i32 @test_vminv_s32(<2 x i32> %a1) {
; CHECK: test_vminv_s32
; 2 x i32 is not supported by the ISA, thus, this is a special case
; CHECK: sminp.2s v[[REGNUM:[0-9]+]], v0, v0
; CHECK-NEXT: fmov w0, s[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vminv.i = tail call i32 @llvm.aarch64.neon.sminv.i32.v2i32(<2 x i32> %a1)
  ret i32 %vminv.i
}

define signext i8 @test_vminvq_s8(<16 x i8> %a1) {
; CHECK: test_vminvq_s8
; CHECK: sminv.16b b[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.b w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vminv.i = tail call i32 @llvm.aarch64.neon.sminv.i32.v16i8(<16 x i8> %a1)
  %0 = trunc i32 %vminv.i to i8
  ret i8 %0
}

define signext i16 @test_vminvq_s16(<8 x i16> %a1) {
; CHECK: test_vminvq_s16
; CHECK: sminv.8h h[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.h w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vminv.i = tail call i32 @llvm.aarch64.neon.sminv.i32.v8i16(<8 x i16> %a1)
  %0 = trunc i32 %vminv.i to i16
  ret i16 %0
}

define i32 @test_vminvq_s32(<4 x i32> %a1) {
; CHECK: test_vminvq_s32
; CHECK: sminv.4s [[REGNUM:s[0-9]+]], v0
; CHECK-NEXT: fmov w0, [[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vminv.i = tail call i32 @llvm.aarch64.neon.sminv.i32.v4i32(<4 x i32> %a1)
  ret i32 %vminv.i
}

define <8 x i8> @test_vminv_s8_used_by_laneop(<8 x i8> %a1, <8 x i8> %a2) {
; CHECK-LABEL: test_vminv_s8_used_by_laneop:
; CHECK: sminv.8b b[[REGNUM:[0-9]+]], v1
; CHECK-NEXT: ins.b v0[3], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.sminv.i32.v8i8(<8 x i8> %a2)
  %1 = trunc i32 %0 to i8
  %2 = insertelement <8 x i8> %a1, i8 %1, i32 3
  ret <8 x i8> %2
}

define <4 x i16> @test_vminv_s16_used_by_laneop(<4 x i16> %a1, <4 x i16> %a2) {
; CHECK-LABEL: test_vminv_s16_used_by_laneop:
; CHECK: sminv.4h h[[REGNUM:[0-9]+]], v1
; CHECK-NEXT: ins.h v0[3], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.sminv.i32.v4i16(<4 x i16> %a2)
  %1 = trunc i32 %0 to i16
  %2 = insertelement <4 x i16> %a1, i16 %1, i32 3
  ret <4 x i16> %2
}

define <2 x i32> @test_vminv_s32_used_by_laneop(<2 x i32> %a1, <2 x i32> %a2) {
; CHECK-LABEL: test_vminv_s32_used_by_laneop:
; CHECK: sminp.2s v[[REGNUM:[0-9]+]], v1, v1
; CHECK-NEXT: ins.s v0[1], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.sminv.i32.v2i32(<2 x i32> %a2)
  %1 = insertelement <2 x i32> %a1, i32 %0, i32 1
  ret <2 x i32> %1
}

define <16 x i8> @test_vminvq_s8_used_by_laneop(<16 x i8> %a1, <16 x i8> %a2) {
; CHECK-LABEL: test_vminvq_s8_used_by_laneop:
; CHECK: sminv.16b b[[REGNUM:[0-9]+]], v1
; CHECK-NEXT: ins.b v0[3], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.sminv.i32.v16i8(<16 x i8> %a2)
  %1 = trunc i32 %0 to i8
  %2 = insertelement <16 x i8> %a1, i8 %1, i32 3
  ret <16 x i8> %2
}

define <8 x i16> @test_vminvq_s16_used_by_laneop(<8 x i16> %a1, <8 x i16> %a2) {
; CHECK-LABEL: test_vminvq_s16_used_by_laneop:
; CHECK: sminv.8h h[[REGNUM:[0-9]+]], v1
; CHECK-NEXT: ins.h v0[3], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.sminv.i32.v8i16(<8 x i16> %a2)
  %1 = trunc i32 %0 to i16
  %2 = insertelement <8 x i16> %a1, i16 %1, i32 3
  ret <8 x i16> %2
}

define <4 x i32> @test_vminvq_s32_used_by_laneop(<4 x i32> %a1, <4 x i32> %a2) {
; CHECK-LABEL: test_vminvq_s32_used_by_laneop:
; CHECK: sminv.4s s[[REGNUM:[0-9]+]], v1
; CHECK-NEXT: ins.s v0[3], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.sminv.i32.v4i32(<4 x i32> %a2)
  %1 = insertelement <4 x i32> %a1, i32 %0, i32 3
  ret <4 x i32> %1
}

declare i32 @llvm.aarch64.neon.sminv.i32.v4i32(<4 x i32>)
declare i32 @llvm.aarch64.neon.sminv.i32.v8i16(<8 x i16>)
declare i32 @llvm.aarch64.neon.sminv.i32.v16i8(<16 x i8>)
declare i32 @llvm.aarch64.neon.sminv.i32.v2i32(<2 x i32>)
declare i32 @llvm.aarch64.neon.sminv.i32.v4i16(<4 x i16>)
declare i32 @llvm.aarch64.neon.sminv.i32.v8i8(<8 x i8>)

