; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <8 x i8> @test_vshr_n_s8(<8 x i8> %a) {
; CHECK: test_vshr_n_s8
; CHECK: sshr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vshr_n = ashr <8 x i8> %a, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <8 x i8> %vshr_n
}

define <4 x i16> @test_vshr_n_s16(<4 x i16> %a) {
; CHECK: test_vshr_n_s16
; CHECK: sshr {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %vshr_n = ashr <4 x i16> %a, <i16 3, i16 3, i16 3, i16 3>
  ret <4 x i16> %vshr_n
}

define <2 x i32> @test_vshr_n_s32(<2 x i32> %a) {
; CHECK: test_vshr_n_s32
; CHECK: sshr {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %vshr_n = ashr <2 x i32> %a, <i32 3, i32 3>
  ret <2 x i32> %vshr_n
}

define <16 x i8> @test_vshrq_n_s8(<16 x i8> %a) {
; CHECK: test_vshrq_n_s8
; CHECK: sshr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vshr_n = ashr <16 x i8> %a, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <16 x i8> %vshr_n
}

define <8 x i16> @test_vshrq_n_s16(<8 x i16> %a) {
; CHECK: test_vshrq_n_s16
; CHECK: sshr {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %vshr_n = ashr <8 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <8 x i16> %vshr_n
}

define <4 x i32> @test_vshrq_n_s32(<4 x i32> %a) {
; CHECK: test_vshrq_n_s32
; CHECK: sshr {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %vshr_n = ashr <4 x i32> %a, <i32 3, i32 3, i32 3, i32 3>
  ret <4 x i32> %vshr_n
}

define <2 x i64> @test_vshrq_n_s64(<2 x i64> %a) {
; CHECK: test_vshrq_n_s64
; CHECK: sshr {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %vshr_n = ashr <2 x i64> %a, <i64 3, i64 3>
  ret <2 x i64> %vshr_n
}

define <8 x i8> @test_vshr_n_u8(<8 x i8> %a) {
; CHECK: test_vshr_n_u8
; CHECK: ushr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vshr_n = lshr <8 x i8> %a, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <8 x i8> %vshr_n
}

define <4 x i16> @test_vshr_n_u16(<4 x i16> %a) {
; CHECK: test_vshr_n_u16
; CHECK: ushr {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %vshr_n = lshr <4 x i16> %a, <i16 3, i16 3, i16 3, i16 3>
  ret <4 x i16> %vshr_n
}

define <2 x i32> @test_vshr_n_u32(<2 x i32> %a) {
; CHECK: test_vshr_n_u32
; CHECK: ushr {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %vshr_n = lshr <2 x i32> %a, <i32 3, i32 3>
  ret <2 x i32> %vshr_n
}

define <16 x i8> @test_vshrq_n_u8(<16 x i8> %a) {
; CHECK: test_vshrq_n_u8
; CHECK: ushr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vshr_n = lshr <16 x i8> %a, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <16 x i8> %vshr_n
}

define <8 x i16> @test_vshrq_n_u16(<8 x i16> %a) {
; CHECK: test_vshrq_n_u16
; CHECK: ushr {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %vshr_n = lshr <8 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <8 x i16> %vshr_n
}

define <4 x i32> @test_vshrq_n_u32(<4 x i32> %a) {
; CHECK: test_vshrq_n_u32
; CHECK: ushr {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %vshr_n = lshr <4 x i32> %a, <i32 3, i32 3, i32 3, i32 3>
  ret <4 x i32> %vshr_n
}

define <2 x i64> @test_vshrq_n_u64(<2 x i64> %a) {
; CHECK: test_vshrq_n_u64
; CHECK: ushr {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %vshr_n = lshr <2 x i64> %a, <i64 3, i64 3>
  ret <2 x i64> %vshr_n
}

define <8 x i8> @test_vsra_n_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vsra_n_s8
; CHECK: ssra {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vsra_n = ashr <8 x i8> %b, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  %1 = add <8 x i8> %vsra_n, %a
  ret <8 x i8> %1
}

define <4 x i16> @test_vsra_n_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vsra_n_s16
; CHECK: ssra {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %vsra_n = ashr <4 x i16> %b, <i16 3, i16 3, i16 3, i16 3>
  %1 = add <4 x i16> %vsra_n, %a
  ret <4 x i16> %1
}

define <2 x i32> @test_vsra_n_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vsra_n_s32
; CHECK: ssra {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %vsra_n = ashr <2 x i32> %b, <i32 3, i32 3>
  %1 = add <2 x i32> %vsra_n, %a
  ret <2 x i32> %1
}

define <16 x i8> @test_vsraq_n_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vsraq_n_s8
; CHECK: ssra {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vsra_n = ashr <16 x i8> %b, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  %1 = add <16 x i8> %vsra_n, %a
  ret <16 x i8> %1
}

define <8 x i16> @test_vsraq_n_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vsraq_n_s16
; CHECK: ssra {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %vsra_n = ashr <8 x i16> %b, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  %1 = add <8 x i16> %vsra_n, %a
  ret <8 x i16> %1
}

define <4 x i32> @test_vsraq_n_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vsraq_n_s32
; CHECK: ssra {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %vsra_n = ashr <4 x i32> %b, <i32 3, i32 3, i32 3, i32 3>
  %1 = add <4 x i32> %vsra_n, %a
  ret <4 x i32> %1
}

define <2 x i64> @test_vsraq_n_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vsraq_n_s64
; CHECK: ssra {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %vsra_n = ashr <2 x i64> %b, <i64 3, i64 3>
  %1 = add <2 x i64> %vsra_n, %a
  ret <2 x i64> %1
}

define <8 x i8> @test_vsra_n_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vsra_n_u8
; CHECK: usra {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vsra_n = lshr <8 x i8> %b, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  %1 = add <8 x i8> %vsra_n, %a
  ret <8 x i8> %1
}

define <4 x i16> @test_vsra_n_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vsra_n_u16
; CHECK: usra {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %vsra_n = lshr <4 x i16> %b, <i16 3, i16 3, i16 3, i16 3>
  %1 = add <4 x i16> %vsra_n, %a
  ret <4 x i16> %1
}

define <2 x i32> @test_vsra_n_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vsra_n_u32
; CHECK: usra {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %vsra_n = lshr <2 x i32> %b, <i32 3, i32 3>
  %1 = add <2 x i32> %vsra_n, %a
  ret <2 x i32> %1
}

define <16 x i8> @test_vsraq_n_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vsraq_n_u8
; CHECK: usra {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vsra_n = lshr <16 x i8> %b, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  %1 = add <16 x i8> %vsra_n, %a
  ret <16 x i8> %1
}

define <8 x i16> @test_vsraq_n_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vsraq_n_u16
; CHECK: usra {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %vsra_n = lshr <8 x i16> %b, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  %1 = add <8 x i16> %vsra_n, %a
  ret <8 x i16> %1
}

define <4 x i32> @test_vsraq_n_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vsraq_n_u32
; CHECK: usra {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %vsra_n = lshr <4 x i32> %b, <i32 3, i32 3, i32 3, i32 3>
  %1 = add <4 x i32> %vsra_n, %a
  ret <4 x i32> %1
}

define <2 x i64> @test_vsraq_n_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vsraq_n_u64
; CHECK: usra {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %vsra_n = lshr <2 x i64> %b, <i64 3, i64 3>
  %1 = add <2 x i64> %vsra_n, %a
  ret <2 x i64> %1
}

define <8 x i8> @test_vrshr_n_s8(<8 x i8> %a) {
; CHECK: test_vrshr_n_s8
; CHECK: srshr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vrshr_n = tail call <8 x i8> @llvm.aarch64.neon.vsrshr.v8i8(<8 x i8> %a, i32 3)
  ret <8 x i8> %vrshr_n
}


define <4 x i16> @test_vrshr_n_s16(<4 x i16> %a) {
; CHECK: test_vrshr_n_s16
; CHECK: srshr {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %vrshr_n = tail call <4 x i16> @llvm.aarch64.neon.vsrshr.v4i16(<4 x i16> %a, i32 3)
  ret <4 x i16> %vrshr_n
}


define <2 x i32> @test_vrshr_n_s32(<2 x i32> %a) {
; CHECK: test_vrshr_n_s32
; CHECK: srshr {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %vrshr_n = tail call <2 x i32> @llvm.aarch64.neon.vsrshr.v2i32(<2 x i32> %a, i32 3)
  ret <2 x i32> %vrshr_n
}


define <16 x i8> @test_vrshrq_n_s8(<16 x i8> %a) {
; CHECK: test_vrshrq_n_s8
; CHECK: srshr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vrshr_n = tail call <16 x i8> @llvm.aarch64.neon.vsrshr.v16i8(<16 x i8> %a, i32 3)
  ret <16 x i8> %vrshr_n
}


define <8 x i16> @test_vrshrq_n_s16(<8 x i16> %a) {
; CHECK: test_vrshrq_n_s16
; CHECK: srshr {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %vrshr_n = tail call <8 x i16> @llvm.aarch64.neon.vsrshr.v8i16(<8 x i16> %a, i32 3)
  ret <8 x i16> %vrshr_n
}


define <4 x i32> @test_vrshrq_n_s32(<4 x i32> %a) {
; CHECK: test_vrshrq_n_s32
; CHECK: srshr {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %vrshr_n = tail call <4 x i32> @llvm.aarch64.neon.vsrshr.v4i32(<4 x i32> %a, i32 3)
  ret <4 x i32> %vrshr_n
}


define <2 x i64> @test_vrshrq_n_s64(<2 x i64> %a) {
; CHECK: test_vrshrq_n_s64
; CHECK: srshr {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %vrshr_n = tail call <2 x i64> @llvm.aarch64.neon.vsrshr.v2i64(<2 x i64> %a, i32 3)
  ret <2 x i64> %vrshr_n
}


define <8 x i8> @test_vrshr_n_u8(<8 x i8> %a) {
; CHECK: test_vrshr_n_u8
; CHECK: urshr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vrshr_n = tail call <8 x i8> @llvm.aarch64.neon.vurshr.v8i8(<8 x i8> %a, i32 3)
  ret <8 x i8> %vrshr_n
}


define <4 x i16> @test_vrshr_n_u16(<4 x i16> %a) {
; CHECK: test_vrshr_n_u16
; CHECK: urshr {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %vrshr_n = tail call <4 x i16> @llvm.aarch64.neon.vurshr.v4i16(<4 x i16> %a, i32 3)
  ret <4 x i16> %vrshr_n
}


define <2 x i32> @test_vrshr_n_u32(<2 x i32> %a) {
; CHECK: test_vrshr_n_u32
; CHECK: urshr {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %vrshr_n = tail call <2 x i32> @llvm.aarch64.neon.vurshr.v2i32(<2 x i32> %a, i32 3)
  ret <2 x i32> %vrshr_n
}


define <16 x i8> @test_vrshrq_n_u8(<16 x i8> %a) {
; CHECK: test_vrshrq_n_u8
; CHECK: urshr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vrshr_n = tail call <16 x i8> @llvm.aarch64.neon.vurshr.v16i8(<16 x i8> %a, i32 3)
  ret <16 x i8> %vrshr_n
}


define <8 x i16> @test_vrshrq_n_u16(<8 x i16> %a) {
; CHECK: test_vrshrq_n_u16
; CHECK: urshr {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %vrshr_n = tail call <8 x i16> @llvm.aarch64.neon.vurshr.v8i16(<8 x i16> %a, i32 3)
  ret <8 x i16> %vrshr_n
}


define <4 x i32> @test_vrshrq_n_u32(<4 x i32> %a) {
; CHECK: test_vrshrq_n_u32
; CHECK: urshr {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %vrshr_n = tail call <4 x i32> @llvm.aarch64.neon.vurshr.v4i32(<4 x i32> %a, i32 3)
  ret <4 x i32> %vrshr_n
}


define <2 x i64> @test_vrshrq_n_u64(<2 x i64> %a) {
; CHECK: test_vrshrq_n_u64
; CHECK: urshr {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %vrshr_n = tail call <2 x i64> @llvm.aarch64.neon.vurshr.v2i64(<2 x i64> %a, i32 3)
  ret <2 x i64> %vrshr_n
}


define <8 x i8> @test_vrsra_n_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vrsra_n_s8
; CHECK: srsra {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %1 = tail call <8 x i8> @llvm.aarch64.neon.vsrshr.v8i8(<8 x i8> %b, i32 3)
  %vrsra_n = add <8 x i8> %1, %a
  ret <8 x i8> %vrsra_n
}

define <4 x i16> @test_vrsra_n_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vrsra_n_s16
; CHECK: srsra {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %1 = tail call <4 x i16> @llvm.aarch64.neon.vsrshr.v4i16(<4 x i16> %b, i32 3)
  %vrsra_n = add <4 x i16> %1, %a
  ret <4 x i16> %vrsra_n
}

define <2 x i32> @test_vrsra_n_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vrsra_n_s32
; CHECK: srsra {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %1 = tail call <2 x i32> @llvm.aarch64.neon.vsrshr.v2i32(<2 x i32> %b, i32 3)
  %vrsra_n = add <2 x i32> %1, %a
  ret <2 x i32> %vrsra_n
}

define <16 x i8> @test_vrsraq_n_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vrsraq_n_s8
; CHECK: srsra {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %1 = tail call <16 x i8> @llvm.aarch64.neon.vsrshr.v16i8(<16 x i8> %b, i32 3)
  %vrsra_n = add <16 x i8> %1, %a
  ret <16 x i8> %vrsra_n
}

define <8 x i16> @test_vrsraq_n_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vrsraq_n_s16
; CHECK: srsra {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %1 = tail call <8 x i16> @llvm.aarch64.neon.vsrshr.v8i16(<8 x i16> %b, i32 3)
  %vrsra_n = add <8 x i16> %1, %a
  ret <8 x i16> %vrsra_n
}

define <4 x i32> @test_vrsraq_n_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vrsraq_n_s32
; CHECK: srsra {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %1 = tail call <4 x i32> @llvm.aarch64.neon.vsrshr.v4i32(<4 x i32> %b, i32 3)
  %vrsra_n = add <4 x i32> %1, %a
  ret <4 x i32> %vrsra_n
}

define <2 x i64> @test_vrsraq_n_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vrsraq_n_s64
; CHECK: srsra {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %1 = tail call <2 x i64> @llvm.aarch64.neon.vsrshr.v2i64(<2 x i64> %b, i32 3)
  %vrsra_n = add <2 x i64> %1, %a
  ret <2 x i64> %vrsra_n
}

define <8 x i8> @test_vrsra_n_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vrsra_n_u8
; CHECK: ursra {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %1 = tail call <8 x i8> @llvm.aarch64.neon.vurshr.v8i8(<8 x i8> %b, i32 3)
  %vrsra_n = add <8 x i8> %1, %a
  ret <8 x i8> %vrsra_n
}

define <4 x i16> @test_vrsra_n_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vrsra_n_u16
; CHECK: ursra {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %1 = tail call <4 x i16> @llvm.aarch64.neon.vurshr.v4i16(<4 x i16> %b, i32 3)
  %vrsra_n = add <4 x i16> %1, %a
  ret <4 x i16> %vrsra_n
}

define <2 x i32> @test_vrsra_n_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vrsra_n_u32
; CHECK: ursra {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %1 = tail call <2 x i32> @llvm.aarch64.neon.vurshr.v2i32(<2 x i32> %b, i32 3)
  %vrsra_n = add <2 x i32> %1, %a
  ret <2 x i32> %vrsra_n
}

define <16 x i8> @test_vrsraq_n_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vrsraq_n_u8
; CHECK: ursra {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %1 = tail call <16 x i8> @llvm.aarch64.neon.vurshr.v16i8(<16 x i8> %b, i32 3)
  %vrsra_n = add <16 x i8> %1, %a
  ret <16 x i8> %vrsra_n
}

define <8 x i16> @test_vrsraq_n_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vrsraq_n_u16
; CHECK: ursra {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %1 = tail call <8 x i16> @llvm.aarch64.neon.vurshr.v8i16(<8 x i16> %b, i32 3)
  %vrsra_n = add <8 x i16> %1, %a
  ret <8 x i16> %vrsra_n
}

define <4 x i32> @test_vrsraq_n_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vrsraq_n_u32
; CHECK: ursra {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %1 = tail call <4 x i32> @llvm.aarch64.neon.vurshr.v4i32(<4 x i32> %b, i32 3)
  %vrsra_n = add <4 x i32> %1, %a
  ret <4 x i32> %vrsra_n
}

define <2 x i64> @test_vrsraq_n_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vrsraq_n_u64
; CHECK: ursra {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %1 = tail call <2 x i64> @llvm.aarch64.neon.vurshr.v2i64(<2 x i64> %b, i32 3)
  %vrsra_n = add <2 x i64> %1, %a
  ret <2 x i64> %vrsra_n
}

define <8 x i8> @test_vsri_n_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vsri_n_s8
; CHECK: sri {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vsri_n = tail call <8 x i8> @llvm.aarch64.neon.vsri.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
  ret <8 x i8> %vsri_n
}


define <4 x i16> @test_vsri_n_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vsri_n_s16
; CHECK: sri {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %vsri = tail call <4 x i16> @llvm.aarch64.neon.vsri.v4i16(<4 x i16> %a, <4 x i16> %b, i32 3)
  ret <4 x i16> %vsri
}


define <2 x i32> @test_vsri_n_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vsri_n_s32
; CHECK: sri {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %vsri = tail call <2 x i32> @llvm.aarch64.neon.vsri.v2i32(<2 x i32> %a, <2 x i32> %b, i32 3)
  ret <2 x i32> %vsri
}


define <16 x i8> @test_vsriq_n_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vsriq_n_s8
; CHECK: sri {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vsri_n = tail call <16 x i8> @llvm.aarch64.neon.vsri.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
  ret <16 x i8> %vsri_n
}


define <8 x i16> @test_vsriq_n_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vsriq_n_s16
; CHECK: sri {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %vsri = tail call <8 x i16> @llvm.aarch64.neon.vsri.v8i16(<8 x i16> %a, <8 x i16> %b, i32 3)
  ret <8 x i16> %vsri
}


define <4 x i32> @test_vsriq_n_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vsriq_n_s32
; CHECK: sri {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %vsri = tail call <4 x i32> @llvm.aarch64.neon.vsri.v4i32(<4 x i32> %a, <4 x i32> %b, i32 3)
  ret <4 x i32> %vsri
}


define <2 x i64> @test_vsriq_n_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vsriq_n_s64
; CHECK: sri {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %vsri = tail call <2 x i64> @llvm.aarch64.neon.vsri.v2i64(<2 x i64> %a, <2 x i64> %b, i32 3)
  ret <2 x i64> %vsri
}

define <8 x i8> @test_vsri_n_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vsri_n_p8
; CHECK: sri {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vsri_n = tail call <8 x i8> @llvm.aarch64.neon.vsri.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
  ret <8 x i8> %vsri_n
}

define <4 x i16> @test_vsri_n_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vsri_n_p16
; CHECK: sri {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #15
  %vsri = tail call <4 x i16> @llvm.aarch64.neon.vsri.v4i16(<4 x i16> %a, <4 x i16> %b, i32 15)
  ret <4 x i16> %vsri
}

define <16 x i8> @test_vsriq_n_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vsriq_n_p8
; CHECK: sri {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vsri_n = tail call <16 x i8> @llvm.aarch64.neon.vsri.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
  ret <16 x i8> %vsri_n
}

define <8 x i16> @test_vsriq_n_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vsriq_n_p16
; CHECK: sri {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #15
  %vsri = tail call <8 x i16> @llvm.aarch64.neon.vsri.v8i16(<8 x i16> %a, <8 x i16> %b, i32 15)
  ret <8 x i16> %vsri
}

define <8 x i8> @test_vsli_n_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vsli_n_s8
; CHECK: sli {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vsli_n = tail call <8 x i8> @llvm.aarch64.neon.vsli.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
  ret <8 x i8> %vsli_n
}

define <4 x i16> @test_vsli_n_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vsli_n_s16
; CHECK: sli {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %vsli = tail call <4 x i16> @llvm.aarch64.neon.vsli.v4i16(<4 x i16> %a, <4 x i16> %b, i32 3)
  ret <4 x i16> %vsli
}

define <2 x i32> @test_vsli_n_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK: test_vsli_n_s32
; CHECK: sli {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %vsli = tail call <2 x i32> @llvm.aarch64.neon.vsli.v2i32(<2 x i32> %a, <2 x i32> %b, i32 3)
  ret <2 x i32> %vsli
}

define <16 x i8> @test_vsliq_n_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vsliq_n_s8
; CHECK: sli {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vsli_n = tail call <16 x i8> @llvm.aarch64.neon.vsli.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
  ret <16 x i8> %vsli_n
}

define <8 x i16> @test_vsliq_n_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vsliq_n_s16
; CHECK: sli {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %vsli = tail call <8 x i16> @llvm.aarch64.neon.vsli.v8i16(<8 x i16> %a, <8 x i16> %b, i32 3)
  ret <8 x i16> %vsli
}

define <4 x i32> @test_vsliq_n_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: test_vsliq_n_s32
; CHECK: sli {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %vsli = tail call <4 x i32> @llvm.aarch64.neon.vsli.v4i32(<4 x i32> %a, <4 x i32> %b, i32 3)
  ret <4 x i32> %vsli
}

define <2 x i64> @test_vsliq_n_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: test_vsliq_n_s64
; CHECK: sli {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %vsli = tail call <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64> %a, <2 x i64> %b, i32 3)
  ret <2 x i64> %vsli
}

define <8 x i8> @test_vsli_n_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vsli_n_p8
; CHECK: sli {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vsli_n = tail call <8 x i8> @llvm.aarch64.neon.vsli.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
  ret <8 x i8> %vsli_n
}

define <4 x i16> @test_vsli_n_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK: test_vsli_n_p16
; CHECK: sli {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #15
  %vsli = tail call <4 x i16> @llvm.aarch64.neon.vsli.v4i16(<4 x i16> %a, <4 x i16> %b, i32 15)
  ret <4 x i16> %vsli
}

define <16 x i8> @test_vsliq_n_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vsliq_n_p8
; CHECK: sli {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vsli_n = tail call <16 x i8> @llvm.aarch64.neon.vsli.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
  ret <16 x i8> %vsli_n
}

define <8 x i16> @test_vsliq_n_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: test_vsliq_n_p16
; CHECK: sli {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #15
  %vsli = tail call <8 x i16> @llvm.aarch64.neon.vsli.v8i16(<8 x i16> %a, <8 x i16> %b, i32 15)
  ret <8 x i16> %vsli
}

define <8 x i8> @test_vqshl_n_s8(<8 x i8> %a) {
; CHECK: test_vqshl_n_s8
; CHECK: sqshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vqshl = tail call <8 x i8> @llvm.arm.neon.vqshifts.v8i8(<8 x i8> %a, <8 x i8> <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>)
  ret <8 x i8> %vqshl
}


define <4 x i16> @test_vqshl_n_s16(<4 x i16> %a) {
; CHECK: test_vqshl_n_s16
; CHECK: sqshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %vqshl = tail call <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16> %a, <4 x i16> <i16 3, i16 3, i16 3, i16 3>)
  ret <4 x i16> %vqshl
}


define <2 x i32> @test_vqshl_n_s32(<2 x i32> %a) {
; CHECK: test_vqshl_n_s32
; CHECK: sqshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %vqshl = tail call <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32> %a, <2 x i32> <i32 3, i32 3>)
  ret <2 x i32> %vqshl
}


define <16 x i8> @test_vqshlq_n_s8(<16 x i8> %a) {
; CHECK: test_vqshlq_n_s8
; CHECK: sqshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vqshl_n = tail call <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8> %a, <16 x i8> <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>)
  ret <16 x i8> %vqshl_n
}


define <8 x i16> @test_vqshlq_n_s16(<8 x i16> %a) {
; CHECK: test_vqshlq_n_s16
; CHECK: sqshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %vqshl = tail call <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16> %a, <8 x i16> <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>)
  ret <8 x i16> %vqshl
}


define <4 x i32> @test_vqshlq_n_s32(<4 x i32> %a) {
; CHECK: test_vqshlq_n_s32
; CHECK: sqshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %vqshl = tail call <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32> %a, <4 x i32> <i32 3, i32 3, i32 3, i32 3>)
  ret <4 x i32> %vqshl
}


define <2 x i64> @test_vqshlq_n_s64(<2 x i64> %a) {
; CHECK: test_vqshlq_n_s64
; CHECK: sqshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %vqshl = tail call <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64> %a, <2 x i64> <i64 3, i64 3>)
  ret <2 x i64> %vqshl
}


define <8 x i8> @test_vqshl_n_u8(<8 x i8> %a) {
; CHECK: test_vqshl_n_u8
; CHECK: uqshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vqshl_n = tail call <8 x i8> @llvm.arm.neon.vqshiftu.v8i8(<8 x i8> %a, <8 x i8> <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>)
  ret <8 x i8> %vqshl_n
}


define <4 x i16> @test_vqshl_n_u16(<4 x i16> %a) {
; CHECK: test_vqshl_n_u16
; CHECK: uqshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %vqshl = tail call <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16> %a, <4 x i16> <i16 3, i16 3, i16 3, i16 3>)
  ret <4 x i16> %vqshl
}


define <2 x i32> @test_vqshl_n_u32(<2 x i32> %a) {
; CHECK: test_vqshl_n_u32
; CHECK: uqshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %vqshl = tail call <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32> %a, <2 x i32> <i32 3, i32 3>)
  ret <2 x i32> %vqshl
}


define <16 x i8> @test_vqshlq_n_u8(<16 x i8> %a) {
; CHECK: test_vqshlq_n_u8
; CHECK: uqshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vqshl_n = tail call <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8> %a, <16 x i8> <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>)
  ret <16 x i8> %vqshl_n
}


define <8 x i16> @test_vqshlq_n_u16(<8 x i16> %a) {
; CHECK: test_vqshlq_n_u16
; CHECK: uqshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %vqshl = tail call <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16> %a, <8 x i16> <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>)
  ret <8 x i16> %vqshl
}


define <4 x i32> @test_vqshlq_n_u32(<4 x i32> %a) {
; CHECK: test_vqshlq_n_u32
; CHECK: uqshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %vqshl = tail call <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32> %a, <4 x i32> <i32 3, i32 3, i32 3, i32 3>)
  ret <4 x i32> %vqshl
}


define <2 x i64> @test_vqshlq_n_u64(<2 x i64> %a) {
; CHECK: test_vqshlq_n_u64
; CHECK: uqshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %vqshl = tail call <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64> %a, <2 x i64> <i64 3, i64 3>)
  ret <2 x i64> %vqshl
}

define <8 x i8> @test_vqshlu_n_s8(<8 x i8> %a) {
; CHECK: test_vqshlu_n_s8
; CHECK: sqshlu {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #3
  %vqshlu = tail call <8 x i8> @llvm.aarch64.neon.vsqshlu.v8i8(<8 x i8> %a, i32 3)
  ret <8 x i8> %vqshlu
}


define <4 x i16> @test_vqshlu_n_s16(<4 x i16> %a) {
; CHECK: test_vqshlu_n_s16
; CHECK: sqshlu {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #3
  %vqshlu = tail call <4 x i16> @llvm.aarch64.neon.vsqshlu.v4i16(<4 x i16> %a, i32 3)
  ret <4 x i16> %vqshlu
}


define <2 x i32> @test_vqshlu_n_s32(<2 x i32> %a) {
; CHECK: test_vqshlu_n_s32
; CHECK: sqshlu {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #3
  %vqshlu = tail call <2 x i32> @llvm.aarch64.neon.vsqshlu.v2i32(<2 x i32> %a, i32 3)
  ret <2 x i32> %vqshlu
}


define <16 x i8> @test_vqshluq_n_s8(<16 x i8> %a) {
; CHECK: test_vqshluq_n_s8
; CHECK: sqshlu {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #3
  %vqshlu = tail call <16 x i8> @llvm.aarch64.neon.vsqshlu.v16i8(<16 x i8> %a, i32 3)
  ret <16 x i8> %vqshlu
}


define <8 x i16> @test_vqshluq_n_s16(<8 x i16> %a) {
; CHECK: test_vqshluq_n_s16
; CHECK: sqshlu {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #3
  %vqshlu = tail call <8 x i16> @llvm.aarch64.neon.vsqshlu.v8i16(<8 x i16> %a, i32 3)
  ret <8 x i16> %vqshlu
}


define <4 x i32> @test_vqshluq_n_s32(<4 x i32> %a) {
; CHECK: test_vqshluq_n_s32
; CHECK: sqshlu {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #3
  %vqshlu = tail call <4 x i32> @llvm.aarch64.neon.vsqshlu.v4i32(<4 x i32> %a, i32 3)
  ret <4 x i32> %vqshlu
}


define <2 x i64> @test_vqshluq_n_s64(<2 x i64> %a) {
; CHECK: test_vqshluq_n_s64
; CHECK: sqshlu {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #3
  %vqshlu = tail call <2 x i64> @llvm.aarch64.neon.vsqshlu.v2i64(<2 x i64> %a, i32 3)
  ret <2 x i64> %vqshlu
}


define <8 x i8> @test_vshrn_n_s16(<8 x i16> %a) {
; CHECK: test_vshrn_n_s16
; CHECK: shrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
  %1 = ashr <8 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  %vshrn_n = trunc <8 x i16> %1 to <8 x i8>
  ret <8 x i8> %vshrn_n
}

define <4 x i16> @test_vshrn_n_s32(<4 x i32> %a) {
; CHECK: test_vshrn_n_s32
; CHECK: shrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
  %1 = ashr <4 x i32> %a, <i32 9, i32 9, i32 9, i32 9>
  %vshrn_n = trunc <4 x i32> %1 to <4 x i16>
  ret <4 x i16> %vshrn_n
}

define <2 x i32> @test_vshrn_n_s64(<2 x i64> %a) {
; CHECK: test_vshrn_n_s64
; CHECK: shrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
  %1 = ashr <2 x i64> %a, <i64 19, i64 19>
  %vshrn_n = trunc <2 x i64> %1 to <2 x i32>
  ret <2 x i32> %vshrn_n
}

define <8 x i8> @test_vshrn_n_u16(<8 x i16> %a) {
; CHECK: test_vshrn_n_u16
; CHECK: shrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
  %1 = lshr <8 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  %vshrn_n = trunc <8 x i16> %1 to <8 x i8>
  ret <8 x i8> %vshrn_n
}

define <4 x i16> @test_vshrn_n_u32(<4 x i32> %a) {
; CHECK: test_vshrn_n_u32
; CHECK: shrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
  %1 = lshr <4 x i32> %a, <i32 9, i32 9, i32 9, i32 9>
  %vshrn_n = trunc <4 x i32> %1 to <4 x i16>
  ret <4 x i16> %vshrn_n
}

define <2 x i32> @test_vshrn_n_u64(<2 x i64> %a) {
; CHECK: test_vshrn_n_u64
; CHECK: shrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
  %1 = lshr <2 x i64> %a, <i64 19, i64 19>
  %vshrn_n = trunc <2 x i64> %1 to <2 x i32>
  ret <2 x i32> %vshrn_n
}

define <16 x i8> @test_vshrn_high_n_s16(<8 x i8> %a, <8 x i16> %b) {
; CHECK: test_vshrn_high_n_s16
; CHECK: shrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
  %1 = ashr <8 x i16> %b, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  %vshrn_n = trunc <8 x i16> %1 to <8 x i8>
  %2 = bitcast <8 x i8> %a to <1 x i64>
  %3 = bitcast <8 x i8> %vshrn_n to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %2, <1 x i64> %3, <2 x i32> <i32 0, i32 1>
  %4 = bitcast <2 x i64> %shuffle.i to <16 x i8>
  ret <16 x i8> %4
}

define <8 x i16> @test_vshrn_high_n_s32(<4 x i16> %a, <4 x i32> %b) {
; CHECK: test_vshrn_high_n_s32
; CHECK: shrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
  %1 = ashr <4 x i32> %b, <i32 9, i32 9, i32 9, i32 9>
  %vshrn_n = trunc <4 x i32> %1 to <4 x i16>
  %2 = bitcast <4 x i16> %a to <1 x i64>
  %3 = bitcast <4 x i16> %vshrn_n to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %2, <1 x i64> %3, <2 x i32> <i32 0, i32 1>
  %4 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %4
}

define <4 x i32> @test_vshrn_high_n_s64(<2 x i32> %a, <2 x i64> %b) {
; CHECK: test_vshrn_high_n_s64
; CHECK: shrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
  %1 = bitcast <2 x i32> %a to <1 x i64>
  %2 = ashr <2 x i64> %b, <i64 19, i64 19>
  %vshrn_n = trunc <2 x i64> %2 to <2 x i32>
  %3 = bitcast <2 x i32> %vshrn_n to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %3, <2 x i32> <i32 0, i32 1>
  %4 = bitcast <2 x i64> %shuffle.i to <4 x i32>
  ret <4 x i32> %4
}

define <16 x i8> @test_vshrn_high_n_u16(<8 x i8> %a, <8 x i16> %b) {
; CHECK: test_vshrn_high_n_u16
; CHECK: shrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
  %1 = lshr <8 x i16> %b, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  %vshrn_n = trunc <8 x i16> %1 to <8 x i8>
  %2 = bitcast <8 x i8> %a to <1 x i64>
  %3 = bitcast <8 x i8> %vshrn_n to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %2, <1 x i64> %3, <2 x i32> <i32 0, i32 1>
  %4 = bitcast <2 x i64> %shuffle.i to <16 x i8>
  ret <16 x i8> %4
}

define <8 x i16> @test_vshrn_high_n_u32(<4 x i16> %a, <4 x i32> %b) {
; CHECK: test_vshrn_high_n_u32
; CHECK: shrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
  %1 = lshr <4 x i32> %b, <i32 9, i32 9, i32 9, i32 9>
  %vshrn_n = trunc <4 x i32> %1 to <4 x i16>
  %2 = bitcast <4 x i16> %a to <1 x i64>
  %3 = bitcast <4 x i16> %vshrn_n to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %2, <1 x i64> %3, <2 x i32> <i32 0, i32 1>
  %4 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %4
}

define <4 x i32> @test_vshrn_high_n_u64(<2 x i32> %a, <2 x i64> %b) {
; CHECK: test_vshrn_high_n_u64
; CHECK: shrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
  %1 = bitcast <2 x i32> %a to <1 x i64>
  %2 = lshr <2 x i64> %b, <i64 19, i64 19>
  %vshrn_n = trunc <2 x i64> %2 to <2 x i32>
  %3 = bitcast <2 x i32> %vshrn_n to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %3, <2 x i32> <i32 0, i32 1>
  %4 = bitcast <2 x i64> %shuffle.i to <4 x i32>
  ret <4 x i32> %4
}

define <8 x i8> @test_vqshrun_n_s16(<8 x i16> %a) {
; CHECK: test_vqshrun_n_s16
; CHECK: sqshrun {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
  %vqshrun = tail call <8 x i8> @llvm.aarch64.neon.vsqshrun.v8i8(<8 x i16> %a, i32 3)
  ret <8 x i8> %vqshrun
}


define <4 x i16> @test_vqshrun_n_s32(<4 x i32> %a) {
; CHECK: test_vqshrun_n_s32
; CHECK: sqshrun {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
  %vqshrun = tail call <4 x i16> @llvm.aarch64.neon.vsqshrun.v4i16(<4 x i32> %a, i32 9)
  ret <4 x i16> %vqshrun
}

define <2 x i32> @test_vqshrun_n_s64(<2 x i64> %a) {
; CHECK: test_vqshrun_n_s64
; CHECK: sqshrun {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
  %vqshrun = tail call <2 x i32> @llvm.aarch64.neon.vsqshrun.v2i32(<2 x i64> %a, i32 19)
  ret <2 x i32> %vqshrun
}

define <16 x i8> @test_vqshrun_high_n_s16(<8 x i8> %a, <8 x i16> %b) {
; CHECK: test_vqshrun_high_n_s16
; CHECK: sqshrun2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
  %vqshrun = tail call <8 x i8> @llvm.aarch64.neon.vsqshrun.v8i8(<8 x i16> %b, i32 3)
  %1 = bitcast <8 x i8> %a to <1 x i64>
  %2 = bitcast <8 x i8> %vqshrun to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <16 x i8>
  ret <16 x i8> %3
}

define <8 x i16> @test_vqshrun_high_n_s32(<4 x i16> %a, <4 x i32> %b) {
; CHECK: test_vqshrun_high_n_s32
; CHECK: sqshrun2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
  %vqshrun = tail call <4 x i16> @llvm.aarch64.neon.vsqshrun.v4i16(<4 x i32> %b, i32 9)
  %1 = bitcast <4 x i16> %a to <1 x i64>
  %2 = bitcast <4 x i16> %vqshrun to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <4 x i32> @test_vqshrun_high_n_s64(<2 x i32> %a, <2 x i64> %b) {
; CHECK: test_vqshrun_high_n_s64
; CHECK: sqshrun2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
  %1 = bitcast <2 x i32> %a to <1 x i64>
  %vqshrun = tail call <2 x i32> @llvm.aarch64.neon.vsqshrun.v2i32(<2 x i64> %b, i32 19)
  %2 = bitcast <2 x i32> %vqshrun to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <4 x i32>
  ret <4 x i32> %3
}

define <8 x i8> @test_vrshrn_n_s16(<8 x i16> %a) {
; CHECK: test_vrshrn_n_s16
; CHECK: rshrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
  %vrshrn = tail call <8 x i8> @llvm.aarch64.neon.vrshrn.v8i8(<8 x i16> %a, i32 3)
  ret <8 x i8> %vrshrn
}


define <4 x i16> @test_vrshrn_n_s32(<4 x i32> %a) {
; CHECK: test_vrshrn_n_s32
; CHECK: rshrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
  %vrshrn = tail call <4 x i16> @llvm.aarch64.neon.vrshrn.v4i16(<4 x i32> %a, i32 9)
  ret <4 x i16> %vrshrn
}


define <2 x i32> @test_vrshrn_n_s64(<2 x i64> %a) {
; CHECK: test_vrshrn_n_s64
; CHECK: rshrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
  %vrshrn = tail call <2 x i32> @llvm.aarch64.neon.vrshrn.v2i32(<2 x i64> %a, i32 19)
  ret <2 x i32> %vrshrn
}

define <16 x i8> @test_vrshrn_high_n_s16(<8 x i8> %a, <8 x i16> %b) {
; CHECK: test_vrshrn_high_n_s16
; CHECK: rshrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
  %vrshrn = tail call <8 x i8> @llvm.aarch64.neon.vrshrn.v8i8(<8 x i16> %b, i32 3)
  %1 = bitcast <8 x i8> %a to <1 x i64>
  %2 = bitcast <8 x i8> %vrshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <16 x i8>
  ret <16 x i8> %3
}

define <8 x i16> @test_vrshrn_high_n_s32(<4 x i16> %a, <4 x i32> %b) {
; CHECK: test_vrshrn_high_n_s32
; CHECK: rshrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
  %vrshrn = tail call <4 x i16> @llvm.aarch64.neon.vrshrn.v4i16(<4 x i32> %b, i32 9)
  %1 = bitcast <4 x i16> %a to <1 x i64>
  %2 = bitcast <4 x i16> %vrshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <4 x i32> @test_vrshrn_high_n_s64(<2 x i32> %a, <2 x i64> %b) {
; CHECK: test_vrshrn_high_n_s64
; CHECK: rshrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
  %1 = bitcast <2 x i32> %a to <1 x i64>
  %vrshrn = tail call <2 x i32> @llvm.aarch64.neon.vrshrn.v2i32(<2 x i64> %b, i32 19)
  %2 = bitcast <2 x i32> %vrshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <4 x i32>
  ret <4 x i32> %3
}

define <8 x i8> @test_vqrshrun_n_s16(<8 x i16> %a) {
; CHECK: test_vqrshrun_n_s16
; CHECK: sqrshrun {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
  %vqrshrun = tail call <8 x i8> @llvm.aarch64.neon.vsqrshrun.v8i8(<8 x i16> %a, i32 3)
  ret <8 x i8> %vqrshrun
}

define <4 x i16> @test_vqrshrun_n_s32(<4 x i32> %a) {
; CHECK: test_vqrshrun_n_s32
; CHECK: sqrshrun {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
  %vqrshrun = tail call <4 x i16> @llvm.aarch64.neon.vsqrshrun.v4i16(<4 x i32> %a, i32 9)
  ret <4 x i16> %vqrshrun
}

define <2 x i32> @test_vqrshrun_n_s64(<2 x i64> %a) {
; CHECK: test_vqrshrun_n_s64
; CHECK: sqrshrun {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
  %vqrshrun = tail call <2 x i32> @llvm.aarch64.neon.vsqrshrun.v2i32(<2 x i64> %a, i32 19)
  ret <2 x i32> %vqrshrun
}

define <16 x i8> @test_vqrshrun_high_n_s16(<8 x i8> %a, <8 x i16> %b) {
; CHECK: test_vqrshrun_high_n_s16
; CHECK: sqrshrun2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
  %vqrshrun = tail call <8 x i8> @llvm.aarch64.neon.vsqrshrun.v8i8(<8 x i16> %b, i32 3)
  %1 = bitcast <8 x i8> %a to <1 x i64>
  %2 = bitcast <8 x i8> %vqrshrun to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <16 x i8>
  ret <16 x i8> %3
}

define <8 x i16> @test_vqrshrun_high_n_s32(<4 x i16> %a, <4 x i32> %b) {
; CHECK: test_vqrshrun_high_n_s32
; CHECK: sqrshrun2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
  %vqrshrun = tail call <4 x i16> @llvm.aarch64.neon.vsqrshrun.v4i16(<4 x i32> %b, i32 9)
  %1 = bitcast <4 x i16> %a to <1 x i64>
  %2 = bitcast <4 x i16> %vqrshrun to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <4 x i32> @test_vqrshrun_high_n_s64(<2 x i32> %a, <2 x i64> %b) {
; CHECK: test_vqrshrun_high_n_s64
; CHECK: sqrshrun2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
  %1 = bitcast <2 x i32> %a to <1 x i64>
  %vqrshrun = tail call <2 x i32> @llvm.aarch64.neon.vsqrshrun.v2i32(<2 x i64> %b, i32 19)
  %2 = bitcast <2 x i32> %vqrshrun to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <4 x i32>
  ret <4 x i32> %3
}

define <8 x i8> @test_vqshrn_n_s16(<8 x i16> %a) {
; CHECK: test_vqshrn_n_s16
; CHECK: sqshrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
  %vqshrn = tail call <8 x i8> @llvm.aarch64.neon.vsqshrn.v8i8(<8 x i16> %a, i32 3)
  ret <8 x i8> %vqshrn
}


define <4 x i16> @test_vqshrn_n_s32(<4 x i32> %a) {
; CHECK: test_vqshrn_n_s32
; CHECK: sqshrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
  %vqshrn = tail call <4 x i16> @llvm.aarch64.neon.vsqshrn.v4i16(<4 x i32> %a, i32 9)
  ret <4 x i16> %vqshrn
}


define <2 x i32> @test_vqshrn_n_s64(<2 x i64> %a) {
; CHECK: test_vqshrn_n_s64
; CHECK: sqshrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
  %vqshrn = tail call <2 x i32> @llvm.aarch64.neon.vsqshrn.v2i32(<2 x i64> %a, i32 19)
  ret <2 x i32> %vqshrn
}


define <8 x i8> @test_vqshrn_n_u16(<8 x i16> %a) {
; CHECK: test_vqshrn_n_u16
; CHECK: uqshrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
  %vqshrn = tail call <8 x i8> @llvm.aarch64.neon.vuqshrn.v8i8(<8 x i16> %a, i32 3)
  ret <8 x i8> %vqshrn
}


define <4 x i16> @test_vqshrn_n_u32(<4 x i32> %a) {
; CHECK: test_vqshrn_n_u32
; CHECK: uqshrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
  %vqshrn = tail call <4 x i16> @llvm.aarch64.neon.vuqshrn.v4i16(<4 x i32> %a, i32 9)
  ret <4 x i16> %vqshrn
}


define <2 x i32> @test_vqshrn_n_u64(<2 x i64> %a) {
; CHECK: test_vqshrn_n_u64
; CHECK: uqshrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
  %vqshrn = tail call <2 x i32> @llvm.aarch64.neon.vuqshrn.v2i32(<2 x i64> %a, i32 19)
  ret <2 x i32> %vqshrn
}


define <16 x i8> @test_vqshrn_high_n_s16(<8 x i8> %a, <8 x i16> %b) {
; CHECK: test_vqshrn_high_n_s16
; CHECK: sqshrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
  %vqshrn = tail call <8 x i8> @llvm.aarch64.neon.vsqshrn.v8i8(<8 x i16> %b, i32 3)
  %1 = bitcast <8 x i8> %a to <1 x i64>
  %2 = bitcast <8 x i8> %vqshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <16 x i8>
  ret <16 x i8> %3
}

define <8 x i16> @test_vqshrn_high_n_s32(<4 x i16> %a, <4 x i32> %b) {
; CHECK: test_vqshrn_high_n_s32
; CHECK: sqshrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
  %vqshrn = tail call <4 x i16> @llvm.aarch64.neon.vsqshrn.v4i16(<4 x i32> %b, i32 9)
  %1 = bitcast <4 x i16> %a to <1 x i64>
  %2 = bitcast <4 x i16> %vqshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <4 x i32> @test_vqshrn_high_n_s64(<2 x i32> %a, <2 x i64> %b) {
; CHECK: test_vqshrn_high_n_s64
; CHECK: sqshrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
  %1 = bitcast <2 x i32> %a to <1 x i64>
  %vqshrn = tail call <2 x i32> @llvm.aarch64.neon.vsqshrn.v2i32(<2 x i64> %b, i32 19)
  %2 = bitcast <2 x i32> %vqshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <4 x i32>
  ret <4 x i32> %3
}

define <16 x i8> @test_vqshrn_high_n_u16(<8 x i8> %a, <8 x i16> %b) {
; CHECK: test_vqshrn_high_n_u16
; CHECK: uqshrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
  %vqshrn = tail call <8 x i8> @llvm.aarch64.neon.vuqshrn.v8i8(<8 x i16> %b, i32 3)
  %1 = bitcast <8 x i8> %a to <1 x i64>
  %2 = bitcast <8 x i8> %vqshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <16 x i8>
  ret <16 x i8> %3
}

define <8 x i16> @test_vqshrn_high_n_u32(<4 x i16> %a, <4 x i32> %b) {
; CHECK: test_vqshrn_high_n_u32
; CHECK: uqshrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
  %vqshrn = tail call <4 x i16> @llvm.aarch64.neon.vuqshrn.v4i16(<4 x i32> %b, i32 9)
  %1 = bitcast <4 x i16> %a to <1 x i64>
  %2 = bitcast <4 x i16> %vqshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <4 x i32> @test_vqshrn_high_n_u64(<2 x i32> %a, <2 x i64> %b) {
; CHECK: test_vqshrn_high_n_u64
; CHECK: uqshrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
  %1 = bitcast <2 x i32> %a to <1 x i64>
  %vqshrn = tail call <2 x i32> @llvm.aarch64.neon.vuqshrn.v2i32(<2 x i64> %b, i32 19)
  %2 = bitcast <2 x i32> %vqshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <4 x i32>
  ret <4 x i32> %3
}

define <8 x i8> @test_vqrshrn_n_s16(<8 x i16> %a) {
; CHECK: test_vqrshrn_n_s16
; CHECK: sqrshrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
  %vqrshrn = tail call <8 x i8> @llvm.aarch64.neon.vsqrshrn.v8i8(<8 x i16> %a, i32 3)
  ret <8 x i8> %vqrshrn
}


define <4 x i16> @test_vqrshrn_n_s32(<4 x i32> %a) {
; CHECK: test_vqrshrn_n_s32
; CHECK: sqrshrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
  %vqrshrn = tail call <4 x i16> @llvm.aarch64.neon.vsqrshrn.v4i16(<4 x i32> %a, i32 9)
  ret <4 x i16> %vqrshrn
}


define <2 x i32> @test_vqrshrn_n_s64(<2 x i64> %a) {
; CHECK: test_vqrshrn_n_s64
; CHECK: sqrshrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
  %vqrshrn = tail call <2 x i32> @llvm.aarch64.neon.vsqrshrn.v2i32(<2 x i64> %a, i32 19)
  ret <2 x i32> %vqrshrn
}


define <8 x i8> @test_vqrshrn_n_u16(<8 x i16> %a) {
; CHECK: test_vqrshrn_n_u16
; CHECK: uqrshrn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, #3
  %vqrshrn = tail call <8 x i8> @llvm.aarch64.neon.vuqrshrn.v8i8(<8 x i16> %a, i32 3)
  ret <8 x i8> %vqrshrn
}


define <4 x i16> @test_vqrshrn_n_u32(<4 x i32> %a) {
; CHECK: test_vqrshrn_n_u32
; CHECK: uqrshrn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, #9
  %vqrshrn = tail call <4 x i16> @llvm.aarch64.neon.vuqrshrn.v4i16(<4 x i32> %a, i32 9)
  ret <4 x i16> %vqrshrn
}


define <2 x i32> @test_vqrshrn_n_u64(<2 x i64> %a) {
; CHECK: test_vqrshrn_n_u64
; CHECK: uqrshrn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, #19
  %vqrshrn = tail call <2 x i32> @llvm.aarch64.neon.vuqrshrn.v2i32(<2 x i64> %a, i32 19)
  ret <2 x i32> %vqrshrn
}


define <16 x i8> @test_vqrshrn_high_n_s16(<8 x i8> %a, <8 x i16> %b) {
; CHECK: test_vqrshrn_high_n_s16
; CHECK: sqrshrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
  %vqrshrn = tail call <8 x i8> @llvm.aarch64.neon.vsqrshrn.v8i8(<8 x i16> %b, i32 3)
  %1 = bitcast <8 x i8> %a to <1 x i64>
  %2 = bitcast <8 x i8> %vqrshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <16 x i8>
  ret <16 x i8> %3
}

define <8 x i16> @test_vqrshrn_high_n_s32(<4 x i16> %a, <4 x i32> %b) {
; CHECK: test_vqrshrn_high_n_s32
; CHECK: sqrshrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
  %vqrshrn = tail call <4 x i16> @llvm.aarch64.neon.vsqrshrn.v4i16(<4 x i32> %b, i32 9)
  %1 = bitcast <4 x i16> %a to <1 x i64>
  %2 = bitcast <4 x i16> %vqrshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <4 x i32> @test_vqrshrn_high_n_s64(<2 x i32> %a, <2 x i64> %b) {
; CHECK: test_vqrshrn_high_n_s64
; CHECK: sqrshrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
  %1 = bitcast <2 x i32> %a to <1 x i64>
  %vqrshrn = tail call <2 x i32> @llvm.aarch64.neon.vsqrshrn.v2i32(<2 x i64> %b, i32 19)
  %2 = bitcast <2 x i32> %vqrshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <4 x i32>
  ret <4 x i32> %3
}

define <16 x i8> @test_vqrshrn_high_n_u16(<8 x i8> %a, <8 x i16> %b) {
; CHECK: test_vqrshrn_high_n_u16
; CHECK: uqrshrn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, #3
  %vqrshrn = tail call <8 x i8> @llvm.aarch64.neon.vuqrshrn.v8i8(<8 x i16> %b, i32 3)
  %1 = bitcast <8 x i8> %a to <1 x i64>
  %2 = bitcast <8 x i8> %vqrshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <16 x i8>
  ret <16 x i8> %3
}

define <8 x i16> @test_vqrshrn_high_n_u32(<4 x i16> %a, <4 x i32> %b) {
; CHECK: test_vqrshrn_high_n_u32
; CHECK: uqrshrn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, #9
  %vqrshrn = tail call <4 x i16> @llvm.aarch64.neon.vuqrshrn.v4i16(<4 x i32> %b, i32 9)
  %1 = bitcast <4 x i16> %a to <1 x i64>
  %2 = bitcast <4 x i16> %vqrshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <8 x i16>
  ret <8 x i16> %3
}

define <4 x i32> @test_vqrshrn_high_n_u64(<2 x i32> %a, <2 x i64> %b) {
; CHECK: test_vqrshrn_high_n_u64
; CHECK: uqrshrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #19
  %1 = bitcast <2 x i32> %a to <1 x i64>
  %vqrshrn = tail call <2 x i32> @llvm.aarch64.neon.vuqrshrn.v2i32(<2 x i64> %b, i32 19)
  %2 = bitcast <2 x i32> %vqrshrn to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %2, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %shuffle.i to <4 x i32>
  ret <4 x i32> %3
}

define <2 x float> @test_vcvt_n_f32_s32(<2 x i32> %a) {
; CHECK: test_vcvt_n_f32_s32
; CHECK: scvtf {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #31
  %vcvt = tail call <2 x float> @llvm.arm.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32> %a, i32 31)
  ret <2 x float> %vcvt
}

define <4 x float> @test_vcvtq_n_f32_s32(<4 x i32> %a) {
; CHECK: test_vcvtq_n_f32_s32
; CHECK: scvtf {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #31
  %vcvt = tail call <4 x float> @llvm.arm.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32> %a, i32 31)
  ret <4 x float> %vcvt
}

define <2 x double> @test_vcvtq_n_f64_s64(<2 x i64> %a) {
; CHECK: test_vcvtq_n_f64_s64
; CHECK: scvtf {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #50
  %vcvt = tail call <2 x double> @llvm.arm.neon.vcvtfxs2fp.v2f64.v2i64(<2 x i64> %a, i32 50)
  ret <2 x double> %vcvt
}

define <2 x float> @test_vcvt_n_f32_u32(<2 x i32> %a) {
; CHECK: test_vcvt_n_f32_u32
; CHECK: ucvtf {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #31
  %vcvt = tail call <2 x float> @llvm.arm.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32> %a, i32 31)
  ret <2 x float> %vcvt
}

define <4 x float> @test_vcvtq_n_f32_u32(<4 x i32> %a) {
; CHECK: test_vcvtq_n_f32_u32
; CHECK: ucvtf {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #31
  %vcvt = tail call <4 x float> @llvm.arm.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32> %a, i32 31)
  ret <4 x float> %vcvt
}

define <2 x double> @test_vcvtq_n_f64_u64(<2 x i64> %a) {
; CHECK: test_vcvtq_n_f64_u64
; CHECK: ucvtf {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #50
  %vcvt = tail call <2 x double> @llvm.arm.neon.vcvtfxu2fp.v2f64.v2i64(<2 x i64> %a, i32 50)
  ret <2 x double> %vcvt
}

define <2 x i32> @test_vcvt_n_s32_f32(<2 x float> %a) {
; CHECK: test_vcvt_n_s32_f32
; CHECK: fcvtzs {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #31
  %vcvt = tail call <2 x i32> @llvm.arm.neon.vcvtfp2fxs.v2i32.v2f32(<2 x float> %a, i32 31)
  ret <2 x i32> %vcvt
}

define <4 x i32> @test_vcvtq_n_s32_f32(<4 x float> %a) {
; CHECK: test_vcvtq_n_s32_f32
; CHECK: fcvtzs {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #31
  %vcvt = tail call <4 x i32> @llvm.arm.neon.vcvtfp2fxs.v4i32.v4f32(<4 x float> %a, i32 31)
  ret <4 x i32> %vcvt
}

define <2 x i64> @test_vcvtq_n_s64_f64(<2 x double> %a) {
; CHECK: test_vcvtq_n_s64_f64
; CHECK: fcvtzs {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #50
  %vcvt = tail call <2 x i64> @llvm.arm.neon.vcvtfp2fxs.v2i64.v2f64(<2 x double> %a, i32 50)
  ret <2 x i64> %vcvt
}

define <2 x i32> @test_vcvt_n_u32_f32(<2 x float> %a) {
; CHECK: test_vcvt_n_u32_f32
; CHECK: fcvtzu {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #31
  %vcvt = tail call <2 x i32> @llvm.arm.neon.vcvtfp2fxu.v2i32.v2f32(<2 x float> %a, i32 31)
  ret <2 x i32> %vcvt
}

define <4 x i32> @test_vcvtq_n_u32_f32(<4 x float> %a) {
; CHECK: test_vcvt_n_u32_f32
; CHECK: fcvtzu {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #31
  %vcvt = tail call <4 x i32> @llvm.arm.neon.vcvtfp2fxu.v4i32.v4f32(<4 x float> %a, i32 31)
  ret <4 x i32> %vcvt
}

define <2 x i64> @test_vcvtq_n_u64_f64(<2 x double> %a) {
; CHECK: test_vcvtq_n_u64_f64
; CHECK: fcvtzu {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #50
  %vcvt = tail call <2 x i64> @llvm.arm.neon.vcvtfp2fxu.v2i64.v2f64(<2 x double> %a, i32 50)
  ret <2 x i64> %vcvt
}

declare <8 x i8> @llvm.aarch64.neon.vsrshr.v8i8(<8 x i8>, i32)

declare <4 x i16> @llvm.aarch64.neon.vsrshr.v4i16(<4 x i16>, i32)

declare <2 x i32> @llvm.aarch64.neon.vsrshr.v2i32(<2 x i32>, i32)

declare <16 x i8> @llvm.aarch64.neon.vsrshr.v16i8(<16 x i8>, i32)

declare <8 x i16> @llvm.aarch64.neon.vsrshr.v8i16(<8 x i16>, i32)

declare <4 x i32> @llvm.aarch64.neon.vsrshr.v4i32(<4 x i32>, i32)

declare <2 x i64> @llvm.aarch64.neon.vsrshr.v2i64(<2 x i64>, i32)

declare <8 x i8> @llvm.aarch64.neon.vurshr.v8i8(<8 x i8>, i32)

declare <4 x i16> @llvm.aarch64.neon.vurshr.v4i16(<4 x i16>, i32)

declare <2 x i32> @llvm.aarch64.neon.vurshr.v2i32(<2 x i32>, i32)

declare <16 x i8> @llvm.aarch64.neon.vurshr.v16i8(<16 x i8>, i32)

declare <8 x i16> @llvm.aarch64.neon.vurshr.v8i16(<8 x i16>, i32)

declare <4 x i32> @llvm.aarch64.neon.vurshr.v4i32(<4 x i32>, i32)

declare <2 x i64> @llvm.aarch64.neon.vurshr.v2i64(<2 x i64>, i32)

declare <8 x i8> @llvm.aarch64.neon.vsri.v8i8(<8 x i8>, <8 x i8>, i32)

declare <4 x i16> @llvm.aarch64.neon.vsri.v4i16(<4 x i16>, <4 x i16>, i32)

declare <2 x i32> @llvm.aarch64.neon.vsri.v2i32(<2 x i32>, <2 x i32>, i32)

declare <16 x i8> @llvm.aarch64.neon.vsri.v16i8(<16 x i8>, <16 x i8>, i32)

declare <8 x i16> @llvm.aarch64.neon.vsri.v8i16(<8 x i16>, <8 x i16>, i32)

declare <4 x i32> @llvm.aarch64.neon.vsri.v4i32(<4 x i32>, <4 x i32>, i32)

declare <2 x i64> @llvm.aarch64.neon.vsri.v2i64(<2 x i64>, <2 x i64>, i32)

declare <8 x i8> @llvm.aarch64.neon.vsli.v8i8(<8 x i8>, <8 x i8>, i32)

declare <4 x i16> @llvm.aarch64.neon.vsli.v4i16(<4 x i16>, <4 x i16>, i32)

declare <2 x i32> @llvm.aarch64.neon.vsli.v2i32(<2 x i32>, <2 x i32>, i32)

declare <16 x i8> @llvm.aarch64.neon.vsli.v16i8(<16 x i8>, <16 x i8>, i32)

declare <8 x i16> @llvm.aarch64.neon.vsli.v8i16(<8 x i16>, <8 x i16>, i32)

declare <4 x i32> @llvm.aarch64.neon.vsli.v4i32(<4 x i32>, <4 x i32>, i32)

declare <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64>, <2 x i64>, i32)

declare <8 x i8> @llvm.aarch64.neon.vsqshlu.v8i8(<8 x i8>, i32)

declare <4 x i16> @llvm.aarch64.neon.vsqshlu.v4i16(<4 x i16>, i32)

declare <2 x i32> @llvm.aarch64.neon.vsqshlu.v2i32(<2 x i32>, i32)

declare <16 x i8> @llvm.aarch64.neon.vsqshlu.v16i8(<16 x i8>, i32)

declare <8 x i16> @llvm.aarch64.neon.vsqshlu.v8i16(<8 x i16>, i32)

declare <4 x i32> @llvm.aarch64.neon.vsqshlu.v4i32(<4 x i32>, i32)

declare <2 x i64> @llvm.aarch64.neon.vsqshlu.v2i64(<2 x i64>, i32)

declare <8 x i8> @llvm.arm.neon.vqshifts.v8i8(<8 x i8>, <8 x i8>)

declare <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16>, <4 x i16>)

declare <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32>, <2 x i32>)

declare <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8>, <16 x i8>)

declare <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16>, <8 x i16>)

declare <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32>, <4 x i32>)

declare <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64>, <2 x i64>)

declare <8 x i8> @llvm.arm.neon.vqshiftu.v8i8(<8 x i8>, <8 x i8>)

declare <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16>, <4 x i16>)

declare <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32>, <2 x i32>) 

declare <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8>, <16 x i8>) 

declare <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16>, <8 x i16>) 

declare <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32>, <4 x i32>)

declare <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64>, <2 x i64>)

declare <8 x i8> @llvm.aarch64.neon.vsqshrun.v8i8(<8 x i16>, i32)

declare <4 x i16> @llvm.aarch64.neon.vsqshrun.v4i16(<4 x i32>, i32)

declare <2 x i32> @llvm.aarch64.neon.vsqshrun.v2i32(<2 x i64>, i32)

declare <8 x i8> @llvm.aarch64.neon.vrshrn.v8i8(<8 x i16>, i32)

declare <4 x i16> @llvm.aarch64.neon.vrshrn.v4i16(<4 x i32>, i32)

declare <2 x i32> @llvm.aarch64.neon.vrshrn.v2i32(<2 x i64>, i32)

declare <8 x i8> @llvm.aarch64.neon.vsqrshrun.v8i8(<8 x i16>, i32)

declare <4 x i16> @llvm.aarch64.neon.vsqrshrun.v4i16(<4 x i32>, i32)

declare <2 x i32> @llvm.aarch64.neon.vsqrshrun.v2i32(<2 x i64>, i32)

declare <8 x i8> @llvm.aarch64.neon.vsqshrn.v8i8(<8 x i16>, i32)

declare <4 x i16> @llvm.aarch64.neon.vsqshrn.v4i16(<4 x i32>, i32)

declare <2 x i32> @llvm.aarch64.neon.vsqshrn.v2i32(<2 x i64>, i32)

declare <8 x i8> @llvm.aarch64.neon.vuqshrn.v8i8(<8 x i16>, i32)

declare <4 x i16> @llvm.aarch64.neon.vuqshrn.v4i16(<4 x i32>, i32)

declare <2 x i32> @llvm.aarch64.neon.vuqshrn.v2i32(<2 x i64>, i32)

declare <8 x i8> @llvm.aarch64.neon.vsqrshrn.v8i8(<8 x i16>, i32)

declare <4 x i16> @llvm.aarch64.neon.vsqrshrn.v4i16(<4 x i32>, i32)

declare <2 x i32> @llvm.aarch64.neon.vsqrshrn.v2i32(<2 x i64>, i32)

declare <8 x i8> @llvm.aarch64.neon.vuqrshrn.v8i8(<8 x i16>, i32)

declare <4 x i16> @llvm.aarch64.neon.vuqrshrn.v4i16(<4 x i32>, i32)

declare <2 x i32> @llvm.aarch64.neon.vuqrshrn.v2i32(<2 x i64>, i32)

declare <2 x float> @llvm.arm.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32>, i32)

declare <4 x float> @llvm.arm.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32>, i32)

declare <2 x double> @llvm.arm.neon.vcvtfxs2fp.v2f64.v2i64(<2 x i64>, i32)

declare <2 x float> @llvm.arm.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32>, i32)

declare <4 x float> @llvm.arm.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32>, i32)

declare <2 x double> @llvm.arm.neon.vcvtfxu2fp.v2f64.v2i64(<2 x i64>, i32)

declare <2 x i32> @llvm.arm.neon.vcvtfp2fxs.v2i32.v2f32(<2 x float>, i32)

declare <4 x i32> @llvm.arm.neon.vcvtfp2fxs.v4i32.v4f32(<4 x float>, i32)

declare <2 x i64> @llvm.arm.neon.vcvtfp2fxs.v2i64.v2f64(<2 x double>, i32)

declare <2 x i32> @llvm.arm.neon.vcvtfp2fxu.v2i32.v2f32(<2 x float>, i32)

declare <4 x i32> @llvm.arm.neon.vcvtfp2fxu.v4i32.v4f32(<4 x float>, i32)

declare <2 x i64> @llvm.arm.neon.vcvtfp2fxu.v2i64.v2f64(<2 x double>, i32)

define <1 x i64> @test_vcvt_n_s64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvt_n_s64_f64
; CHECK: fcvtzs d{{[0-9]+}}, d{{[0-9]+}}, #64
  %1 = tail call <1 x i64> @llvm.arm.neon.vcvtfp2fxs.v1i64.v1f64(<1 x double> %a, i32 64)
  ret <1 x i64> %1
}

define <1 x i64> @test_vcvt_n_u64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvt_n_u64_f64
; CHECK: fcvtzu d{{[0-9]+}}, d{{[0-9]+}}, #64
  %1 = tail call <1 x i64> @llvm.arm.neon.vcvtfp2fxu.v1i64.v1f64(<1 x double> %a, i32 64)
  ret <1 x i64> %1
}

define <1 x double> @test_vcvt_n_f64_s64(<1 x i64> %a) {
; CHECK-LABEL: test_vcvt_n_f64_s64
; CHECK: scvtf d{{[0-9]+}}, d{{[0-9]+}}, #64
  %1 = tail call <1 x double> @llvm.arm.neon.vcvtfxs2fp.v1f64.v1i64(<1 x i64> %a, i32 64)
  ret <1 x double> %1
}

define <1 x double> @test_vcvt_n_f64_u64(<1 x i64> %a) {
; CHECK-LABEL: test_vcvt_n_f64_u64
; CHECK: ucvtf d{{[0-9]+}}, d{{[0-9]+}}, #64
  %1 = tail call <1 x double> @llvm.arm.neon.vcvtfxu2fp.v1f64.v1i64(<1 x i64> %a, i32 64)
  ret <1 x double> %1
}

declare <1 x i64> @llvm.arm.neon.vcvtfp2fxs.v1i64.v1f64(<1 x double>, i32)
declare <1 x i64> @llvm.arm.neon.vcvtfp2fxu.v1i64.v1f64(<1 x double>, i32)
declare <1 x double> @llvm.arm.neon.vcvtfxs2fp.v1f64.v1i64(<1 x i64>, i32)
declare <1 x double> @llvm.arm.neon.vcvtfxu2fp.v1f64.v1i64(<1 x i64>, i32)