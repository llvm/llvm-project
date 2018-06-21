; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <8 x i8> @test_vext_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vext_s8:
; CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x2|2}}
entry:
  %vext = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  ret <8 x i8> %vext
}

define <4 x i16> @test_vext_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vext_s16:
; CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x6|6}}
entry:
  %vext = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x i16> %vext
}

define <2 x i32> @test_vext_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vext_s32:
; CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x4|4}}
entry:
  %vext = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x i32> %vext
}

define <1 x i64> @test_vext_s64(<1 x i64> %a, <1 x i64> %b) {
; CHECK-LABEL: test_vext_s64:
entry:
  %vext = shufflevector <1 x i64> %a, <1 x i64> %b, <1 x i32> <i32 0>
  ret <1 x i64> %vext
}

define <16 x i8> @test_vextq_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vextq_s8:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x2|2}}
entry:
  %vext = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  ret <16 x i8> %vext
}

define <8 x i16> @test_vextq_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vextq_s16:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x6|6}}
entry:
  %vext = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  ret <8 x i16> %vext
}

define <4 x i32> @test_vextq_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vextq_s32:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x4|4}}
entry:
  %vext = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i32> %vext
}

define <2 x i64> @test_vextq_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vextq_s64:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x8|8}}
entry:
  %vext = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x i64> %vext
}

define <8 x i8> @test_vext_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vext_u8:
; CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x2|2}}
entry:
  %vext = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  ret <8 x i8> %vext
}

define <4 x i16> @test_vext_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vext_u16:
; CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x6|6}}
entry:
  %vext = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x i16> %vext
}

define <2 x i32> @test_vext_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vext_u32:
; CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x4|4}}
entry:
  %vext = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x i32> %vext
}

define <1 x i64> @test_vext_u64(<1 x i64> %a, <1 x i64> %b) {
; CHECK-LABEL: test_vext_u64:
entry:
  %vext = shufflevector <1 x i64> %a, <1 x i64> %b, <1 x i32> <i32 0>
  ret <1 x i64> %vext
}

define <16 x i8> @test_vextq_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vextq_u8:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x2|2}}
entry:
  %vext = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  ret <16 x i8> %vext
}

define <8 x i16> @test_vextq_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vextq_u16:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x6|6}}
entry:
  %vext = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  ret <8 x i16> %vext
}

define <4 x i32> @test_vextq_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vextq_u32:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x4|4}}
entry:
  %vext = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i32> %vext
}

define <2 x i64> @test_vextq_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vextq_u64:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x8|8}}
entry:
  %vext = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x i64> %vext
}

define <2 x float> @test_vext_f32(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: test_vext_f32:
; CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x4|4}}
entry:
  %vext = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %vext
}

define <1 x double> @test_vext_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vext_f64:
entry:
  %vext = shufflevector <1 x double> %a, <1 x double> %b, <1 x i32> <i32 0>
  ret <1 x double> %vext
}

define <4 x float> @test_vextq_f32(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: test_vextq_f32:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x4|4}}
entry:
  %vext = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x float> %vext
}

define <2 x double> @test_vextq_f64(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test_vextq_f64:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x8|8}}
entry:
  %vext = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x double> %vext
}

define <8 x i8> @test_vext_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vext_p8:
; CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x2|2}}
entry:
  %vext = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  ret <8 x i8> %vext
}

define <4 x i16> @test_vext_p16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vext_p16:
; CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x6|6}}
entry:
  %vext = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x i16> %vext
}

define <16 x i8> @test_vextq_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vextq_p8:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x2|2}}
entry:
  %vext = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  ret <16 x i8> %vext
}

define <8 x i16> @test_vextq_p16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vextq_p16:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x6|6}}
entry:
  %vext = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  ret <8 x i16> %vext
}

define <8 x i8> @test_undef_vext_s8(<8 x i8> %a) {
; CHECK-LABEL: test_undef_vext_s8:
; CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x2|2}}
entry:
  %vext = shufflevector <8 x i8> %a, <8 x i8> undef, <8 x i32> <i32 10, i32 10, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  ret <8 x i8> %vext
}

define <16 x i8> @test_undef_vextq_s8(<16 x i8> %a) {
; CHECK-LABEL: test_undef_vextq_s8:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x6|6}}
entry:
  %vext = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> <i32 20, i32 20, i32 20, i32 20, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 20, i32 20, i32 20, i32 20, i32 20>
  ret <16 x i8> %vext
}

define <4 x i16> @test_undef_vext_s16(<4 x i16> %a) {
; CHECK-LABEL: test_undef_vext_s16:
; CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x4|4}}
entry:
  %vext = shufflevector <4 x i16> %a, <4 x i16> undef, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  ret <4 x i16> %vext
}

define <8 x i16> @test_undef_vextq_s16(<8 x i16> %a) {
; CHECK-LABEL: test_undef_vextq_s16:
; CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x6|6}}
entry:
  %vext = shufflevector <8 x i16> %a, <8 x i16> undef, <8 x i32> <i32 10, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  ret <8 x i16> %vext
}
