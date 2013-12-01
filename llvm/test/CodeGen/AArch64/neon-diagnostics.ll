; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <2 x float> @test_vfma_lane_f32(<2 x float> %a, <2 x float> %b, <2 x float> %v) {
; CHECK: test_vfma_lane_f32:
; CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[{{[0-9]+}}]
; CHECK: fadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %shuffle = shufflevector <2 x float> %v, <2 x float> undef, <2 x i32> <i32 1, i32 1>
  %mul = fmul <2 x float> %shuffle, %b
  %add = fadd <2 x float> %mul, %a
  ret <2 x float> %add
}

define <4 x i32> @test_vshrn_not_match(<2 x i32> %a, <2 x i64> %b) {
; CHECK: test_vshrn_not_match
; CHECK-NOT: shrn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, #35
  %1 = bitcast <2 x i32> %a to <1 x i64>
  %2 = ashr <2 x i64> %b, <i64 35, i64 35>
  %vshrn_n = trunc <2 x i64> %2 to <2 x i32>
  %3 = bitcast <2 x i32> %vshrn_n to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %1, <1 x i64> %3, <2 x i32> <i32 0, i32 1>
  %4 = bitcast <2 x i64> %shuffle.i to <4 x i32>
  ret <4 x i32> %4
}