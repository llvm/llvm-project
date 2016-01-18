; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s


define <16 x i8> @ins16bw(<16 x i8> %tmp1, i8 %tmp2) {
; CHECK-LABEL: ins16bw:
; CHECK: ins {{v[0-9]+}}.b[15], {{w[0-9]+}}
  %tmp3 = insertelement <16 x i8> %tmp1, i8 %tmp2, i32 15
  ret <16 x i8> %tmp3
}

define <8 x i16> @ins8hw(<8 x i16> %tmp1, i16 %tmp2) {
; CHECK-LABEL: ins8hw:
; CHECK: ins {{v[0-9]+}}.h[6], {{w[0-9]+}}
  %tmp3 = insertelement <8 x i16> %tmp1, i16 %tmp2, i32 6
  ret <8 x i16> %tmp3
}

define <4 x i32> @ins4sw(<4 x i32> %tmp1, i32 %tmp2) {
; CHECK-LABEL: ins4sw:
; CHECK: ins {{v[0-9]+}}.s[2], {{w[0-9]+}}
  %tmp3 = insertelement <4 x i32> %tmp1, i32 %tmp2, i32 2
  ret <4 x i32> %tmp3
}

define <2 x i64> @ins2dw(<2 x i64> %tmp1, i64 %tmp2) {
; CHECK-LABEL: ins2dw:
; CHECK: ins {{v[0-9]+}}.d[1], {{x[0-9]+}}
  %tmp3 = insertelement <2 x i64> %tmp1, i64 %tmp2, i32 1
  ret <2 x i64> %tmp3
}

define <8 x i8> @ins8bw(<8 x i8> %tmp1, i8 %tmp2) {
; CHECK-LABEL: ins8bw:
; CHECK: ins {{v[0-9]+}}.b[5], {{w[0-9]+}}
  %tmp3 = insertelement <8 x i8> %tmp1, i8 %tmp2, i32 5
  ret <8 x i8> %tmp3
}

define <4 x i16> @ins4hw(<4 x i16> %tmp1, i16 %tmp2) {
; CHECK-LABEL: ins4hw:
; CHECK: ins {{v[0-9]+}}.h[3], {{w[0-9]+}}
  %tmp3 = insertelement <4 x i16> %tmp1, i16 %tmp2, i32 3
  ret <4 x i16> %tmp3
}

define <2 x i32> @ins2sw(<2 x i32> %tmp1, i32 %tmp2) {
; CHECK-LABEL: ins2sw:
; CHECK: ins {{v[0-9]+}}.s[1], {{w[0-9]+}}
  %tmp3 = insertelement <2 x i32> %tmp1, i32 %tmp2, i32 1
  ret <2 x i32> %tmp3
}

define <16 x i8> @ins16b16(<16 x i8> %tmp1, <16 x i8> %tmp2) {
; CHECK-LABEL: ins16b16:
; CHECK: ins {{v[0-9]+}}.b[15], {{v[0-9]+}}.b[2]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 2
  %tmp4 = insertelement <16 x i8> %tmp2, i8 %tmp3, i32 15
  ret <16 x i8> %tmp4
}

define <8 x i16> @ins8h8(<8 x i16> %tmp1, <8 x i16> %tmp2) {
; CHECK-LABEL: ins8h8:
; CHECK: ins {{v[0-9]+}}.h[7], {{v[0-9]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = insertelement <8 x i16> %tmp2, i16 %tmp3, i32 7
  ret <8 x i16> %tmp4
}

define <4 x i32> @ins4s4(<4 x i32> %tmp1, <4 x i32> %tmp2) {
; CHECK-LABEL: ins4s4:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[2]
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  %tmp4 = insertelement <4 x i32> %tmp2, i32 %tmp3, i32 1
  ret <4 x i32> %tmp4
}

define <2 x i64> @ins2d2(<2 x i64> %tmp1, <2 x i64> %tmp2) {
; CHECK-LABEL: ins2d2:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
  %tmp3 = extractelement <2 x i64> %tmp1, i32 0
  %tmp4 = insertelement <2 x i64> %tmp2, i64 %tmp3, i32 1
  ret <2 x i64> %tmp4
}

define <4 x float> @ins4f4(<4 x float> %tmp1, <4 x float> %tmp2) {
; CHECK-LABEL: ins4f4:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[2]
  %tmp3 = extractelement <4 x float> %tmp1, i32 2
  %tmp4 = insertelement <4 x float> %tmp2, float %tmp3, i32 1
  ret <4 x float> %tmp4
}

define <2 x double> @ins2df2(<2 x double> %tmp1, <2 x double> %tmp2) {
; CHECK-LABEL: ins2df2:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
  %tmp3 = extractelement <2 x double> %tmp1, i32 0
  %tmp4 = insertelement <2 x double> %tmp2, double %tmp3, i32 1
  ret <2 x double> %tmp4
}

define <16 x i8> @ins8b16(<8 x i8> %tmp1, <16 x i8> %tmp2) {
; CHECK-LABEL: ins8b16:
; CHECK: ins {{v[0-9]+}}.b[15], {{v[0-9]+}}.b[2]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 2
  %tmp4 = insertelement <16 x i8> %tmp2, i8 %tmp3, i32 15
  ret <16 x i8> %tmp4
}

define <8 x i16> @ins4h8(<4 x i16> %tmp1, <8 x i16> %tmp2) {
; CHECK-LABEL: ins4h8:
; CHECK: ins {{v[0-9]+}}.h[7], {{v[0-9]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = insertelement <8 x i16> %tmp2, i16 %tmp3, i32 7
  ret <8 x i16> %tmp4
}

define <4 x i32> @ins2s4(<2 x i32> %tmp1, <4 x i32> %tmp2) {
; CHECK-LABEL: ins2s4:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[1]
  %tmp3 = extractelement <2 x i32> %tmp1, i32 1
  %tmp4 = insertelement <4 x i32> %tmp2, i32 %tmp3, i32 1
  ret <4 x i32> %tmp4
}

define <2 x i64> @ins1d2(<1 x i64> %tmp1, <2 x i64> %tmp2) {
; CHECK-LABEL: ins1d2:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
  %tmp3 = extractelement <1 x i64> %tmp1, i32 0
  %tmp4 = insertelement <2 x i64> %tmp2, i64 %tmp3, i32 1
  ret <2 x i64> %tmp4
}

define <4 x float> @ins2f4(<2 x float> %tmp1, <4 x float> %tmp2) {
; CHECK-LABEL: ins2f4:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[1]
  %tmp3 = extractelement <2 x float> %tmp1, i32 1
  %tmp4 = insertelement <4 x float> %tmp2, float %tmp3, i32 1
  ret <4 x float> %tmp4
}

define <2 x double> @ins1f2(<1 x double> %tmp1, <2 x double> %tmp2) {
; CHECK-LABEL: ins1f2:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
  %tmp3 = extractelement <1 x double> %tmp1, i32 0
  %tmp4 = insertelement <2 x double> %tmp2, double %tmp3, i32 1
  ret <2 x double> %tmp4
}

define <8 x i8> @ins16b8(<16 x i8> %tmp1, <8 x i8> %tmp2) {
; CHECK-LABEL: ins16b8:
; CHECK: ins {{v[0-9]+}}.b[7], {{v[0-9]+}}.b[2]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 2
  %tmp4 = insertelement <8 x i8> %tmp2, i8 %tmp3, i32 7
  ret <8 x i8> %tmp4
}

define <4 x i16> @ins8h4(<8 x i16> %tmp1, <4 x i16> %tmp2) {
; CHECK-LABEL: ins8h4:
; CHECK: ins {{v[0-9]+}}.h[3], {{v[0-9]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = insertelement <4 x i16> %tmp2, i16 %tmp3, i32 3
  ret <4 x i16> %tmp4
}

define <2 x i32> @ins4s2(<4 x i32> %tmp1, <2 x i32> %tmp2) {
; CHECK-LABEL: ins4s2:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[2]
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  %tmp4 = insertelement <2 x i32> %tmp2, i32 %tmp3, i32 1
  ret <2 x i32> %tmp4
}

define <1 x i64> @ins2d1(<2 x i64> %tmp1, <1 x i64> %tmp2) {
; CHECK-LABEL: ins2d1:
; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[0]
  %tmp3 = extractelement <2 x i64> %tmp1, i32 0
  %tmp4 = insertelement <1 x i64> %tmp2, i64 %tmp3, i32 0
  ret <1 x i64> %tmp4
}

define <2 x float> @ins4f2(<4 x float> %tmp1, <2 x float> %tmp2) {
; CHECK-LABEL: ins4f2:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[2]
  %tmp3 = extractelement <4 x float> %tmp1, i32 2
  %tmp4 = insertelement <2 x float> %tmp2, float %tmp3, i32 1
  ret <2 x float> %tmp4
}

define <1 x double> @ins2f1(<2 x double> %tmp1, <1 x double> %tmp2) {
; CHECK-LABEL: ins2f1:
; CHECK: mov {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp3 = extractelement <2 x double> %tmp1, i32 1
  %tmp4 = insertelement <1 x double> %tmp2, double %tmp3, i32 0
  ret <1 x double> %tmp4
}

define <8 x i8> @ins8b8(<8 x i8> %tmp1, <8 x i8> %tmp2) {
; CHECK-LABEL: ins8b8:
; CHECK: ins {{v[0-9]+}}.b[4], {{v[0-9]+}}.b[2]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 2
  %tmp4 = insertelement <8 x i8> %tmp2, i8 %tmp3, i32 4
  ret <8 x i8> %tmp4
}

define <4 x i16> @ins4h4(<4 x i16> %tmp1, <4 x i16> %tmp2) {
; CHECK-LABEL: ins4h4:
; CHECK: ins {{v[0-9]+}}.h[3], {{v[0-9]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = insertelement <4 x i16> %tmp2, i16 %tmp3, i32 3
  ret <4 x i16> %tmp4
}

define <2 x i32> @ins2s2(<2 x i32> %tmp1, <2 x i32> %tmp2) {
; CHECK-LABEL: ins2s2:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
  %tmp3 = extractelement <2 x i32> %tmp1, i32 0
  %tmp4 = insertelement <2 x i32> %tmp2, i32 %tmp3, i32 1
  ret <2 x i32> %tmp4
}

define <1 x i64> @ins1d1(<1 x i64> %tmp1, <1 x i64> %tmp2) {
; CHECK-LABEL: ins1d1:
; CHECK: ins {{v[0-9]+}}.d[0], {{v[0-9]+}}.d[0]
  %tmp3 = extractelement <1 x i64> %tmp1, i32 0
  %tmp4 = insertelement <1 x i64> %tmp2, i64 %tmp3, i32 0
  ret <1 x i64> %tmp4
}

define <2 x float> @ins2f2(<2 x float> %tmp1, <2 x float> %tmp2) {
; CHECK-LABEL: ins2f2:
; CHECK: ins {{v[0-9]+}}.s[1], {{v[0-9]+}}.s[0]
  %tmp3 = extractelement <2 x float> %tmp1, i32 0
  %tmp4 = insertelement <2 x float> %tmp2, float %tmp3, i32 1
  ret <2 x float> %tmp4
}

define <1 x double> @ins1df1(<1 x double> %tmp1, <1 x double> %tmp2) {
; CHECK-LABEL: ins1df1:
; CHECK-NOT: ins {{v[0-9]+}}
  %tmp3 = extractelement <1 x double> %tmp1, i32 0
  %tmp4 = insertelement <1 x double> %tmp2, double %tmp3, i32 0
  ret <1 x double> %tmp4
}

define i32 @umovw16b(<16 x i8> %tmp1) {
; CHECK-LABEL: umovw16b:
; CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.b[8]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 8
  %tmp4 = zext i8 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw8h(<8 x i16> %tmp1) {
; CHECK-LABEL: umovw8h:
; CHECK: umov {{w[0-9]+}}, {{v[0-9]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = zext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw4s(<4 x i32> %tmp1) {
; CHECK-LABEL: umovw4s:
; CHECK: mov {{w[0-9]+}}, {{v[0-9]+}}.s[2]
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  ret i32 %tmp3
}

define i64 @umovx2d(<2 x i64> %tmp1) {
; CHECK-LABEL: umovx2d:
; CHECK: mov {{x[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp3 = extractelement <2 x i64> %tmp1, i32 1
  ret i64 %tmp3
}

define i32 @umovw8b(<8 x i8> %tmp1) {
; CHECK-LABEL: umovw8b:
; CHECK: mov {{w[0-9]+}}, {{v[0-9]+}}.b[7]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 7
  %tmp4 = zext i8 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw4h(<4 x i16> %tmp1) {
; CHECK-LABEL: umovw4h:
; CHECK: mov {{w[0-9]+}}, {{v[0-9]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = zext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw2s(<2 x i32> %tmp1) {
; CHECK-LABEL: umovw2s:
; CHECK: mov {{w[0-9]+}}, {{v[0-9]+}}.s[1]
  %tmp3 = extractelement <2 x i32> %tmp1, i32 1
  ret i32 %tmp3
}

define i64 @umovx1d(<1 x i64> %tmp1) {
; CHECK-LABEL: umovx1d:
; CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
  %tmp3 = extractelement <1 x i64> %tmp1, i32 0
  ret i64 %tmp3
}

define i32 @smovw16b(<16 x i8> %tmp1) {
; CHECK-LABEL: smovw16b:
; CHECK: smov {{w[0-9]+}}, {{v[0-9]+}}.b[8]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 8
  %tmp4 = sext i8 %tmp3 to i32
  %tmp5 = add i32 %tmp4, %tmp4
  ret i32 %tmp5
}

define i32 @smovw8h(<8 x i16> %tmp1) {
; CHECK-LABEL: smovw8h:
; CHECK: smov {{w[0-9]+}}, {{v[0-9]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  %tmp5 = add i32 %tmp4, %tmp4
  ret i32 %tmp5
}

define i32 @smovx16b(<16 x i8> %tmp1) {
; CHECK-LABEL: smovx16b:
; CHECK: smov {{[xw][0-9]+}}, {{v[0-9]+}}.b[8]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 8
  %tmp4 = sext i8 %tmp3 to i32
  %tmp5 = add i32 %tmp4, %tmp4
  ret i32 %tmp5
}

define i32 @smovx8h(<8 x i16> %tmp1) {
; CHECK-LABEL: smovx8h:
; CHECK: smov {{[xw][0-9]+}}, {{v[0-9]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i64 @smovx4s(<4 x i32> %tmp1) {
; CHECK-LABEL: smovx4s:
; CHECK: smov {{x[0-9]+}}, {{v[0-9]+}}.s[2]
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  %tmp4 = sext i32 %tmp3 to i64
  ret i64 %tmp4
}

define i32 @smovw8b(<8 x i8> %tmp1) {
; CHECK-LABEL: smovw8b:
; CHECK: smov {{w[0-9]+}}, {{v[0-9]+}}.b[4]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 4
  %tmp4 = sext i8 %tmp3 to i32
  %tmp5 = add i32 %tmp4, %tmp4
  ret i32 %tmp5
}

define i32 @smovw4h(<4 x i16> %tmp1) {
; CHECK-LABEL: smovw4h:
; CHECK: smov {{w[0-9]+}}, {{v[0-9]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  %tmp5 = add i32 %tmp4, %tmp4
  ret i32 %tmp5
}

define i32 @smovx8b(<8 x i8> %tmp1) {
; CHECK-LABEL: smovx8b:
; CHECK: smov {{[xw][0-9]+}}, {{v[0-9]+}}.b[6]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 6
  %tmp4 = sext i8 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @smovx4h(<4 x i16> %tmp1) {
; CHECK-LABEL: smovx4h:
; CHECK: smov {{[xw][0-9]+}}, {{v[0-9]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i64 @smovx2s(<2 x i32> %tmp1) {
; CHECK-LABEL: smovx2s:
; CHECK: smov {{x[0-9]+}}, {{v[0-9]+}}.s[1]
  %tmp3 = extractelement <2 x i32> %tmp1, i32 1
  %tmp4 = sext i32 %tmp3 to i64
  ret i64 %tmp4
}

define <8 x i8> @test_vcopy_lane_s8(<8 x i8> %v1, <8 x i8> %v2) {
; CHECK-LABEL: test_vcopy_lane_s8:
; CHECK: ins  {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
  %vset_lane = shufflevector <8 x i8> %v1, <8 x i8> %v2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 11, i32 6, i32 7>
  ret <8 x i8> %vset_lane
}

define <16 x i8> @test_vcopyq_laneq_s8(<16 x i8> %v1, <16 x i8> %v2) {
; CHECK-LABEL: test_vcopyq_laneq_s8:
; CHECK: ins  {{v[0-9]+}}.b[14], {{v[0-9]+}}.b[6]
  %vset_lane = shufflevector <16 x i8> %v1, <16 x i8> %v2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 22, i32 15>
  ret <16 x i8> %vset_lane
}

define <8 x i8> @test_vcopy_lane_swap_s8(<8 x i8> %v1, <8 x i8> %v2) {
; CHECK-LABEL: test_vcopy_lane_swap_s8:
; CHECK: ins {{v[0-9]+}}.b[7], {{v[0-9]+}}.b[0]
  %vset_lane = shufflevector <8 x i8> %v1, <8 x i8> %v2, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 0>
  ret <8 x i8> %vset_lane
}

define <16 x i8> @test_vcopyq_laneq_swap_s8(<16 x i8> %v1, <16 x i8> %v2) {
; CHECK-LABEL: test_vcopyq_laneq_swap_s8:
; CHECK: ins {{v[0-9]+}}.b[0], {{v[0-9]+}}.b[15]
  %vset_lane = shufflevector <16 x i8> %v1, <16 x i8> %v2, <16 x i32> <i32 15, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vset_lane
}

define <8 x i8> @test_vdup_n_u8(i8 %v1) #0 {
; CHECK-LABEL: test_vdup_n_u8:
; CHECK: dup {{v[0-9]+}}.8b, {{w[0-9]+}}
  %vecinit.i = insertelement <8 x i8> undef, i8 %v1, i32 0
  %vecinit1.i = insertelement <8 x i8> %vecinit.i, i8 %v1, i32 1
  %vecinit2.i = insertelement <8 x i8> %vecinit1.i, i8 %v1, i32 2
  %vecinit3.i = insertelement <8 x i8> %vecinit2.i, i8 %v1, i32 3
  %vecinit4.i = insertelement <8 x i8> %vecinit3.i, i8 %v1, i32 4
  %vecinit5.i = insertelement <8 x i8> %vecinit4.i, i8 %v1, i32 5
  %vecinit6.i = insertelement <8 x i8> %vecinit5.i, i8 %v1, i32 6
  %vecinit7.i = insertelement <8 x i8> %vecinit6.i, i8 %v1, i32 7
  ret <8 x i8> %vecinit7.i
}

define <4 x i16> @test_vdup_n_u16(i16 %v1) #0 {
; CHECK-LABEL: test_vdup_n_u16:
; CHECK: dup {{v[0-9]+}}.4h, {{w[0-9]+}}
  %vecinit.i = insertelement <4 x i16> undef, i16 %v1, i32 0
  %vecinit1.i = insertelement <4 x i16> %vecinit.i, i16 %v1, i32 1
  %vecinit2.i = insertelement <4 x i16> %vecinit1.i, i16 %v1, i32 2
  %vecinit3.i = insertelement <4 x i16> %vecinit2.i, i16 %v1, i32 3
  ret <4 x i16> %vecinit3.i
}

define <2 x i32> @test_vdup_n_u32(i32 %v1) #0 {
; CHECK-LABEL: test_vdup_n_u32:
; CHECK: dup {{v[0-9]+}}.2s, {{w[0-9]+}}
  %vecinit.i = insertelement <2 x i32> undef, i32 %v1, i32 0
  %vecinit1.i = insertelement <2 x i32> %vecinit.i, i32 %v1, i32 1
  ret <2 x i32> %vecinit1.i
}

define <1 x i64> @test_vdup_n_u64(i64 %v1) #0 {
; CHECK-LABEL: test_vdup_n_u64:
; CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
  %vecinit.i = insertelement <1 x i64> undef, i64 %v1, i32 0
  ret <1 x i64> %vecinit.i
}

define <16 x i8> @test_vdupq_n_u8(i8 %v1) #0 {
; CHECK-LABEL: test_vdupq_n_u8:
; CHECK: dup {{v[0-9]+}}.16b, {{w[0-9]+}}
  %vecinit.i = insertelement <16 x i8> undef, i8 %v1, i32 0
  %vecinit1.i = insertelement <16 x i8> %vecinit.i, i8 %v1, i32 1
  %vecinit2.i = insertelement <16 x i8> %vecinit1.i, i8 %v1, i32 2
  %vecinit3.i = insertelement <16 x i8> %vecinit2.i, i8 %v1, i32 3
  %vecinit4.i = insertelement <16 x i8> %vecinit3.i, i8 %v1, i32 4
  %vecinit5.i = insertelement <16 x i8> %vecinit4.i, i8 %v1, i32 5
  %vecinit6.i = insertelement <16 x i8> %vecinit5.i, i8 %v1, i32 6
  %vecinit7.i = insertelement <16 x i8> %vecinit6.i, i8 %v1, i32 7
  %vecinit8.i = insertelement <16 x i8> %vecinit7.i, i8 %v1, i32 8
  %vecinit9.i = insertelement <16 x i8> %vecinit8.i, i8 %v1, i32 9
  %vecinit10.i = insertelement <16 x i8> %vecinit9.i, i8 %v1, i32 10
  %vecinit11.i = insertelement <16 x i8> %vecinit10.i, i8 %v1, i32 11
  %vecinit12.i = insertelement <16 x i8> %vecinit11.i, i8 %v1, i32 12
  %vecinit13.i = insertelement <16 x i8> %vecinit12.i, i8 %v1, i32 13
  %vecinit14.i = insertelement <16 x i8> %vecinit13.i, i8 %v1, i32 14
  %vecinit15.i = insertelement <16 x i8> %vecinit14.i, i8 %v1, i32 15
  ret <16 x i8> %vecinit15.i
}

define <8 x i16> @test_vdupq_n_u16(i16 %v1) #0 {
; CHECK-LABEL: test_vdupq_n_u16:
; CHECK: dup {{v[0-9]+}}.8h, {{w[0-9]+}}
  %vecinit.i = insertelement <8 x i16> undef, i16 %v1, i32 0
  %vecinit1.i = insertelement <8 x i16> %vecinit.i, i16 %v1, i32 1
  %vecinit2.i = insertelement <8 x i16> %vecinit1.i, i16 %v1, i32 2
  %vecinit3.i = insertelement <8 x i16> %vecinit2.i, i16 %v1, i32 3
  %vecinit4.i = insertelement <8 x i16> %vecinit3.i, i16 %v1, i32 4
  %vecinit5.i = insertelement <8 x i16> %vecinit4.i, i16 %v1, i32 5
  %vecinit6.i = insertelement <8 x i16> %vecinit5.i, i16 %v1, i32 6
  %vecinit7.i = insertelement <8 x i16> %vecinit6.i, i16 %v1, i32 7
  ret <8 x i16> %vecinit7.i
}

define <4 x i32> @test_vdupq_n_u32(i32 %v1) #0 {
; CHECK-LABEL: test_vdupq_n_u32:
; CHECK: dup {{v[0-9]+}}.4s, {{w[0-9]+}}
  %vecinit.i = insertelement <4 x i32> undef, i32 %v1, i32 0
  %vecinit1.i = insertelement <4 x i32> %vecinit.i, i32 %v1, i32 1
  %vecinit2.i = insertelement <4 x i32> %vecinit1.i, i32 %v1, i32 2
  %vecinit3.i = insertelement <4 x i32> %vecinit2.i, i32 %v1, i32 3
  ret <4 x i32> %vecinit3.i
}

define <2 x i64> @test_vdupq_n_u64(i64 %v1) #0 {
; CHECK-LABEL: test_vdupq_n_u64:
; CHECK: dup {{v[0-9]+}}.2d, {{x[0-9]+}}
  %vecinit.i = insertelement <2 x i64> undef, i64 %v1, i32 0
  %vecinit1.i = insertelement <2 x i64> %vecinit.i, i64 %v1, i32 1
  ret <2 x i64> %vecinit1.i
}

define <8 x i8> @test_vdup_lane_s8(<8 x i8> %v1) #0 {
; CHECK-LABEL: test_vdup_lane_s8:
; CHECK: dup {{v[0-9]+}}.8b, {{v[0-9]+}}.b[5]
  %shuffle = shufflevector <8 x i8> %v1, <8 x i8> undef, <8 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <8 x i8> %shuffle
}

define <4 x i16> @test_vdup_lane_s16(<4 x i16> %v1) #0 {
; CHECK-LABEL: test_vdup_lane_s16:
; CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
  %shuffle = shufflevector <4 x i16> %v1, <4 x i16> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  ret <4 x i16> %shuffle
}

define <2 x i32> @test_vdup_lane_s32(<2 x i32> %v1) #0 {
; CHECK-LABEL: test_vdup_lane_s32:
; CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
  %shuffle = shufflevector <2 x i32> %v1, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i32> %shuffle
}

define <16 x i8> @test_vdupq_lane_s8(<8 x i8> %v1) #0 {
; CHECK-LABEL: test_vdupq_lane_s8:
; CHECK: {{v[0-9]+}}.16b, {{v[0-9]+}}.b[5]
  %shuffle = shufflevector <8 x i8> %v1, <8 x i8> undef, <16 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <16 x i8> %shuffle
}

define <8 x i16> @test_vdupq_lane_s16(<4 x i16> %v1) #0 {
; CHECK-LABEL: test_vdupq_lane_s16:
; CHECK: {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
  %shuffle = shufflevector <4 x i16> %v1, <4 x i16> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  ret <8 x i16> %shuffle
}

define <4 x i32> @test_vdupq_lane_s32(<2 x i32> %v1) #0 {
; CHECK-LABEL: test_vdupq_lane_s32:
; CHECK: {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
  %shuffle = shufflevector <2 x i32> %v1, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %shuffle
}

define <2 x i64> @test_vdupq_lane_s64(<1 x i64> %v1) #0 {
; CHECK-LABEL: test_vdupq_lane_s64:
; CHECK: {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
  %shuffle = shufflevector <1 x i64> %v1, <1 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %shuffle
}

define <8 x i8> @test_vdup_laneq_s8(<16 x i8> %v1) #0 {
; CHECK-LABEL: test_vdup_laneq_s8:
; CHECK: dup {{v[0-9]+}}.8b, {{v[0-9]+}}.b[5]
  %shuffle = shufflevector <16 x i8> %v1, <16 x i8> undef, <8 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <8 x i8> %shuffle
}

define <4 x i16> @test_vdup_laneq_s16(<8 x i16> %v1) #0 {
; CHECK-LABEL: test_vdup_laneq_s16:
; CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
  %shuffle = shufflevector <8 x i16> %v1, <8 x i16> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  ret <4 x i16> %shuffle
}

define <2 x i32> @test_vdup_laneq_s32(<4 x i32> %v1) #0 {
; CHECK-LABEL: test_vdup_laneq_s32:
; CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
  %shuffle = shufflevector <4 x i32> %v1, <4 x i32> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i32> %shuffle
}

define <16 x i8> @test_vdupq_laneq_s8(<16 x i8> %v1) #0 {
; CHECK-LABEL: test_vdupq_laneq_s8:
; CHECK: dup {{v[0-9]+}}.16b, {{v[0-9]+}}.b[5]
  %shuffle = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <16 x i8> %shuffle
}

define <8 x i16> @test_vdupq_laneq_s16(<8 x i16> %v1) #0 {
; CHECK-LABEL: test_vdupq_laneq_s16:
; CHECK: {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
  %shuffle = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  ret <8 x i16> %shuffle
}

define <4 x i32> @test_vdupq_laneq_s32(<4 x i32> %v1) #0 {
; CHECK-LABEL: test_vdupq_laneq_s32:
; CHECK: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
  %shuffle = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %shuffle
}

define <2 x i64> @test_vdupq_laneq_s64(<2 x i64> %v1) #0 {
; CHECK-LABEL: test_vdupq_laneq_s64:
; CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
  %shuffle = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %shuffle
}

define i64 @test_bitcastv8i8toi64(<8 x i8> %in) {
; CHECK-LABEL: test_bitcastv8i8toi64:
   %res = bitcast <8 x i8> %in to i64
; CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
   ret i64 %res
}

define i64 @test_bitcastv4i16toi64(<4 x i16> %in) {
; CHECK-LABEL: test_bitcastv4i16toi64:
   %res = bitcast <4 x i16> %in to i64
; CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
   ret i64 %res
}

define i64 @test_bitcastv2i32toi64(<2 x i32> %in) {
; CHECK-LABEL: test_bitcastv2i32toi64:
   %res = bitcast <2 x i32> %in to i64
; CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
   ret i64 %res
}

define i64 @test_bitcastv2f32toi64(<2 x float> %in) {
; CHECK-LABEL: test_bitcastv2f32toi64:
   %res = bitcast <2 x float> %in to i64
; CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
   ret i64 %res
}

define i64 @test_bitcastv1i64toi64(<1 x i64> %in) {
; CHECK-LABEL: test_bitcastv1i64toi64:
   %res = bitcast <1 x i64> %in to i64
; CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
   ret i64 %res
}

define i64 @test_bitcastv1f64toi64(<1 x double> %in) {
; CHECK-LABEL: test_bitcastv1f64toi64:
   %res = bitcast <1 x double> %in to i64
; CHECK: fmov {{x[0-9]+}}, {{d[0-9]+}}
   ret i64 %res
}

define <8 x i8> @test_bitcasti64tov8i8(i64 %in) {
; CHECK-LABEL: test_bitcasti64tov8i8:
   %res = bitcast i64 %in to <8 x i8>
; CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
   ret <8 x i8> %res
}

define <4 x i16> @test_bitcasti64tov4i16(i64 %in) {
; CHECK-LABEL: test_bitcasti64tov4i16:
   %res = bitcast i64 %in to <4 x i16>
; CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
   ret <4 x i16> %res
}

define <2 x i32> @test_bitcasti64tov2i32(i64 %in) {
; CHECK-LABEL: test_bitcasti64tov2i32:
   %res = bitcast i64 %in to <2 x i32>
; CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
   ret <2 x i32> %res
}

define <2 x float> @test_bitcasti64tov2f32(i64 %in) {
; CHECK-LABEL: test_bitcasti64tov2f32:
   %res = bitcast i64 %in to <2 x float>
; CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
   ret <2 x float> %res
}

define <1 x i64> @test_bitcasti64tov1i64(i64 %in) {
; CHECK-LABEL: test_bitcasti64tov1i64:
   %res = bitcast i64 %in to <1 x i64>
; CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
   ret <1 x i64> %res
}

define <1 x double> @test_bitcasti64tov1f64(i64 %in) {
; CHECK-LABEL: test_bitcasti64tov1f64:
   %res = bitcast i64 %in to <1 x double>
; CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
   ret <1 x double> %res
}

define <1 x i64> @test_bitcastv8i8tov1f64(<8 x i8> %a) #0 {
; CHECK-LABEL: test_bitcastv8i8tov1f64:
; CHECK: neg {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK-NEXT: fcvtzs {{[xd][0-9]+}}, {{d[0-9]+}}
  %sub.i = sub <8 x i8> zeroinitializer, %a
  %1 = bitcast <8 x i8> %sub.i to <1 x double>
  %vcvt.i = fptosi <1 x double> %1 to <1 x i64>
  ret <1 x i64> %vcvt.i
}

define <1 x i64> @test_bitcastv4i16tov1f64(<4 x i16> %a) #0 {
; CHECK-LABEL: test_bitcastv4i16tov1f64:
; CHECK: neg {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK-NEXT: fcvtzs {{[dx][0-9]+}}, {{d[0-9]+}}
  %sub.i = sub <4 x i16> zeroinitializer, %a
  %1 = bitcast <4 x i16> %sub.i to <1 x double>
  %vcvt.i = fptosi <1 x double> %1 to <1 x i64>
  ret <1 x i64> %vcvt.i
}

define <1 x i64> @test_bitcastv2i32tov1f64(<2 x i32> %a) #0 {
; CHECK-LABEL: test_bitcastv2i32tov1f64:
; CHECK: neg {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
; CHECK-NEXT: fcvtzs {{[xd][0-9]+}}, {{d[0-9]+}}
  %sub.i = sub <2 x i32> zeroinitializer, %a
  %1 = bitcast <2 x i32> %sub.i to <1 x double>
  %vcvt.i = fptosi <1 x double> %1 to <1 x i64>
  ret <1 x i64> %vcvt.i
}

define <1 x i64> @test_bitcastv1i64tov1f64(<1 x i64> %a) #0 {
; CHECK-LABEL: test_bitcastv1i64tov1f64:
; CHECK: neg {{d[0-9]+}}, {{d[0-9]+}}
; CHECK-NEXT: fcvtzs {{[dx][0-9]+}}, {{d[0-9]+}}
  %sub.i = sub <1 x i64> zeroinitializer, %a
  %1 = bitcast <1 x i64> %sub.i to <1 x double>
  %vcvt.i = fptosi <1 x double> %1 to <1 x i64>
  ret <1 x i64> %vcvt.i
}

define <1 x i64> @test_bitcastv2f32tov1f64(<2 x float> %a) #0 {
; CHECK-LABEL: test_bitcastv2f32tov1f64:
; CHECK: fneg {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
; CHECK-NEXT: fcvtzs {{[xd][0-9]+}}, {{d[0-9]+}}
  %sub.i = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %a
  %1 = bitcast <2 x float> %sub.i to <1 x double>
  %vcvt.i = fptosi <1 x double> %1 to <1 x i64>
  ret <1 x i64> %vcvt.i
}

define <8 x i8> @test_bitcastv1f64tov8i8(<1 x i64> %a) #0 {
; CHECK-LABEL: test_bitcastv1f64tov8i8:
; CHECK: scvtf {{d[0-9]+}}, {{[xd][0-9]+}}
; CHECK-NEXT: neg {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  %vcvt.i = sitofp <1 x i64> %a to <1 x double>
  %1 = bitcast <1 x double> %vcvt.i to <8 x i8>
  %sub.i = sub <8 x i8> zeroinitializer, %1
  ret <8 x i8> %sub.i
}

define <4 x i16> @test_bitcastv1f64tov4i16(<1 x i64> %a) #0 {
; CHECK-LABEL: test_bitcastv1f64tov4i16:
; CHECK: scvtf {{d[0-9]+}}, {{[xd][0-9]+}}
; CHECK-NEXT: neg {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  %vcvt.i = sitofp <1 x i64> %a to <1 x double>
  %1 = bitcast <1 x double> %vcvt.i to <4 x i16>
  %sub.i = sub <4 x i16> zeroinitializer, %1
  ret <4 x i16> %sub.i
}

define <2 x i32> @test_bitcastv1f64tov2i32(<1 x i64> %a) #0 {
; CHECK-LABEL: test_bitcastv1f64tov2i32:
; CHECK: scvtf {{d[0-9]+}}, {{[xd][0-9]+}}
; CHECK-NEXT: neg {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  %vcvt.i = sitofp <1 x i64> %a to <1 x double>
  %1 = bitcast <1 x double> %vcvt.i to <2 x i32>
  %sub.i = sub <2 x i32> zeroinitializer, %1
  ret <2 x i32> %sub.i
}

define <1 x i64> @test_bitcastv1f64tov1i64(<1 x i64> %a) #0 {
; CHECK-LABEL: test_bitcastv1f64tov1i64:
; CHECK: scvtf {{d[0-9]+}}, {{[xd][0-9]+}}
; CHECK-NEXT: neg {{d[0-9]+}}, {{d[0-9]+}}
  %vcvt.i = sitofp <1 x i64> %a to <1 x double>
  %1 = bitcast <1 x double> %vcvt.i to <1 x i64>
  %sub.i = sub <1 x i64> zeroinitializer, %1
  ret <1 x i64> %sub.i
}

define <2 x float> @test_bitcastv1f64tov2f32(<1 x i64> %a) #0 {
; CHECK-LABEL: test_bitcastv1f64tov2f32:
; CHECK: scvtf {{d[0-9]+}}, {{[xd][0-9]+}}
; CHECK-NEXT: fneg {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  %vcvt.i = sitofp <1 x i64> %a to <1 x double>
  %1 = bitcast <1 x double> %vcvt.i to <2 x float>
  %sub.i = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %1
  ret <2 x float> %sub.i
}

; Test insert element into an undef vector
define <8 x i8> @scalar_to_vector.v8i8(i8 %a) {
; CHECK-LABEL: scalar_to_vector.v8i8:
; CHECK: fmov {{s[0-9]+}}, {{w[0-9]+}}
  %b = insertelement <8 x i8> undef, i8 %a, i32 0
  ret <8 x i8> %b
}

define <16 x i8> @scalar_to_vector.v16i8(i8 %a) {
; CHECK-LABEL: scalar_to_vector.v16i8:
; CHECK: fmov {{s[0-9]+}}, {{w[0-9]+}}
  %b = insertelement <16 x i8> undef, i8 %a, i32 0
  ret <16 x i8> %b
}

define <4 x i16> @scalar_to_vector.v4i16(i16 %a) {
; CHECK-LABEL: scalar_to_vector.v4i16:
; CHECK: fmov {{s[0-9]+}}, {{w[0-9]+}}
  %b = insertelement <4 x i16> undef, i16 %a, i32 0
  ret <4 x i16> %b
}

define <8 x i16> @scalar_to_vector.v8i16(i16 %a) {
; CHECK-LABEL: scalar_to_vector.v8i16:
; CHECK: fmov {{s[0-9]+}}, {{w[0-9]+}}
  %b = insertelement <8 x i16> undef, i16 %a, i32 0
  ret <8 x i16> %b
}

define <2 x i32> @scalar_to_vector.v2i32(i32 %a) {
; CHECK-LABEL: scalar_to_vector.v2i32:
; CHECK: fmov {{s[0-9]+}}, {{w[0-9]+}}
  %b = insertelement <2 x i32> undef, i32 %a, i32 0
  ret <2 x i32> %b
}

define <4 x i32> @scalar_to_vector.v4i32(i32 %a) {
; CHECK-LABEL: scalar_to_vector.v4i32:
; CHECK: fmov {{s[0-9]+}}, {{w[0-9]+}}
  %b = insertelement <4 x i32> undef, i32 %a, i32 0
  ret <4 x i32> %b
}

define <2 x i64> @scalar_to_vector.v2i64(i64 %a) {
; CHECK-LABEL: scalar_to_vector.v2i64:
; CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
  %b = insertelement <2 x i64> undef, i64 %a, i32 0
  ret <2 x i64> %b
}

define <8 x i8> @testDUP.v1i8(<1 x i8> %a) {
; CHECK-LABEL: testDUP.v1i8:
; CHECK: dup v0.8b, v0.b[0]
  %b = extractelement <1 x i8> %a, i32 0
  %c = insertelement <8 x i8> undef, i8 %b, i32 0
  %d = insertelement <8 x i8> %c, i8 %b, i32 1
  %e = insertelement <8 x i8> %d, i8 %b, i32 2
  %f = insertelement <8 x i8> %e, i8 %b, i32 3
  %g = insertelement <8 x i8> %f, i8 %b, i32 4
  %h = insertelement <8 x i8> %g, i8 %b, i32 5
  %i = insertelement <8 x i8> %h, i8 %b, i32 6
  %j = insertelement <8 x i8> %i, i8 %b, i32 7
  ret <8 x i8> %j
}

define <8 x i16> @testDUP.v1i16(<1 x i16> %a) {
; CHECK-LABEL: testDUP.v1i16:
; CHECK: dup v0.8h, v0.h[0]
  %b = extractelement <1 x i16> %a, i32 0
  %c = insertelement <8 x i16> undef, i16 %b, i32 0
  %d = insertelement <8 x i16> %c, i16 %b, i32 1
  %e = insertelement <8 x i16> %d, i16 %b, i32 2
  %f = insertelement <8 x i16> %e, i16 %b, i32 3
  %g = insertelement <8 x i16> %f, i16 %b, i32 4
  %h = insertelement <8 x i16> %g, i16 %b, i32 5
  %i = insertelement <8 x i16> %h, i16 %b, i32 6
  %j = insertelement <8 x i16> %i, i16 %b, i32 7
  ret <8 x i16> %j
}

define <4 x i32> @testDUP.v1i32(<1 x i32> %a) {
; CHECK-LABEL: testDUP.v1i32:
; CHECK: dup v0.4s, v0.s[0]
  %b = extractelement <1 x i32> %a, i32 0
  %c = insertelement <4 x i32> undef, i32 %b, i32 0
  %d = insertelement <4 x i32> %c, i32 %b, i32 1
  %e = insertelement <4 x i32> %d, i32 %b, i32 2
  %f = insertelement <4 x i32> %e, i32 %b, i32 3
  ret <4 x i32> %f
}

define <8 x i8> @getl(<16 x i8> %x) #0 {
; CHECK-LABEL: getl:
; CHECK: ret
  %vecext = extractelement <16 x i8> %x, i32 0
  %vecinit = insertelement <8 x i8> undef, i8 %vecext, i32 0
  %vecext1 = extractelement <16 x i8> %x, i32 1
  %vecinit2 = insertelement <8 x i8> %vecinit, i8 %vecext1, i32 1
  %vecext3 = extractelement <16 x i8> %x, i32 2
  %vecinit4 = insertelement <8 x i8> %vecinit2, i8 %vecext3, i32 2
  %vecext5 = extractelement <16 x i8> %x, i32 3
  %vecinit6 = insertelement <8 x i8> %vecinit4, i8 %vecext5, i32 3
  %vecext7 = extractelement <16 x i8> %x, i32 4
  %vecinit8 = insertelement <8 x i8> %vecinit6, i8 %vecext7, i32 4
  %vecext9 = extractelement <16 x i8> %x, i32 5
  %vecinit10 = insertelement <8 x i8> %vecinit8, i8 %vecext9, i32 5
  %vecext11 = extractelement <16 x i8> %x, i32 6
  %vecinit12 = insertelement <8 x i8> %vecinit10, i8 %vecext11, i32 6
  %vecext13 = extractelement <16 x i8> %x, i32 7
  %vecinit14 = insertelement <8 x i8> %vecinit12, i8 %vecext13, i32 7
  ret <8 x i8> %vecinit14
}

define <4 x i16> @test_dup_v2i32_v4i16(<2 x i32> %a) {
; CHECK-LABEL: test_dup_v2i32_v4i16:
; CHECK: dup v0.4h, v0.h[2]
entry:
  %x = extractelement <2 x i32> %a, i32 1
  %vget_lane = trunc i32 %x to i16
  %vecinit.i = insertelement <4 x i16> undef, i16 %vget_lane, i32 0
  %vecinit1.i = insertelement <4 x i16> %vecinit.i, i16 %vget_lane, i32 1
  %vecinit2.i = insertelement <4 x i16> %vecinit1.i, i16 %vget_lane, i32 2
  %vecinit3.i = insertelement <4 x i16> %vecinit2.i, i16 %vget_lane, i32 3
  ret <4 x i16> %vecinit3.i
}

define <8 x i16> @test_dup_v4i32_v8i16(<4 x i32> %a) {
; CHECK-LABEL: test_dup_v4i32_v8i16:
; CHECK: dup v0.8h, v0.h[6]
entry:
  %x = extractelement <4 x i32> %a, i32 3
  %vget_lane = trunc i32 %x to i16
  %vecinit.i = insertelement <8 x i16> undef, i16 %vget_lane, i32 0
  %vecinit1.i = insertelement <8 x i16> %vecinit.i, i16 %vget_lane, i32 1
  %vecinit2.i = insertelement <8 x i16> %vecinit1.i, i16 %vget_lane, i32 2
  %vecinit3.i = insertelement <8 x i16> %vecinit2.i, i16 %vget_lane, i32 3
  %vecinit4.i = insertelement <8 x i16> %vecinit3.i, i16 %vget_lane, i32 4
  %vecinit5.i = insertelement <8 x i16> %vecinit4.i, i16 %vget_lane, i32 5
  %vecinit6.i = insertelement <8 x i16> %vecinit5.i, i16 %vget_lane, i32 6
  %vecinit7.i = insertelement <8 x i16> %vecinit6.i, i16 %vget_lane, i32 7
  ret <8 x i16> %vecinit7.i
}

define <4 x i16> @test_dup_v1i64_v4i16(<1 x i64> %a) {
; CHECK-LABEL: test_dup_v1i64_v4i16:
; CHECK: dup v0.4h, v0.h[0]
entry:
  %x = extractelement <1 x i64> %a, i32 0
  %vget_lane = trunc i64 %x to i16
  %vecinit.i = insertelement <4 x i16> undef, i16 %vget_lane, i32 0
  %vecinit1.i = insertelement <4 x i16> %vecinit.i, i16 %vget_lane, i32 1
  %vecinit2.i = insertelement <4 x i16> %vecinit1.i, i16 %vget_lane, i32 2
  %vecinit3.i = insertelement <4 x i16> %vecinit2.i, i16 %vget_lane, i32 3
  ret <4 x i16> %vecinit3.i
}

define <2 x i32> @test_dup_v1i64_v2i32(<1 x i64> %a) {
; CHECK-LABEL: test_dup_v1i64_v2i32:
; CHECK: dup v0.2s, v0.s[0]
entry:
  %x = extractelement <1 x i64> %a, i32 0
  %vget_lane = trunc i64 %x to i32
  %vecinit.i = insertelement <2 x i32> undef, i32 %vget_lane, i32 0
  %vecinit1.i = insertelement <2 x i32> %vecinit.i, i32 %vget_lane, i32 1
  ret <2 x i32> %vecinit1.i
}

define <8 x i16> @test_dup_v2i64_v8i16(<2 x i64> %a) {
; CHECK-LABEL: test_dup_v2i64_v8i16:
; CHECK: dup v0.8h, v0.h[4]
entry:
  %x = extractelement <2 x i64> %a, i32 1
  %vget_lane = trunc i64 %x to i16
  %vecinit.i = insertelement <8 x i16> undef, i16 %vget_lane, i32 0
  %vecinit1.i = insertelement <8 x i16> %vecinit.i, i16 %vget_lane, i32 1
  %vecinit2.i = insertelement <8 x i16> %vecinit1.i, i16 %vget_lane, i32 2
  %vecinit3.i = insertelement <8 x i16> %vecinit2.i, i16 %vget_lane, i32 3
  %vecinit4.i = insertelement <8 x i16> %vecinit3.i, i16 %vget_lane, i32 4
  %vecinit5.i = insertelement <8 x i16> %vecinit4.i, i16 %vget_lane, i32 5
  %vecinit6.i = insertelement <8 x i16> %vecinit5.i, i16 %vget_lane, i32 6
  %vecinit7.i = insertelement <8 x i16> %vecinit6.i, i16 %vget_lane, i32 7
  ret <8 x i16> %vecinit7.i
}

define <4 x i32> @test_dup_v2i64_v4i32(<2 x i64> %a) {
; CHECK-LABEL: test_dup_v2i64_v4i32:
; CHECK: dup v0.4s, v0.s[2]
entry:
  %x = extractelement <2 x i64> %a, i32 1
  %vget_lane = trunc i64 %x to i32
  %vecinit.i = insertelement <4 x i32> undef, i32 %vget_lane, i32 0
  %vecinit1.i = insertelement <4 x i32> %vecinit.i, i32 %vget_lane, i32 1
  %vecinit2.i = insertelement <4 x i32> %vecinit1.i, i32 %vget_lane, i32 2
  %vecinit3.i = insertelement <4 x i32> %vecinit2.i, i32 %vget_lane, i32 3
  ret <4 x i32> %vecinit3.i
}

define <4 x i16> @test_dup_v4i32_v4i16(<4 x i32> %a) {
; CHECK-LABEL: test_dup_v4i32_v4i16:
; CHECK: dup v0.4h, v0.h[2]
entry:
  %x = extractelement <4 x i32> %a, i32 1
  %vget_lane = trunc i32 %x to i16
  %vecinit.i = insertelement <4 x i16> undef, i16 %vget_lane, i32 0
  %vecinit1.i = insertelement <4 x i16> %vecinit.i, i16 %vget_lane, i32 1
  %vecinit2.i = insertelement <4 x i16> %vecinit1.i, i16 %vget_lane, i32 2
  %vecinit3.i = insertelement <4 x i16> %vecinit2.i, i16 %vget_lane, i32 3
  ret <4 x i16> %vecinit3.i
}

define <4 x i16> @test_dup_v2i64_v4i16(<2 x i64> %a) {
; CHECK-LABEL: test_dup_v2i64_v4i16:
; CHECK: dup v0.4h, v0.h[0]
entry:
  %x = extractelement <2 x i64> %a, i32 0
  %vget_lane = trunc i64 %x to i16
  %vecinit.i = insertelement <4 x i16> undef, i16 %vget_lane, i32 0
  %vecinit1.i = insertelement <4 x i16> %vecinit.i, i16 %vget_lane, i32 1
  %vecinit2.i = insertelement <4 x i16> %vecinit1.i, i16 %vget_lane, i32 2
  %vecinit3.i = insertelement <4 x i16> %vecinit2.i, i16 %vget_lane, i32 3
  ret <4 x i16> %vecinit3.i
}

define <2 x i32> @test_dup_v2i64_v2i32(<2 x i64> %a) {
; CHECK-LABEL: test_dup_v2i64_v2i32:
; CHECK: dup v0.2s, v0.s[0]
entry:
  %x = extractelement <2 x i64> %a, i32 0
  %vget_lane = trunc i64 %x to i32
  %vecinit.i = insertelement <2 x i32> undef, i32 %vget_lane, i32 0
  %vecinit1.i = insertelement <2 x i32> %vecinit.i, i32 %vget_lane, i32 1
  ret <2 x i32> %vecinit1.i
}


define <2 x float> @test_scalar_to_vector_f32_to_v2f32(<2 x float> %a) {
; CHECK-LABEL: test_scalar_to_vector_f32_to_v2f32:
; CHECK: fmaxp s{{[0-9]+}}, v{{[0-9]+}}.2s
; CHECK-NEXT: ret
entry:
  %0 = call float @llvm.aarch64.neon.fmaxv.f32.v2f32(<2 x float> %a)
  %1 = insertelement <1 x float> undef, float %0, i32 0
  %2 = extractelement <1 x float> %1, i32 0
  %vecinit1.i = insertelement <2 x float> undef, float %2, i32 0
  ret <2 x float> %vecinit1.i
}

define <4 x float> @test_scalar_to_vector_f32_to_v4f32(<2 x float> %a) {
; CHECK-LABEL: test_scalar_to_vector_f32_to_v4f32:
; CHECK: fmaxp s{{[0-9]+}}, v{{[0-9]+}}.2s
; CHECK-NEXT: ret
entry:
  %0 = call float @llvm.aarch64.neon.fmaxv.f32.v2f32(<2 x float> %a)
  %1 = insertelement <1 x float> undef, float %0, i32 0
  %2 = extractelement <1 x float> %1, i32 0
  %vecinit1.i = insertelement <4 x float> undef, float %2, i32 0
  ret <4 x float> %vecinit1.i
}

declare float @llvm.aarch64.neon.fmaxv.f32.v2f32(<2 x float>)

define <2 x i32> @test_concat_undef_v1i32(<2 x i32> %a) {
; CHECK-LABEL: test_concat_undef_v1i32:
; CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
entry:
  %0 = extractelement <2 x i32> %a, i32 0
  %vecinit1.i = insertelement <2 x i32> undef, i32 %0, i32 1
  ret <2 x i32> %vecinit1.i
}

declare i32 @llvm.aarch64.neon.sqabs.i32(i32) #4

define <2 x i32> @test_concat_v1i32_undef(i32 %a) {
; CHECK-LABEL: test_concat_v1i32_undef:
; CHECK: sqabs s{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NEXT: ret
entry:
  %b = tail call i32 @llvm.aarch64.neon.sqabs.i32(i32 %a)
  %vecinit.i432 = insertelement <2 x i32> undef, i32 %b, i32 0
  ret <2 x i32> %vecinit.i432
}

define <2 x i32> @test_concat_same_v1i32_v1i32(<2 x i32> %a) {
; CHECK-LABEL: test_concat_same_v1i32_v1i32:
; CHECK: dup v{{[0-9]+}}.2s, v{{[0-9]+}}.s[0]
entry:
  %0 = extractelement <2 x i32> %a, i32 0
  %vecinit.i = insertelement <2 x i32> undef, i32 %0, i32 0
  %vecinit1.i = insertelement <2 x i32> %vecinit.i, i32 %0, i32 1
  ret <2 x i32> %vecinit1.i
}

define <2 x i32> @test_concat_diff_v1i32_v1i32(i32 %a, i32 %b) {
; CHECK-LABEL: test_concat_diff_v1i32_v1i32:
; CHECK: sqabs s{{[0-9]+}}, s{{[0-9]+}}
; CHECK: sqabs s{{[0-9]+}}, s{{[0-9]+}}
; CHECK: ins {{v[0-9]+}}.s[1], w{{[0-9]+}}
entry:
  %c = tail call i32 @llvm.aarch64.neon.sqabs.i32(i32 %a)
  %d = insertelement <2 x i32> undef, i32 %c, i32 0
  %e = tail call i32 @llvm.aarch64.neon.sqabs.i32(i32 %b)
  %f = insertelement <2 x i32> undef, i32 %e, i32 0
  %h = shufflevector <2 x i32> %d, <2 x i32> %f, <2 x i32> <i32 0, i32 2>
  ret <2 x i32> %h
}

define <16 x i8> @test_concat_v16i8_v16i8_v16i8(<16 x i8> %x, <16 x i8> %y) #0 {
; CHECK-LABEL: test_concat_v16i8_v16i8_v16i8:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecinit30 = shufflevector <16 x i8> %x, <16 x i8> %y, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  ret <16 x i8> %vecinit30
}

define <16 x i8> @test_concat_v16i8_v8i8_v16i8(<8 x i8> %x, <16 x i8> %y) #0 {
; CHECK-LABEL: test_concat_v16i8_v8i8_v16i8:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecext = extractelement <8 x i8> %x, i32 0
  %vecinit = insertelement <16 x i8> undef, i8 %vecext, i32 0
  %vecext1 = extractelement <8 x i8> %x, i32 1
  %vecinit2 = insertelement <16 x i8> %vecinit, i8 %vecext1, i32 1
  %vecext3 = extractelement <8 x i8> %x, i32 2
  %vecinit4 = insertelement <16 x i8> %vecinit2, i8 %vecext3, i32 2
  %vecext5 = extractelement <8 x i8> %x, i32 3
  %vecinit6 = insertelement <16 x i8> %vecinit4, i8 %vecext5, i32 3
  %vecext7 = extractelement <8 x i8> %x, i32 4
  %vecinit8 = insertelement <16 x i8> %vecinit6, i8 %vecext7, i32 4
  %vecext9 = extractelement <8 x i8> %x, i32 5
  %vecinit10 = insertelement <16 x i8> %vecinit8, i8 %vecext9, i32 5
  %vecext11 = extractelement <8 x i8> %x, i32 6
  %vecinit12 = insertelement <16 x i8> %vecinit10, i8 %vecext11, i32 6
  %vecext13 = extractelement <8 x i8> %x, i32 7
  %vecinit14 = insertelement <16 x i8> %vecinit12, i8 %vecext13, i32 7
  %vecinit30 = shufflevector <16 x i8> %vecinit14, <16 x i8> %y, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  ret <16 x i8> %vecinit30
}

define <16 x i8> @test_concat_v16i8_v16i8_v8i8(<16 x i8> %x, <8 x i8> %y) #0 {
; CHECK-LABEL: test_concat_v16i8_v16i8_v8i8:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecext = extractelement <16 x i8> %x, i32 0
  %vecinit = insertelement <16 x i8> undef, i8 %vecext, i32 0
  %vecext1 = extractelement <16 x i8> %x, i32 1
  %vecinit2 = insertelement <16 x i8> %vecinit, i8 %vecext1, i32 1
  %vecext3 = extractelement <16 x i8> %x, i32 2
  %vecinit4 = insertelement <16 x i8> %vecinit2, i8 %vecext3, i32 2
  %vecext5 = extractelement <16 x i8> %x, i32 3
  %vecinit6 = insertelement <16 x i8> %vecinit4, i8 %vecext5, i32 3
  %vecext7 = extractelement <16 x i8> %x, i32 4
  %vecinit8 = insertelement <16 x i8> %vecinit6, i8 %vecext7, i32 4
  %vecext9 = extractelement <16 x i8> %x, i32 5
  %vecinit10 = insertelement <16 x i8> %vecinit8, i8 %vecext9, i32 5
  %vecext11 = extractelement <16 x i8> %x, i32 6
  %vecinit12 = insertelement <16 x i8> %vecinit10, i8 %vecext11, i32 6
  %vecext13 = extractelement <16 x i8> %x, i32 7
  %vecinit14 = insertelement <16 x i8> %vecinit12, i8 %vecext13, i32 7
  %vecext15 = extractelement <8 x i8> %y, i32 0
  %vecinit16 = insertelement <16 x i8> %vecinit14, i8 %vecext15, i32 8
  %vecext17 = extractelement <8 x i8> %y, i32 1
  %vecinit18 = insertelement <16 x i8> %vecinit16, i8 %vecext17, i32 9
  %vecext19 = extractelement <8 x i8> %y, i32 2
  %vecinit20 = insertelement <16 x i8> %vecinit18, i8 %vecext19, i32 10
  %vecext21 = extractelement <8 x i8> %y, i32 3
  %vecinit22 = insertelement <16 x i8> %vecinit20, i8 %vecext21, i32 11
  %vecext23 = extractelement <8 x i8> %y, i32 4
  %vecinit24 = insertelement <16 x i8> %vecinit22, i8 %vecext23, i32 12
  %vecext25 = extractelement <8 x i8> %y, i32 5
  %vecinit26 = insertelement <16 x i8> %vecinit24, i8 %vecext25, i32 13
  %vecext27 = extractelement <8 x i8> %y, i32 6
  %vecinit28 = insertelement <16 x i8> %vecinit26, i8 %vecext27, i32 14
  %vecext29 = extractelement <8 x i8> %y, i32 7
  %vecinit30 = insertelement <16 x i8> %vecinit28, i8 %vecext29, i32 15
  ret <16 x i8> %vecinit30
}

define <16 x i8> @test_concat_v16i8_v8i8_v8i8(<8 x i8> %x, <8 x i8> %y) #0 {
; CHECK-LABEL: test_concat_v16i8_v8i8_v8i8:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecext = extractelement <8 x i8> %x, i32 0
  %vecinit = insertelement <16 x i8> undef, i8 %vecext, i32 0
  %vecext1 = extractelement <8 x i8> %x, i32 1
  %vecinit2 = insertelement <16 x i8> %vecinit, i8 %vecext1, i32 1
  %vecext3 = extractelement <8 x i8> %x, i32 2
  %vecinit4 = insertelement <16 x i8> %vecinit2, i8 %vecext3, i32 2
  %vecext5 = extractelement <8 x i8> %x, i32 3
  %vecinit6 = insertelement <16 x i8> %vecinit4, i8 %vecext5, i32 3
  %vecext7 = extractelement <8 x i8> %x, i32 4
  %vecinit8 = insertelement <16 x i8> %vecinit6, i8 %vecext7, i32 4
  %vecext9 = extractelement <8 x i8> %x, i32 5
  %vecinit10 = insertelement <16 x i8> %vecinit8, i8 %vecext9, i32 5
  %vecext11 = extractelement <8 x i8> %x, i32 6
  %vecinit12 = insertelement <16 x i8> %vecinit10, i8 %vecext11, i32 6
  %vecext13 = extractelement <8 x i8> %x, i32 7
  %vecinit14 = insertelement <16 x i8> %vecinit12, i8 %vecext13, i32 7
  %vecext15 = extractelement <8 x i8> %y, i32 0
  %vecinit16 = insertelement <16 x i8> %vecinit14, i8 %vecext15, i32 8
  %vecext17 = extractelement <8 x i8> %y, i32 1
  %vecinit18 = insertelement <16 x i8> %vecinit16, i8 %vecext17, i32 9
  %vecext19 = extractelement <8 x i8> %y, i32 2
  %vecinit20 = insertelement <16 x i8> %vecinit18, i8 %vecext19, i32 10
  %vecext21 = extractelement <8 x i8> %y, i32 3
  %vecinit22 = insertelement <16 x i8> %vecinit20, i8 %vecext21, i32 11
  %vecext23 = extractelement <8 x i8> %y, i32 4
  %vecinit24 = insertelement <16 x i8> %vecinit22, i8 %vecext23, i32 12
  %vecext25 = extractelement <8 x i8> %y, i32 5
  %vecinit26 = insertelement <16 x i8> %vecinit24, i8 %vecext25, i32 13
  %vecext27 = extractelement <8 x i8> %y, i32 6
  %vecinit28 = insertelement <16 x i8> %vecinit26, i8 %vecext27, i32 14
  %vecext29 = extractelement <8 x i8> %y, i32 7
  %vecinit30 = insertelement <16 x i8> %vecinit28, i8 %vecext29, i32 15
  ret <16 x i8> %vecinit30
}

define <8 x i16> @test_concat_v8i16_v8i16_v8i16(<8 x i16> %x, <8 x i16> %y) #0 {
; CHECK-LABEL: test_concat_v8i16_v8i16_v8i16:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecinit14 = shufflevector <8 x i16> %x, <8 x i16> %y, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  ret <8 x i16> %vecinit14
}

define <8 x i16> @test_concat_v8i16_v4i16_v8i16(<4 x i16> %x, <8 x i16> %y) #0 {
; CHECK-LABEL: test_concat_v8i16_v4i16_v8i16:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecext = extractelement <4 x i16> %x, i32 0
  %vecinit = insertelement <8 x i16> undef, i16 %vecext, i32 0
  %vecext1 = extractelement <4 x i16> %x, i32 1
  %vecinit2 = insertelement <8 x i16> %vecinit, i16 %vecext1, i32 1
  %vecext3 = extractelement <4 x i16> %x, i32 2
  %vecinit4 = insertelement <8 x i16> %vecinit2, i16 %vecext3, i32 2
  %vecext5 = extractelement <4 x i16> %x, i32 3
  %vecinit6 = insertelement <8 x i16> %vecinit4, i16 %vecext5, i32 3
  %vecinit14 = shufflevector <8 x i16> %vecinit6, <8 x i16> %y, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  ret <8 x i16> %vecinit14
}

define <8 x i16> @test_concat_v8i16_v8i16_v4i16(<8 x i16> %x, <4 x i16> %y) #0 {
; CHECK-LABEL: test_concat_v8i16_v8i16_v4i16:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecext = extractelement <8 x i16> %x, i32 0
  %vecinit = insertelement <8 x i16> undef, i16 %vecext, i32 0
  %vecext1 = extractelement <8 x i16> %x, i32 1
  %vecinit2 = insertelement <8 x i16> %vecinit, i16 %vecext1, i32 1
  %vecext3 = extractelement <8 x i16> %x, i32 2
  %vecinit4 = insertelement <8 x i16> %vecinit2, i16 %vecext3, i32 2
  %vecext5 = extractelement <8 x i16> %x, i32 3
  %vecinit6 = insertelement <8 x i16> %vecinit4, i16 %vecext5, i32 3
  %vecext7 = extractelement <4 x i16> %y, i32 0
  %vecinit8 = insertelement <8 x i16> %vecinit6, i16 %vecext7, i32 4
  %vecext9 = extractelement <4 x i16> %y, i32 1
  %vecinit10 = insertelement <8 x i16> %vecinit8, i16 %vecext9, i32 5
  %vecext11 = extractelement <4 x i16> %y, i32 2
  %vecinit12 = insertelement <8 x i16> %vecinit10, i16 %vecext11, i32 6
  %vecext13 = extractelement <4 x i16> %y, i32 3
  %vecinit14 = insertelement <8 x i16> %vecinit12, i16 %vecext13, i32 7
  ret <8 x i16> %vecinit14
}

define <8 x i16> @test_concat_v8i16_v4i16_v4i16(<4 x i16> %x, <4 x i16> %y) #0 {
; CHECK-LABEL: test_concat_v8i16_v4i16_v4i16:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecext = extractelement <4 x i16> %x, i32 0
  %vecinit = insertelement <8 x i16> undef, i16 %vecext, i32 0
  %vecext1 = extractelement <4 x i16> %x, i32 1
  %vecinit2 = insertelement <8 x i16> %vecinit, i16 %vecext1, i32 1
  %vecext3 = extractelement <4 x i16> %x, i32 2
  %vecinit4 = insertelement <8 x i16> %vecinit2, i16 %vecext3, i32 2
  %vecext5 = extractelement <4 x i16> %x, i32 3
  %vecinit6 = insertelement <8 x i16> %vecinit4, i16 %vecext5, i32 3
  %vecext7 = extractelement <4 x i16> %y, i32 0
  %vecinit8 = insertelement <8 x i16> %vecinit6, i16 %vecext7, i32 4
  %vecext9 = extractelement <4 x i16> %y, i32 1
  %vecinit10 = insertelement <8 x i16> %vecinit8, i16 %vecext9, i32 5
  %vecext11 = extractelement <4 x i16> %y, i32 2
  %vecinit12 = insertelement <8 x i16> %vecinit10, i16 %vecext11, i32 6
  %vecext13 = extractelement <4 x i16> %y, i32 3
  %vecinit14 = insertelement <8 x i16> %vecinit12, i16 %vecext13, i32 7
  ret <8 x i16> %vecinit14
}

define <4 x i32> @test_concat_v4i32_v4i32_v4i32(<4 x i32> %x, <4 x i32> %y) #0 {
; CHECK-LABEL: test_concat_v4i32_v4i32_v4i32:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecinit6 = shufflevector <4 x i32> %x, <4 x i32> %y, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  ret <4 x i32> %vecinit6
}

define <4 x i32> @test_concat_v4i32_v2i32_v4i32(<2 x i32> %x, <4 x i32> %y) #0 {
; CHECK-LABEL: test_concat_v4i32_v2i32_v4i32:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecext = extractelement <2 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecext1 = extractelement <2 x i32> %x, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %vecext1, i32 1
  %vecinit6 = shufflevector <4 x i32> %vecinit2, <4 x i32> %y, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  ret <4 x i32> %vecinit6
}

define <4 x i32> @test_concat_v4i32_v4i32_v2i32(<4 x i32> %x, <2 x i32> %y) #0 {
; CHECK-LABEL: test_concat_v4i32_v4i32_v2i32:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecext = extractelement <4 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecext1 = extractelement <4 x i32> %x, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %vecext1, i32 1
  %vecext3 = extractelement <2 x i32> %y, i32 0
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %vecext3, i32 2
  %vecext5 = extractelement <2 x i32> %y, i32 1
  %vecinit6 = insertelement <4 x i32> %vecinit4, i32 %vecext5, i32 3
  ret <4 x i32> %vecinit6
}

define <4 x i32> @test_concat_v4i32_v2i32_v2i32(<2 x i32> %x, <2 x i32> %y) #0 {
; CHECK-LABEL: test_concat_v4i32_v2i32_v2i32:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecinit6 = shufflevector <2 x i32> %x, <2 x i32> %y, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %vecinit6
}

define <2 x i64> @test_concat_v2i64_v2i64_v2i64(<2 x i64> %x, <2 x i64> %y) #0 {
; CHECK-LABEL: test_concat_v2i64_v2i64_v2i64:
; CHECK: zip1 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vecinit2 = shufflevector <2 x i64> %x, <2 x i64> %y, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %vecinit2
}

define <2 x i64> @test_concat_v2i64_v1i64_v2i64(<1 x i64> %x, <2 x i64> %y) #0 {
; CHECK-LABEL: test_concat_v2i64_v1i64_v2i64:
; CHECK: zip1 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vecext = extractelement <1 x i64> %x, i32 0
  %vecinit = insertelement <2 x i64> undef, i64 %vecext, i32 0
  %vecinit2 = shufflevector <2 x i64> %vecinit, <2 x i64> %y, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %vecinit2
}

define <2 x i64> @test_concat_v2i64_v2i64_v1i64(<2 x i64> %x, <1 x i64> %y) #0 {
; CHECK-LABEL: test_concat_v2i64_v2i64_v1i64:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecext = extractelement <2 x i64> %x, i32 0
  %vecinit = insertelement <2 x i64> undef, i64 %vecext, i32 0
  %vecext1 = extractelement <1 x i64> %y, i32 0
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %vecext1, i32 1
  ret <2 x i64> %vecinit2
}

define <2 x i64> @test_concat_v2i64_v1i64_v1i64(<1 x i64> %x, <1 x i64> %y) #0 {
; CHECK-LABEL: test_concat_v2i64_v1i64_v1i64:
; CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
entry:
  %vecext = extractelement <1 x i64> %x, i32 0
  %vecinit = insertelement <2 x i64> undef, i64 %vecext, i32 0
  %vecext1 = extractelement <1 x i64> %y, i32 0
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %vecext1, i32 1
  ret <2 x i64> %vecinit2
}


define <4 x i16> @concat_vector_v4i16_const() {
; CHECK-LABEL: concat_vector_v4i16_const:
; CHECK: movi {{d[0-9]+}}, #0
 %r = shufflevector <1 x i16> zeroinitializer, <1 x i16> undef, <4 x i32> zeroinitializer
 ret <4 x i16> %r
}

define <4 x i16> @concat_vector_v4i16_const_one() {
; CHECK-LABEL: concat_vector_v4i16_const_one:
; CHECK: movi {{v[0-9]+}}.4h, #0x1
 %r = shufflevector <1 x i16> <i16 1>, <1 x i16> undef, <4 x i32> zeroinitializer
 ret <4 x i16> %r
}

define <4 x i32> @concat_vector_v4i32_const() {
; CHECK-LABEL: concat_vector_v4i32_const:
; CHECK: movi {{v[0-9]+}}.2d, #0
 %r = shufflevector <1 x i32> zeroinitializer, <1 x i32> undef, <4 x i32> zeroinitializer
 ret <4 x i32> %r
}

define <8 x i8> @concat_vector_v8i8_const() {
; CHECK-LABEL: concat_vector_v8i8_const:
; CHECK: movi {{d[0-9]+}}, #0
 %r = shufflevector <1 x i8> zeroinitializer, <1 x i8> undef, <8 x i32> zeroinitializer
 ret <8 x i8> %r
}

define <8 x i16> @concat_vector_v8i16_const() {
; CHECK-LABEL: concat_vector_v8i16_const:
; CHECK: movi {{v[0-9]+}}.2d, #0
 %r = shufflevector <1 x i16> zeroinitializer, <1 x i16> undef, <8 x i32> zeroinitializer
 ret <8 x i16> %r
}

define <8 x i16> @concat_vector_v8i16_const_one() {
; CHECK-LABEL: concat_vector_v8i16_const_one:
; CHECK: movi {{v[0-9]+}}.8h, #0x1
 %r = shufflevector <1 x i16> <i16 1>, <1 x i16> undef, <8 x i32> zeroinitializer
 ret <8 x i16> %r
}

define <16 x i8> @concat_vector_v16i8_const() {
; CHECK-LABEL: concat_vector_v16i8_const:
; CHECK: movi {{v[0-9]+}}.2d, #0
 %r = shufflevector <1 x i8> zeroinitializer, <1 x i8> undef, <16 x i32> zeroinitializer
 ret <16 x i8> %r
}

define <4 x i16> @concat_vector_v4i16(<1 x i16> %a) {
; CHECK-LABEL: concat_vector_v4i16:
; CHECK: dup v0.4h, v0.h[0]
 %r = shufflevector <1 x i16> %a, <1 x i16> undef, <4 x i32> zeroinitializer
 ret <4 x i16> %r
}

define <4 x i32> @concat_vector_v4i32(<1 x i32> %a) {
; CHECK-LABEL: concat_vector_v4i32:
; CHECK: dup v0.4s, v0.s[0]
 %r = shufflevector <1 x i32> %a, <1 x i32> undef, <4 x i32> zeroinitializer
 ret <4 x i32> %r
}

define <8 x i8> @concat_vector_v8i8(<1 x i8> %a) {
; CHECK-LABEL: concat_vector_v8i8:
; CHECK: dup v0.8b, v0.b[0]
 %r = shufflevector <1 x i8> %a, <1 x i8> undef, <8 x i32> zeroinitializer
 ret <8 x i8> %r
}

define <8 x i16> @concat_vector_v8i16(<1 x i16> %a) {
; CHECK-LABEL: concat_vector_v8i16:
; CHECK: dup v0.8h, v0.h[0]
 %r = shufflevector <1 x i16> %a, <1 x i16> undef, <8 x i32> zeroinitializer
 ret <8 x i16> %r
}

define <16 x i8> @concat_vector_v16i8(<1 x i8> %a) {
; CHECK-LABEL: concat_vector_v16i8:
; CHECK: dup v0.16b, v0.b[0]
 %r = shufflevector <1 x i8> %a, <1 x i8> undef, <16 x i32> zeroinitializer
 ret <16 x i8> %r
}
