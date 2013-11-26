; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s


define <16 x i8> @ins16bw(<16 x i8> %tmp1, i8 %tmp2) {
;CHECK: ins {{v[0-31]+}}.b[15], {{w[0-31]+}}
  %tmp3 = insertelement <16 x i8> %tmp1, i8 %tmp2, i32 15
  ret <16 x i8> %tmp3
}

define <8 x i16> @ins8hw(<8 x i16> %tmp1, i16 %tmp2) {
;CHECK: ins {{v[0-31]+}}.h[6], {{w[0-31]+}}
  %tmp3 = insertelement <8 x i16> %tmp1, i16 %tmp2, i32 6
  ret <8 x i16> %tmp3
}

define <4 x i32> @ins4sw(<4 x i32> %tmp1, i32 %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[2], {{w[0-31]+}}
  %tmp3 = insertelement <4 x i32> %tmp1, i32 %tmp2, i32 2
  ret <4 x i32> %tmp3
}

define <2 x i64> @ins2dw(<2 x i64> %tmp1, i64 %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[1], {{x[0-31]+}}
  %tmp3 = insertelement <2 x i64> %tmp1, i64 %tmp2, i32 1
  ret <2 x i64> %tmp3
}

define <8 x i8> @ins8bw(<8 x i8> %tmp1, i8 %tmp2) {
;CHECK: ins {{v[0-31]+}}.b[5], {{w[0-31]+}}
  %tmp3 = insertelement <8 x i8> %tmp1, i8 %tmp2, i32 5
  ret <8 x i8> %tmp3
}

define <4 x i16> @ins4hw(<4 x i16> %tmp1, i16 %tmp2) {
;CHECK: ins {{v[0-31]+}}.h[3], {{w[0-31]+}}
  %tmp3 = insertelement <4 x i16> %tmp1, i16 %tmp2, i32 3
  ret <4 x i16> %tmp3
}

define <2 x i32> @ins2sw(<2 x i32> %tmp1, i32 %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{w[0-31]+}}
  %tmp3 = insertelement <2 x i32> %tmp1, i32 %tmp2, i32 1
  ret <2 x i32> %tmp3
}

define <16 x i8> @ins16b16(<16 x i8> %tmp1, <16 x i8> %tmp2) {
;CHECK: ins {{v[0-31]+}}.b[15], {{v[0-31]+}}.b[2]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 2
  %tmp4 = insertelement <16 x i8> %tmp2, i8 %tmp3, i32 15
  ret <16 x i8> %tmp4
}

define <8 x i16> @ins8h8(<8 x i16> %tmp1, <8 x i16> %tmp2) {
;CHECK: ins {{v[0-31]+}}.h[7], {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = insertelement <8 x i16> %tmp2, i16 %tmp3, i32 7
  ret <8 x i16> %tmp4
}

define <4 x i32> @ins4s4(<4 x i32> %tmp1, <4 x i32> %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{v[0-31]+}}.s[2]
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  %tmp4 = insertelement <4 x i32> %tmp2, i32 %tmp3, i32 1
  ret <4 x i32> %tmp4
}

define <2 x i64> @ins2d2(<2 x i64> %tmp1, <2 x i64> %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[1], {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <2 x i64> %tmp1, i32 0
  %tmp4 = insertelement <2 x i64> %tmp2, i64 %tmp3, i32 1
  ret <2 x i64> %tmp4
}

define <4 x float> @ins4f4(<4 x float> %tmp1, <4 x float> %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{v[0-31]+}}.s[2]
  %tmp3 = extractelement <4 x float> %tmp1, i32 2
  %tmp4 = insertelement <4 x float> %tmp2, float %tmp3, i32 1
  ret <4 x float> %tmp4
}

define <2 x double> @ins2df2(<2 x double> %tmp1, <2 x double> %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[1], {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <2 x double> %tmp1, i32 0
  %tmp4 = insertelement <2 x double> %tmp2, double %tmp3, i32 1
  ret <2 x double> %tmp4
}

define <16 x i8> @ins8b16(<8 x i8> %tmp1, <16 x i8> %tmp2) {
;CHECK: ins {{v[0-31]+}}.b[15], {{v[0-31]+}}.b[2]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 2
  %tmp4 = insertelement <16 x i8> %tmp2, i8 %tmp3, i32 15
  ret <16 x i8> %tmp4
}

define <8 x i16> @ins4h8(<4 x i16> %tmp1, <8 x i16> %tmp2) {
;CHECK: ins {{v[0-31]+}}.h[7], {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = insertelement <8 x i16> %tmp2, i16 %tmp3, i32 7
  ret <8 x i16> %tmp4
}

define <4 x i32> @ins2s4(<2 x i32> %tmp1, <4 x i32> %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{v[0-31]+}}.s[1]
  %tmp3 = extractelement <2 x i32> %tmp1, i32 1
  %tmp4 = insertelement <4 x i32> %tmp2, i32 %tmp3, i32 1
  ret <4 x i32> %tmp4
}

define <2 x i64> @ins1d2(<1 x i64> %tmp1, <2 x i64> %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[1], {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <1 x i64> %tmp1, i32 0
  %tmp4 = insertelement <2 x i64> %tmp2, i64 %tmp3, i32 1
  ret <2 x i64> %tmp4
}

define <4 x float> @ins2f4(<2 x float> %tmp1, <4 x float> %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{v[0-31]+}}.s[1]
  %tmp3 = extractelement <2 x float> %tmp1, i32 1
  %tmp4 = insertelement <4 x float> %tmp2, float %tmp3, i32 1
  ret <4 x float> %tmp4
}

define <2 x double> @ins1f2(<1 x double> %tmp1, <2 x double> %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[1], {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <1 x double> %tmp1, i32 0
  %tmp4 = insertelement <2 x double> %tmp2, double %tmp3, i32 1
  ret <2 x double> %tmp4
}

define <8 x i8> @ins16b8(<16 x i8> %tmp1, <8 x i8> %tmp2) {
;CHECK: ins {{v[0-31]+}}.b[7], {{v[0-31]+}}.b[2]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 2
  %tmp4 = insertelement <8 x i8> %tmp2, i8 %tmp3, i32 7
  ret <8 x i8> %tmp4
}

define <4 x i16> @ins8h4(<8 x i16> %tmp1, <4 x i16> %tmp2) {
;CHECK: ins {{v[0-31]+}}.h[3], {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = insertelement <4 x i16> %tmp2, i16 %tmp3, i32 3
  ret <4 x i16> %tmp4
}

define <2 x i32> @ins4s2(<4 x i32> %tmp1, <2 x i32> %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{v[0-31]+}}.s[2]
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  %tmp4 = insertelement <2 x i32> %tmp2, i32 %tmp3, i32 1
  ret <2 x i32> %tmp4
}

define <1 x i64> @ins2d1(<2 x i64> %tmp1, <1 x i64> %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[0], {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <2 x i64> %tmp1, i32 0
  %tmp4 = insertelement <1 x i64> %tmp2, i64 %tmp3, i32 0
  ret <1 x i64> %tmp4
}

define <2 x float> @ins4f2(<4 x float> %tmp1, <2 x float> %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{v[0-31]+}}.s[2]
  %tmp3 = extractelement <4 x float> %tmp1, i32 2
  %tmp4 = insertelement <2 x float> %tmp2, float %tmp3, i32 1
  ret <2 x float> %tmp4
}

define <1 x double> @ins2f1(<2 x double> %tmp1, <1 x double> %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[0], {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <2 x double> %tmp1, i32 0
  %tmp4 = insertelement <1 x double> %tmp2, double %tmp3, i32 0
  ret <1 x double> %tmp4
}

define <8 x i8> @ins8b8(<8 x i8> %tmp1, <8 x i8> %tmp2) {
;CHECK: ins {{v[0-31]+}}.b[4], {{v[0-31]+}}.b[2]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 2
  %tmp4 = insertelement <8 x i8> %tmp2, i8 %tmp3, i32 4
  ret <8 x i8> %tmp4
}

define <4 x i16> @ins4h4(<4 x i16> %tmp1, <4 x i16> %tmp2) {
;CHECK: ins {{v[0-31]+}}.h[3], {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = insertelement <4 x i16> %tmp2, i16 %tmp3, i32 3
  ret <4 x i16> %tmp4
}

define <2 x i32> @ins2s2(<2 x i32> %tmp1, <2 x i32> %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{v[0-31]+}}.s[0]
  %tmp3 = extractelement <2 x i32> %tmp1, i32 0
  %tmp4 = insertelement <2 x i32> %tmp2, i32 %tmp3, i32 1
  ret <2 x i32> %tmp4
}

define <1 x i64> @ins1d1(<1 x i64> %tmp1, <1 x i64> %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[0], {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <1 x i64> %tmp1, i32 0
  %tmp4 = insertelement <1 x i64> %tmp2, i64 %tmp3, i32 0
  ret <1 x i64> %tmp4
}

define <2 x float> @ins2f2(<2 x float> %tmp1, <2 x float> %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{v[0-31]+}}.s[0]
  %tmp3 = extractelement <2 x float> %tmp1, i32 0
  %tmp4 = insertelement <2 x float> %tmp2, float %tmp3, i32 1
  ret <2 x float> %tmp4
}

define <1 x double> @ins1df1(<1 x double> %tmp1, <1 x double> %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[0], {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <1 x double> %tmp1, i32 0
  %tmp4 = insertelement <1 x double> %tmp2, double %tmp3, i32 0
  ret <1 x double> %tmp4
}

define i32 @umovw16b(<16 x i8> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.b[8]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 8
  %tmp4 = zext i8 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw8h(<8 x i16> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = zext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw4s(<4 x i32> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.s[2]
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  ret i32 %tmp3
}

define i64 @umovx2d(<2 x i64> %tmp1) {
;CHECK: umov {{x[0-31]+}}, {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <2 x i64> %tmp1, i32 0
  ret i64 %tmp3
}

define i32 @umovw8b(<8 x i8> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.b[7]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 7
  %tmp4 = zext i8 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw4h(<4 x i16> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = zext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw2s(<2 x i32> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.s[1]
  %tmp3 = extractelement <2 x i32> %tmp1, i32 1
  ret i32 %tmp3
}

define i64 @umovx1d(<1 x i64> %tmp1) {
;CHECK: fmov {{x[0-31]+}}, {{d[0-31]+}}
  %tmp3 = extractelement <1 x i64> %tmp1, i32 0
  ret i64 %tmp3
}

define i32 @smovw16b(<16 x i8> %tmp1) {
;CHECK: smov {{w[0-31]+}}, {{v[0-31]+}}.b[8]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 8
  %tmp4 = sext i8 %tmp3 to i32
  %tmp5 = add i32 5, %tmp4
  ret i32 %tmp5
}

define i32 @smovw8h(<8 x i16> %tmp1) {
;CHECK: smov {{w[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  %tmp5 = add i32 5, %tmp4
  ret i32 %tmp5
}

define i32 @smovx16b(<16 x i8> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.b[8]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 8
  %tmp4 = sext i8 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @smovx8h(<8 x i16> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i64 @smovx4s(<4 x i32> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.s[2]
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  %tmp4 = sext i32 %tmp3 to i64
  ret i64 %tmp4
}

define i32 @smovw8b(<8 x i8> %tmp1) {
;CHECK: smov {{w[0-31]+}}, {{v[0-31]+}}.b[4]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 4
  %tmp4 = sext i8 %tmp3 to i32
  %tmp5 = add i32 5, %tmp4
  ret i32 %tmp5
}

define i32 @smovw4h(<4 x i16> %tmp1) {
;CHECK: smov {{w[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  %tmp5 = add i32 5, %tmp4
  ret i32 %tmp5
}

define i32 @smovx8b(<8 x i8> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.b[6]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 6
  %tmp4 = sext i8 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @smovx4h(<4 x i16> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i64 @smovx2s(<2 x i32> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.s[1]
  %tmp3 = extractelement <2 x i32> %tmp1, i32 1
  %tmp4 = sext i32 %tmp3 to i64
  ret i64 %tmp4
}

define <8 x i8> @test_vcopy_lane_s8(<8 x i8> %v1, <8 x i8> %v2) {
;CHECK: ins  {{v[0-9]+}}.b[5], {{v[0-9]+}}.b[3]
  %vset_lane = shufflevector <8 x i8> %v1, <8 x i8> %v2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 11, i32 6, i32 7>
  ret <8 x i8> %vset_lane
}

define <16 x i8> @test_vcopyq_laneq_s8(<16 x i8> %v1, <16 x i8> %v2) {
;CHECK: ins  {{v[0-9]+}}.b[14], {{v[0-9]+}}.b[6]
  %vset_lane = shufflevector <16 x i8> %v1, <16 x i8> %v2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 22, i32 15>
  ret <16 x i8> %vset_lane
}

define <8 x i8> @test_vcopy_lane_swap_s8(<8 x i8> %v1, <8 x i8> %v2) {
;CHECK: ins {{v[0-9]+}}.b[7], {{v[0-9]+}}.b[0]
  %vset_lane = shufflevector <8 x i8> %v1, <8 x i8> %v2, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 0>
  ret <8 x i8> %vset_lane
}

define <16 x i8> @test_vcopyq_laneq_swap_s8(<16 x i8> %v1, <16 x i8> %v2) {
;CHECK: ins {{v[0-9]+}}.b[0], {{v[0-9]+}}.b[15]
  %vset_lane = shufflevector <16 x i8> %v1, <16 x i8> %v2, <16 x i32> <i32 15, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vset_lane
}

define <8 x i8> @test_vdup_n_u8(i8 %v1) #0 {
;CHECK: dup {{v[0-9]+}}.8b, {{w[0-9]+}}
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
;CHECK: dup {{v[0-9]+}}.4h, {{w[0-9]+}}
  %vecinit.i = insertelement <4 x i16> undef, i16 %v1, i32 0
  %vecinit1.i = insertelement <4 x i16> %vecinit.i, i16 %v1, i32 1
  %vecinit2.i = insertelement <4 x i16> %vecinit1.i, i16 %v1, i32 2
  %vecinit3.i = insertelement <4 x i16> %vecinit2.i, i16 %v1, i32 3
  ret <4 x i16> %vecinit3.i
}

define <2 x i32> @test_vdup_n_u32(i32 %v1) #0 {
;CHECK: dup {{v[0-9]+}}.2s, {{w[0-9]+}}
  %vecinit.i = insertelement <2 x i32> undef, i32 %v1, i32 0
  %vecinit1.i = insertelement <2 x i32> %vecinit.i, i32 %v1, i32 1
  ret <2 x i32> %vecinit1.i
}

define <1 x i64> @test_vdup_n_u64(i64 %v1) #0 {
;CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
  %vecinit.i = insertelement <1 x i64> undef, i64 %v1, i32 0
  ret <1 x i64> %vecinit.i
}

define <16 x i8> @test_vdupq_n_u8(i8 %v1) #0 {
;CHECK: dup {{v[0-9]+}}.16b, {{w[0-9]+}}
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
;CHECK: dup {{v[0-9]+}}.8h, {{w[0-9]+}}
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
;CHECK: dup {{v[0-9]+}}.4s, {{w[0-9]+}}
  %vecinit.i = insertelement <4 x i32> undef, i32 %v1, i32 0
  %vecinit1.i = insertelement <4 x i32> %vecinit.i, i32 %v1, i32 1
  %vecinit2.i = insertelement <4 x i32> %vecinit1.i, i32 %v1, i32 2
  %vecinit3.i = insertelement <4 x i32> %vecinit2.i, i32 %v1, i32 3
  ret <4 x i32> %vecinit3.i
}

define <2 x i64> @test_vdupq_n_u64(i64 %v1) #0 {
;CHECK: dup {{v[0-9]+}}.2d, {{x[0-9]+}}
  %vecinit.i = insertelement <2 x i64> undef, i64 %v1, i32 0
  %vecinit1.i = insertelement <2 x i64> %vecinit.i, i64 %v1, i32 1
  ret <2 x i64> %vecinit1.i
}

define <8 x i8> @test_vdup_lane_s8(<8 x i8> %v1) #0 {
;CHECK: dup {{v[0-9]+}}.8b, {{v[0-9]+}}.b[5]
  %shuffle = shufflevector <8 x i8> %v1, <8 x i8> undef, <8 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <8 x i8> %shuffle
}

define <4 x i16> @test_vdup_lane_s16(<4 x i16> %v1) #0 {
;CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
  %shuffle = shufflevector <4 x i16> %v1, <4 x i16> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  ret <4 x i16> %shuffle
}

define <2 x i32> @test_vdup_lane_s32(<2 x i32> %v1) #0 {
;CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
  %shuffle = shufflevector <2 x i32> %v1, <2 x i32> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i32> %shuffle
}

define <16 x i8> @test_vdupq_lane_s8(<8 x i8> %v1) #0 {
;CHECK: {{v[0-9]+}}.16b, {{v[0-9]+}}.b[5]
  %shuffle = shufflevector <8 x i8> %v1, <8 x i8> undef, <16 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <16 x i8> %shuffle
}

define <8 x i16> @test_vdupq_lane_s16(<4 x i16> %v1) #0 {
;CHECK: {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
  %shuffle = shufflevector <4 x i16> %v1, <4 x i16> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  ret <8 x i16> %shuffle
}

define <4 x i32> @test_vdupq_lane_s32(<2 x i32> %v1) #0 {
;CHECK: {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
  %shuffle = shufflevector <2 x i32> %v1, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %shuffle
}

define <2 x i64> @test_vdupq_lane_s64(<1 x i64> %v1) #0 {
;CHECK: {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
  %shuffle = shufflevector <1 x i64> %v1, <1 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %shuffle
}

define <8 x i8> @test_vdup_laneq_s8(<16 x i8> %v1) #0 {
;CHECK: dup {{v[0-9]+}}.8b, {{v[0-9]+}}.b[5]
  %shuffle = shufflevector <16 x i8> %v1, <16 x i8> undef, <8 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <8 x i8> %shuffle
}

define <4 x i16> @test_vdup_laneq_s16(<8 x i16> %v1) #0 {
;CHECK: dup {{v[0-9]+}}.4h, {{v[0-9]+}}.h[2]
  %shuffle = shufflevector <8 x i16> %v1, <8 x i16> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  ret <4 x i16> %shuffle
}

define <2 x i32> @test_vdup_laneq_s32(<4 x i32> %v1) #0 {
;CHECK: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
  %shuffle = shufflevector <4 x i32> %v1, <4 x i32> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i32> %shuffle
}

define <16 x i8> @test_vdupq_laneq_s8(<16 x i8> %v1) #0 {
;CHECK: dup {{v[0-9]+}}.16b, {{v[0-9]+}}.b[5]
  %shuffle = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <16 x i8> %shuffle
}

define <8 x i16> @test_vdupq_laneq_s16(<8 x i16> %v1) #0 {
;CHECK: {{v[0-9]+}}.8h, {{v[0-9]+}}.h[2]
  %shuffle = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  ret <8 x i16> %shuffle
}

define <4 x i32> @test_vdupq_laneq_s32(<4 x i32> %v1) #0 {
;CHECK: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
  %shuffle = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %shuffle
}

define <2 x i64> @test_vdupq_laneq_s64(<2 x i64> %v1) #0 {
;CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
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