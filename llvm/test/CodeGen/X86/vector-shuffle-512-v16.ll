; RUN: llc < %s -mcpu=x86-64 -mattr=+avx512f | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx512bw | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512BW

target triple = "x86_64-unknown-unknown"

define <16 x float> @shuffle_v16f32_00_10_01_11_04_14_05_15_08_18_09_19_0c_1c_0d_1d(<16 x float> %a, <16 x float> %b) {
; ALL-LABEL: shuffle_v16f32_00_10_01_11_04_14_05_15_08_18_09_19_0c_1c_0d_1d:
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklps {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[1],zmm1[1],zmm0[4],zmm1[4],zmm0[5],zmm1[5],zmm0[8],zmm1[8],zmm0[9],zmm1[9],zmm0[12],zmm1[12],zmm0[13],zmm1[13]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x float> %a, <16 x float> %b, <16 x i32><i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  ret <16 x float> %shuffle
}

define <16 x float> @shuffle_v16f32_00_zz_01_zz_04_zz_05_zz_08_zz_09_zz_0c_zz_0d_zz(<16 x float> %a, <16 x float> %b) {
; ALL-LABEL: shuffle_v16f32_00_zz_01_zz_04_zz_05_zz_08_zz_09_zz_0c_zz_0d_zz:
; ALL:       # BB#0:
; ALL-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; ALL-NEXT:    vunpcklps {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[1],zmm1[1],zmm0[4],zmm1[4],zmm0[5],zmm1[5],zmm0[8],zmm1[8],zmm0[9],zmm1[9],zmm0[12],zmm1[12],zmm0[13],zmm1[13]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x float> %a, <16 x float> zeroinitializer, <16 x i32><i32 0, i32 16, i32 1, i32 16, i32 4, i32 16, i32 5, i32 16, i32 8, i32 16, i32 9, i32 16, i32 12, i32 16, i32 13, i32 16>
  ret <16 x float> %shuffle
}

define <16 x float> @shuffle_v16f32_vunpcklps_swap(<16 x float> %a, <16 x float> %b) {
; ALL-LABEL: shuffle_v16f32_vunpcklps_swap:
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklps {{.*#+}} zmm0 = zmm1[0],zmm0[0],zmm1[1],zmm0[1],zmm1[4],zmm0[4],zmm1[5],zmm0[5],zmm1[8],zmm0[8],zmm1[9],zmm0[9],zmm1[12],zmm0[12],zmm1[13],zmm0[13]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x float> %a, <16 x float> %b, <16 x i32> <i32 16, i32 0, i32 17, i32 1, i32 20, i32 4, i32 21, i32 5, i32 24, i32 8, i32 25, i32 9, i32 28, i32 12, i32 29, i32 13>
  ret <16 x float> %shuffle
}

define <16 x i32> @shuffle_v16i32_00_10_01_11_04_14_05_15_08_18_09_19_0c_1c_0d_1d(<16 x i32> %a, <16 x i32> %b) {
; ALL-LABEL: shuffle_v16i32_00_10_01_11_04_14_05_15_08_18_09_19_0c_1c_0d_1d:
; ALL:       # BB#0:
; ALL-NEXT:    vpunpckldq {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[1],zmm1[1],zmm0[4],zmm1[4],zmm0[5],zmm1[5],zmm0[8],zmm1[8],zmm0[9],zmm1[9],zmm0[12],zmm1[12],zmm0[13],zmm1[13]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32><i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  ret <16 x i32> %shuffle
}

define <16 x i32> @shuffle_v16i32_zz_10_zz_11_zz_14_zz_15_zz_18_zz_19_zz_1c_zz_1d(<16 x i32> %a, <16 x i32> %b) {
; ALL-LABEL: shuffle_v16i32_zz_10_zz_11_zz_14_zz_15_zz_18_zz_19_zz_1c_zz_1d:
; ALL:       # BB#0:
; ALL-NEXT:    vpxord %zmm0, %zmm0, %zmm0
; ALL-NEXT:    vpunpckldq {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[1],zmm1[1],zmm0[4],zmm1[4],zmm0[5],zmm1[5],zmm0[8],zmm1[8],zmm0[9],zmm1[9],zmm0[12],zmm1[12],zmm0[13],zmm1[13]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x i32> zeroinitializer, <16 x i32> %b, <16 x i32><i32 15, i32 16, i32 13, i32 17, i32 11, i32 20, i32 9, i32 21, i32 7, i32 24, i32 5, i32 25, i32 3, i32 28, i32 1, i32 29>
  ret <16 x i32> %shuffle
}

define <16 x float> @shuffle_v16f32_02_12_03_13_06_16_07_17_0a_1a_0b_1b_0e_1e_0f_1f(<16 x float> %a, <16 x float> %b) {
; ALL-LABEL: shuffle_v16f32_02_12_03_13_06_16_07_17_0a_1a_0b_1b_0e_1e_0f_1f:
; ALL:       # BB#0:
; ALL-NEXT:    vunpckhps {{.*#+}} zmm0 = zmm0[2],zmm1[2],zmm0[3],zmm1[3],zmm0[6],zmm1[6],zmm0[7],zmm1[7],zmm0[10],zmm1[10],zmm0[11],zmm1[11],zmm0[14],zmm1[14],zmm0[15],zmm1[15]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x float> %a, <16 x float> %b, <16 x i32><i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  ret <16 x float> %shuffle
}

define <16 x float> @shuffle_v16f32_zz_12_zz_13_zz_16_zz_17_zz_1a_zz_1b_zz_1e_zz_1f(<16 x float> %a, <16 x float> %b) {
; ALL-LABEL: shuffle_v16f32_zz_12_zz_13_zz_16_zz_17_zz_1a_zz_1b_zz_1e_zz_1f:
; ALL:       # BB#0:
; ALL-NEXT:    vpxord %zmm0, %zmm0, %zmm0
; ALL-NEXT:    vunpckhps {{.*#+}} zmm0 = zmm0[2],zmm1[2],zmm0[3],zmm1[3],zmm0[6],zmm1[6],zmm0[7],zmm1[7],zmm0[10],zmm1[10],zmm0[11],zmm1[11],zmm0[14],zmm1[14],zmm0[15],zmm1[15]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x float> zeroinitializer, <16 x float> %b, <16 x i32><i32 0, i32 18, i32 0, i32 19, i32 4, i32 22, i32 4, i32 23, i32 6, i32 26, i32 6, i32 27, i32 8, i32 30, i32 8, i32 31>
  ret <16 x float> %shuffle
}

define <16 x i32> @shuffle_v16i32_02_12_03_13_06_16_07_17_0a_1a_0b_1b_0e_1e_0f_1f(<16 x i32> %a, <16 x i32> %b) {
; ALL-LABEL: shuffle_v16i32_02_12_03_13_06_16_07_17_0a_1a_0b_1b_0e_1e_0f_1f:
; ALL:       # BB#0:
; ALL-NEXT:    vpunpckhdq {{.*#+}} zmm0 = zmm0[2],zmm1[2],zmm0[3],zmm1[3],zmm0[6],zmm1[6],zmm0[7],zmm1[7],zmm0[10],zmm1[10],zmm0[11],zmm1[11],zmm0[14],zmm1[14],zmm0[15],zmm1[15]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32><i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i32> %shuffle
}

define <16 x i32> @shuffle_v16i32_02_zz_03_zz_06_zz_07_zz_0a_zz_0b_zz_0e_zz_0f_zz(<16 x i32> %a, <16 x i32> %b) {
; ALL-LABEL: shuffle_v16i32_02_zz_03_zz_06_zz_07_zz_0a_zz_0b_zz_0e_zz_0f_zz:
; ALL:       # BB#0:
; ALL-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; ALL-NEXT:    vpunpckhdq {{.*#+}} zmm0 = zmm0[2],zmm1[2],zmm0[3],zmm1[3],zmm0[6],zmm1[6],zmm0[7],zmm1[7],zmm0[10],zmm1[10],zmm0[11],zmm1[11],zmm0[14],zmm1[14],zmm0[15],zmm1[15]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x i32> %a, <16 x i32> zeroinitializer, <16 x i32><i32 2, i32 30, i32 3, i32 28, i32 6, i32 26, i32 7, i32 24, i32 10, i32 22, i32 11, i32 20, i32 14, i32 18, i32 15, i32 16>
  ret <16 x i32> %shuffle
}

define <16 x float> @shuffle_v16f32_02_05_u_u_07_u_0a_01_00_05_u_04_07_u_0a_01(<16 x float> %a)  {
; ALL-LABEL: shuffle_v16f32_02_05_u_u_07_u_0a_01_00_05_u_04_07_u_0a_01:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa32 {{.*#+}} zmm1 = <2,5,u,u,7,u,10,1,0,5,u,4,7,u,10,1>
; ALL-NEXT:    vpermps %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %c = shufflevector <16 x float> %a, <16 x float> undef, <16 x i32> <i32 2, i32 5, i32 undef, i32 undef, i32 7, i32 undef, i32 10, i32 1,  i32 0, i32 5, i32 undef, i32 4, i32 7, i32 undef, i32 10, i32 1>
  ret <16 x float> %c
}

define <16 x i32> @shuffle_v16i32_02_05_u_u_07_u_0a_01_00_05_u_04_07_u_0a_01(<16 x i32> %a)  {
; ALL-LABEL: shuffle_v16i32_02_05_u_u_07_u_0a_01_00_05_u_04_07_u_0a_01:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa32 {{.*#+}} zmm1 = <2,5,u,u,7,u,10,1,0,5,u,4,7,u,10,1>
; ALL-NEXT:    vpermd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %c = shufflevector <16 x i32> %a, <16 x i32> undef, <16 x i32> <i32 2, i32 5, i32 undef, i32 undef, i32 7, i32 undef, i32 10, i32 1,  i32 0, i32 5, i32 undef, i32 4, i32 7, i32 undef, i32 10, i32 1>
  ret <16 x i32> %c
}

define <16 x i32> @shuffle_v16i32_0f_1f_0e_16_0d_1d_04_1e_0b_1b_0a_1a_09_19_08_18(<16 x i32> %a, <16 x i32> %b)  {
; ALL-LABEL: shuffle_v16i32_0f_1f_0e_16_0d_1d_04_1e_0b_1b_0a_1a_09_19_08_18:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa32 {{.*#+}} zmm2 = [15,31,14,22,13,29,4,28,11,27,10,26,9,25,8,24]
; ALL-NEXT:    vpermt2d %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %c = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32> <i32 15, i32 31, i32 14, i32 22, i32 13, i32 29, i32 4, i32 28, i32 11, i32 27, i32 10, i32 26, i32 9, i32 25, i32 8, i32 24>
  ret <16 x i32> %c
}

define <16 x float> @shuffle_v16f32_0f_1f_0e_16_0d_1d_04_1e_0b_1b_0a_1a_09_19_08_18(<16 x float> %a, <16 x float> %b)  {
; ALL-LABEL: shuffle_v16f32_0f_1f_0e_16_0d_1d_04_1e_0b_1b_0a_1a_09_19_08_18:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa32 {{.*#+}} zmm2 = [15,31,14,22,13,29,4,28,11,27,10,26,9,25,8,24]
; ALL-NEXT:    vpermt2ps %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %c = shufflevector <16 x float> %a, <16 x float> %b, <16 x i32> <i32 15, i32 31, i32 14, i32 22, i32 13, i32 29, i32 4, i32 28, i32 11, i32 27, i32 10, i32 26, i32 9, i32 25, i32 8, i32 24>
  ret <16 x float> %c
}

define <16 x float> @shuffle_v16f32_load_0f_1f_0e_16_0d_1d_04_1e_0b_1b_0a_1a_09_19_08_18(<16 x float> %a, <16 x float>* %b)  {
; ALL-LABEL: shuffle_v16f32_load_0f_1f_0e_16_0d_1d_04_1e_0b_1b_0a_1a_09_19_08_18:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa32 {{.*#+}} zmm1 = [15,31,14,22,13,29,4,28,11,27,10,26,9,25,8,24]
; ALL-NEXT:    vpermt2ps (%rdi), %zmm1, %zmm0
; ALL-NEXT:    retq
  %c = load <16 x float>, <16 x float>* %b
  %d = shufflevector <16 x float> %a, <16 x float> %c, <16 x i32> <i32 15, i32 31, i32 14, i32 22, i32 13, i32 29, i32 4, i32 28, i32 11, i32 27, i32 10, i32 26, i32 9, i32 25, i32 8, i32 24>
  ret <16 x float> %d
}

define <16 x i32> @shuffle_v16i32_load_0f_1f_0e_16_0d_1d_04_1e_0b_1b_0a_1a_09_19_08_18(<16 x i32> %a, <16 x i32>* %b)  {
; ALL-LABEL: shuffle_v16i32_load_0f_1f_0e_16_0d_1d_04_1e_0b_1b_0a_1a_09_19_08_18:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa32 {{.*#+}} zmm1 = [15,31,14,22,13,29,4,28,11,27,10,26,9,25,8,24]
; ALL-NEXT:    vpermt2d (%rdi), %zmm1, %zmm0
; ALL-NEXT:    retq
  %c = load <16 x i32>, <16 x i32>* %b
  %d = shufflevector <16 x i32> %a, <16 x i32> %c, <16 x i32> <i32 15, i32 31, i32 14, i32 22, i32 13, i32 29, i32 4, i32 28, i32 11, i32 27, i32 10, i32 26, i32 9, i32 25, i32 8, i32 24>
  ret <16 x i32> %d
}

define <16 x i32> @shuffle_v16i32_0_1_2_19_u_u_u_u_u_u_u_u_u_u_u_u(<16 x i32> %a, <16 x i32> %b)  {
; ALL-LABEL: shuffle_v16i32_0_1_2_19_u_u_u_u_u_u_u_u_u_u_u_u:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa32 {{.*#+}} zmm2 = <0,1,2,19,u,u,u,u,u,u,u,u,u,u,u,u>
; ALL-NEXT:    vpermt2d %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %c = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i32> %c
}

define <8 x float> @shuffle_v16f32_extract_256(float* %RET, float* %a) {
; ALL-LABEL: shuffle_v16f32_extract_256:
; ALL:       # BB#0:
; ALL-NEXT:    vmovups (%rsi), %zmm0
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0
; ALL-NEXT:    retq
  %ptr_a = bitcast float* %a to <16 x float>*
  %v_a = load <16 x float>, <16 x float>* %ptr_a, align 4
  %v2 = shufflevector <16 x float> %v_a, <16 x float> undef, <8 x i32>  <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <8 x float> %v2
}
