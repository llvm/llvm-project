; RUN: llc < %s -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX1
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX2

target triple = "x86_64-unknown-unknown"

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastw %xmm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,1,0,3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,5,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastw %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastw %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_03_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_03_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_03_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastw %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_04_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_04_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_04_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastw %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_05_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_05_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_05_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastw %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_06_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_06_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_00_06_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastw %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastw %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    vpbroadcastw %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_09_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_09_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,0,1,0,1,0,1,0,1,0,1,6,7,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_09_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = <0,0,255,255,u,u,u,u,u,u,u,u,u,u,u,u,255,255,u,u,u,u,u,u,u,u,u,u,u,u,u,u>
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1,16,17,16,17,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 9, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_10_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_10_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,0,1,0,1,0,1,0,1,10,11,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_10_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3,4,5,6,7]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1,16,17,16,17,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_11_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_11_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,0,1,0,1,0,1,14,15,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_11_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3,4,5,6,7]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1,16,17,16,17,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 11, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_12_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_12_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_12_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3,4,5,6,7]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 12, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_13_00_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_13_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_13_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3,4,5,6,7]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 13, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3,4,5,6,7]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 14, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3,4,5,6,7]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_07_07_07_07_07_07_07_15_15_15_15_15_15_15_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_07_07_07_07_07_07_07_07_15_15_15_15_15_15_15_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_07_07_07_07_07_07_07_07_15_15_15_15_15_15_15_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,30,31,30,31,30,31,30,31,30,31,30,31,30,31,30,31]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[0,0,0,0,4,5,6,7,8,8,8,8,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,4,4,4,4,8,9,10,11,12,12,12,12]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 8, i32 8, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_03_03_03_03_07_07_07_07_11_11_11_11_15_15_15_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_03_03_03_03_07_07_07_07_11_11_11_11_15_15_15_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm0[3,3,3,3,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,7,7,7,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[3,3,3,3,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,7,7,7,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_03_03_03_03_07_07_07_07_11_11_11_11_15_15_15_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[3,3,3,3,4,5,6,7,11,11,11,11,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,7,7,7,7,8,9,10,11,15,15,15,15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 3, i32 3, i32 3, i32 3, i32 7, i32 7, i32 7, i32 7, i32 11, i32 11, i32 11, i32 11, i32 15, i32 15, i32 15, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_02_02_04_04_06_06_08_08_10_10_12_12_14_14(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_02_02_04_04_06_06_08_08_10_10_12_12_14_14:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm0[0,0,2,2,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,6,6]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,2,2,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,6,6]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_02_02_04_04_06_06_08_08_10_10_12_12_14_14:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[0,0,2,2,4,5,6,7,8,8,10,10,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,4,4,6,6,8,9,10,11,12,12,14,14]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6, i32 8, i32 8, i32 10, i32 10, i32 12, i32 12, i32 14, i32 14>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_01_01_03_03_05_05_07_07_09_09_11_11_13_13_15_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_01_01_03_03_05_05_07_07_09_09_11_11_13_13_15_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm0[1,1,3,3,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,5,5,7,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[1,1,3,3,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,5,5,7,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_01_01_03_03_05_05_07_07_09_09_11_11_13_13_15_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[1,1,3,3,4,5,6,7,9,9,11,11,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,5,5,7,7,8,9,10,11,13,13,15,15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7, i32 9, i32 9, i32 11, i32 11, i32 13, i32 13, i32 15, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_01_00_00_00_00_00_00_00_01_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_01_00_00_00_00_00_00_00_01_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_01_00_00_00_00_00_00_00_01_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_02_00_00_00_00_00_00_00_02_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_02_00_00_00_00_00_00_00_02_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_02_00_00_00_00_00_00_00_02_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_03_00_00_00_00_00_00_00_03_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_03_00_00_00_00_00_00_00_03_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_03_00_00_00_00_00_00_00_03_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_04_00_00_00_00_00_00_00_04_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_04_00_00_00_00_00_00_00_04_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_04_00_00_00_00_00_00_00_04_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_05_00_00_00_00_00_00_00_05_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_05_00_00_00_00_00_00_00_05_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_05_00_00_00_00_00_00_00_05_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_06_00_00_00_00_00_00_00_06_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_06_00_00_00_00_00_00_00_06_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_06_00_00_00_00_00_00_00_06_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_07_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_07_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_17_02_19_04_21_06_23_08_25_10_27_12_29_14_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_17_02_19_04_21_06_23_08_25_10_27_12_29_14_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_17_02_19_04_21_06_23_08_25_10_27_12_29_14_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7],ymm0[8],ymm1[9],ymm0[10],ymm1[11],ymm0[12],ymm1[13],ymm0[14],ymm1[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 17, i32 2, i32 19, i32 4, i32 21, i32 6, i32 23, i32 8, i32 25, i32 10, i32 27, i32 12, i32 29, i32 14, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_16_01_18_03_20_05_22_07_24_09_26_11_28_13_30_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_16_01_18_03_20_05_22_07_24_09_26_11_28_13_30_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm1[0],xmm0[1],xmm1[2],xmm0[3],xmm1[4],xmm0[5],xmm1[6],xmm0[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_16_01_18_03_20_05_22_07_24_09_26_11_28_13_30_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm1[0],ymm0[1],ymm1[2],ymm0[3],ymm1[4],ymm0[5],ymm1[6],ymm0[7],ymm1[8],ymm0[9],ymm1[10],ymm0[11],ymm1[12],ymm0[13],ymm1[14],ymm0[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 16, i32 1, i32 18, i32 3, i32 20, i32 5, i32 22, i32 7, i32 24, i32 9, i32 26, i32 11, i32 28, i32 13, i32 30, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_01_18_19_04_05_22_23_08_09_26_27_12_13_30_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_01_18_19_04_05_22_23_08_09_26_27_12_13_30_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vblendps {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_01_18_19_04_05_22_23_08_09_26_27_12_13_30_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 4, i32 5, i32 22, i32 23, i32 8, i32 9, i32 26, i32 27, i32 12, i32 13, i32 30, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_16_17_18_19_04_05_06_07_24_25_26_27_12_13_14_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_16_17_18_19_04_05_06_07_24_25_26_27_12_13_14_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vblendpd {{.*#+}} ymm0 = ymm1[0],ymm0[1],ymm1[2],ymm0[3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_16_17_18_19_04_05_06_07_24_25_26_27_12_13_14_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3],ymm1[4,5],ymm0[6,7]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_01_02_03_04_05_06_07_08_09_10_11_12_13_14_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_01_02_03_04_05_06_07_08_09_10_11_12_13_14_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm2[0,1,2,3,4,5,6],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_01_02_03_04_05_06_07_08_09_10_11_12_13_14_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_16_01_02_03_04_05_06_07_08_09_10_11_12_13_14_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_16_01_02_03_04_05_06_07_08_09_10_11_12_13_14_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm1[0],xmm0[1,2,3,4,5,6,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_16_01_02_03_04_05_06_07_08_09_10_11_12_13_14_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 16, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_17_02_19_04_21_06_23_24_09_26_11_28_13_30_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_17_02_19_04_21_06_23_24_09_26_11_28_13_30_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_17_02_19_04_21_06_23_24_09_26_11_28_13_30_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [255,255,0,0,255,255,0,0,255,255,0,0,255,255,0,0,0,0,255,255,0,0,255,255,0,0,255,255,0,0,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 17, i32 2, i32 19, i32 4, i32 21, i32 6, i32 23, i32 24, i32 9, i32 26, i32 11, i32 28, i32 13, i32 30, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_16_01_18_03_20_05_22_07_08_25_10_27_12_29_14_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_16_01_18_03_20_05_22_07_08_25_10_27_12_29_14_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm1[0],xmm0[1],xmm1[2],xmm0[3],xmm1[4],xmm0[5],xmm1[6],xmm0[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_16_01_18_03_20_05_22_07_08_25_10_27_12_29_14_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,255,255,0,0,255,255,0,0,255,255,0,0,255,255,255,255,0,0,255,255,0,0,255,255,0,0,255,255,0,0]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 16, i32 1, i32 18, i32 3, i32 20, i32 5, i32 22, i32 7, i32 8, i32 25, i32 10, i32 27, i32 12, i32 29, i32 14, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_01_18_19_20_21_06_07_08_09_26_27_12_13_30_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_01_18_19_20_21_06_07_08_09_26_27_12_13_30_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vblendps {{.*#+}} ymm0 = ymm0[0],ymm1[1,2],ymm0[3,4],ymm1[5],ymm0[6],ymm1[7]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_01_18_19_20_21_06_07_08_09_26_27_12_13_30_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0],ymm1[1,2],ymm0[3,4],ymm1[5],ymm0[6],ymm1[7]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 6, i32 7, i32 8, i32 9, i32 26, i32 27, i32 12, i32 13, i32 30, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_16_00_16_00_16_00_16_00_16_00_16_00_16_00_16(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_16_00_16_00_16_00_16_00_16_00_16_00_16_00_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_16_00_16_00_16_00_16_00_16_00_16_00_16_00_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastw %xmm1, %ymm1
; AVX2-NEXT:    vpbroadcastd %xmm0, %ymm0
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7],ymm0[8],ymm1[9],ymm0[10],ymm1[11],ymm0[12],ymm1[13],ymm0[14],ymm1[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_16_00_16_00_16_00_16_08_24_08_24_08_24_08_24(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_16_00_16_00_16_00_16_08_24_08_24_08_24_08_24:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm2 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm2 = xmm2[0,0,0,0]
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_16_00_16_00_16_00_16_08_24_08_24_08_24_08_24:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[0,1,0,1,4,5,0,1,0,1,0,1,12,13,0,1,16,17,16,17,20,21,16,17,16,17,16,17,28,29,16,17]
; AVX2-NEXT:    vpshufd {{.*#+}} ymm0 = ymm0[0,0,0,0,4,4,4,4]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7],ymm0[8],ymm1[9],ymm0[10],ymm1[11],ymm0[12],ymm1[13],ymm0[14],ymm1[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 8, i32 24, i32 8, i32 24, i32 8, i32 24, i32 8, i32 24>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_16_16_16_16_04_05_06_07_24_24_24_24_12_13_14_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_16_16_16_16_04_05_06_07_24_24_24_24_12_13_14_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufd {{.*#+}} xmm2 = xmm2[2,3,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm3 = xmm3[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm2 = xmm3[0],xmm2[0]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm1[0],xmm0[0]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_16_16_16_16_04_05_06_07_24_24_24_24_12_13_14_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm1 = ymm1[0,0,0,0,4,5,6,7,8,8,8,8,12,13,14,15]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3],ymm1[4,5],ymm0[6,7]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 16, i32 16, i32 16, i32 16, i32 4, i32 5, i32 6, i32 7, i32 24, i32 24, i32 24, i32 24, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_19_18_17_16_07_06_05_04_27_26_25_24_15_14_13_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_19_18_17_16_07_06_05_04_27_26_25_24_15_14_13_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm2 = xmm2[3,2,1,0,4,5,6,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpshufd {{.*#+}} xmm3 = xmm3[2,3,2,3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm3 = xmm3[3,2,1,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm3[0]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[3,2,1,0,4,5,6,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,2,3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[3,2,1,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm1[0],xmm0[0]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_19_18_17_16_07_06_05_04_27_26_25_24_15_14_13_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3],ymm1[4,5],ymm0[6,7]
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[3,2,1,0,4,5,6,7,11,10,9,8,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 19, i32 18, i32 17, i32 16, i32 7, i32 6, i32 5, i32 4, i32 27, i32 26, i32 25, i32 24, i32 15, i32 14, i32 13, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_19_18_17_16_03_02_01_00_27_26_25_24_11_10_09_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_19_18_17_16_03_02_01_00_27_26_25_24_11_10_09_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm2 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [12,13,8,9,4,5,0,1,14,15,10,11,6,7,2,3]
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; AVX1-NEXT:    vpshufb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_19_18_17_16_03_02_01_00_27_26_25_24_11_10_09_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm1 = ymm1[3,2,1,0,4,5,6,7,11,10,9,8,12,13,14,15]
; AVX2-NEXT:    vpshufd {{.*#+}} ymm0 = ymm0[0,1,0,1,4,5,4,5]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3],ymm1[4,5],ymm0[6,7]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 19, i32 18, i32 17, i32 16, i32 3, i32 2, i32 1, i32 0, i32 27, i32 26, i32 25, i32 24, i32 11, i32 10, i32 9, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_01_00_08_08_08_08_08_08_09_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_01_00_08_08_08_08_08_08_09_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_01_00_08_08_08_08_08_08_09_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1,16,17,16,17,16,17,16,17,16,17,16,17,18,19,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 9, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_08_08_08_10_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_08_08_08_10_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_08_08_08_10_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1,16,17,16,17,16,17,16,17,16,17,20,21,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 10, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_08_11_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_08_11_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_08_11_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1,16,17,16,17,16,17,16,17,22,23,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 11, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_12_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_12_08_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_12_08_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1,16,17,16,17,16,17,24,25,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 12, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_13_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_13_08_08_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_13_08_08_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1,16,17,16,17,26,27,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 13, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_06_00_00_00_00_00_00_08_14_08_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_06_00_00_00_00_00_00_08_14_08_08_08_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_06_00_00_00_00_00_00_08_14_08_08_08_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1,16,17,28,29,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 14, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_00_00_00_00_00_00_00_15_08_08_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_07_00_00_00_00_00_00_00_15_08_08_08_08_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_07_00_00_00_00_00_00_00_15_08_08_08_08_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1,30,31,16,17,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 15, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_16_01_17_02_18_03_19_08_24_09_25_10_26_11_27(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_16_01_17_02_18_03_19_08_24_09_25_10_26_11_27:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm2 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_16_01_17_02_18_03_19_08_24_09_25_10_26_11_27:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpunpcklwd {{.*#+}} ymm0 = ymm0[0],ymm1[0],ymm0[1],ymm1[1],ymm0[2],ymm1[2],ymm0[3],ymm1[3],ymm0[8],ymm1[8],ymm0[9],ymm1[9],ymm0[10],ymm1[10],ymm0[11],ymm1[11]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_20_05_21_06_22_07_23_12_28_13_29_14_30_15_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_20_05_21_06_22_07_23_12_28_13_29_14_30_15_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm2 = xmm3[4],xmm2[4],xmm3[5],xmm2[5],xmm3[6],xmm2[6],xmm3[7],xmm2[7]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_20_05_21_06_22_07_23_12_28_13_29_14_30_15_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpunpckhwd {{.*#+}} ymm0 = ymm0[4],ymm1[4],ymm0[5],ymm1[5],ymm0[6],ymm1[6],ymm0[7],ymm1[7],ymm0[12],ymm1[12],ymm0[13],ymm1[13],ymm0[14],ymm1[14],ymm0[15],ymm1[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_16_01_17_02_18_03_19_12_28_13_29_14_30_15_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_16_01_17_02_18_03_19_12_28_13_29_14_30_15_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm2 = xmm3[4],xmm2[4],xmm3[5],xmm2[5],xmm3[6],xmm2[6],xmm3[7],xmm2[7]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_16_01_17_02_18_03_19_12_28_13_29_14_30_15_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,0,1,u,u,2,3,u,u,4,5,u,u,6,7,u,u,24,25,u,u,26,27,u,u,28,29,u,u,30,31]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,u,u,2,3,u,u,4,5,u,u,6,7,u,u,24,25,u,u,26,27,u,u,28,29,u,u,30,31,u,u]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7],ymm0[8],ymm1[9],ymm0[10],ymm1[11],ymm0[12],ymm1[13],ymm0[14],ymm1[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_20_05_21_06_22_07_23_08_24_09_25_10_26_11_27(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_20_05_21_06_22_07_23_08_24_09_25_10_26_11_27:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm2 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_20_05_21_06_22_07_23_08_24_09_25_10_26_11_27:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,8,9,u,u,10,11,u,u,12,13,u,u,14,15,u,u,16,17,u,u,18,19,u,u,20,21,u,u,22,23]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[8,9,u,u,10,11,u,u,12,13,u,u,14,15,u,u,16,17,u,u,18,19,u,u,20,21,u,u,22,23,u,u]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7],ymm0[8],ymm1[9],ymm0[10],ymm1[11],ymm0[12],ymm1[13],ymm0[14],ymm1[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_01_00_08_09_08_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_01_00_08_09_08_08_08_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,2,3,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_01_00_08_09_08_08_08_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1,16,17,18,19,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 8, i32 9, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_10_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_10_08_08_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,4,5,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_10_08_08_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1,16,17,16,17,20,21,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 8, i32 8, i32 10, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_11_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_11_08_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,6,7,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_11_08_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1,16,17,16,17,16,17,22,23,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 11, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_08_12_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_08_12_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,8,9,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_08_12_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,24,25,16,17,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 12, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_08_08_08_13_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_08_08_08_13_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,10,11,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_08_08_08_13_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,16,17,26,27,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 13, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_06_00_00_00_00_00_00_08_08_08_08_08_08_14_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_06_00_00_00_00_00_00_08_08_08_08_08_08_14_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,12,13,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_06_00_00_00_00_00_00_08_08_08_08_08_08_14_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,16,17,16,17,28,29,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 14, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_00_00_00_00_00_00_00_08_08_08_08_08_08_08_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_07_00_00_00_00_00_00_00_08_08_08_08_08_08_08_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,14,15]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_07_00_00_00_00_00_00_00_08_08_08_08_08_08_08_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,16,17,16,17,16,17,30,31]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_02_02_04_04_06_06_14_14_12_12_10_10_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_02_02_04_04_06_06_14_14_12_12_10_10_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm0[0,0,2,2,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,6,6]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[12,13,12,13,8,9,8,9,4,5,4,5,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_02_02_04_04_06_06_14_14_12_12_10_10_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,4,5,4,5,8,9,8,9,12,13,12,13,28,29,28,29,24,25,24,25,20,21,20,21,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6, i32 14, i32 14, i32 12, i32 12, i32 10, i32 10, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_04_04_04_00_00_00_00_08_08_08_08_12_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_04_04_04_00_00_00_00_08_08_08_08_12_12_12_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[8,9,8,9,8,9,8,9,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_04_04_04_00_00_00_00_08_08_08_08_12_12_12_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[8,9,8,9,8,9,8,9,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,24,25,24,25,24,25,24,25]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 4, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_uu_uu_00_00_00_00_00_08_08_uu_uu_08_08_14_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_uu_uu_00_00_00_00_00_08_08_uu_uu_08_08_14_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,1,2,3,4,5,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,4,5,6,7,0,1,0,1,12,13,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_uu_uu_00_00_00_00_00_08_08_uu_uu_08_08_14_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,u,u,u,u,0,1,0,1,0,1,0,1,0,1,16,17,16,17,u,u,u,u,16,17,16,17,28,29,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 undef, i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 undef, i32 undef, i32 8, i32 8, i32 14, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_uu_00_00_00_00_00_00_08_08_uu_uu_08_08_08_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_07_uu_00_00_00_00_00_00_08_08_uu_uu_08_08_08_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[14,15,2,3,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,4,5,6,7,0,1,0,1,0,1,14,15]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_07_uu_00_00_00_00_00_00_08_08_uu_uu_08_08_08_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[14,15,u,u,0,1,0,1,0,1,0,1,0,1,0,1,16,17,16,17,u,u,u,u,16,17,16,17,16,17,30,31]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 undef, i32 undef, i32 8, i32 8, i32 8, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_uu_uu_02_04_04_uu_06_14_14_uu_12_10_10_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_uu_uu_02_04_04_uu_06_14_14_uu_12_10_10_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm0[0,1,2,2,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,6,6]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[12,13,12,13,12,13,8,9,4,5,4,5,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_uu_uu_02_04_04_uu_06_14_14_uu_12_10_10_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,u,u,u,u,4,5,8,9,8,9,u,u,12,13,28,29,28,29,u,u,24,25,20,21,20,21,16,17,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 undef, i32 undef, i32 2, i32 4, i32 4, i32 undef, i32 6, i32 14, i32 14, i32 undef, i32 12, i32 10, i32 10, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_04_04_04_uu_uu_uu_uu_08_08_08_uu_uu_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_04_04_04_uu_uu_uu_uu_08_08_08_uu_uu_12_12_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm0[2,1,2,3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,0,3,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_04_04_04_uu_uu_uu_uu_08_08_08_uu_uu_12_12_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[8,9,8,9,8,9,8,9,u,u,u,u,u,u,u,u,16,17,16,17,16,17,u,u,u,u,24,25,24,25,24,25]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 4, i32 4, i32 4, i32 undef, i32 undef, i32 undef, i32 undef, i32 8, i32 8, i32 8, i32 undef, i32 undef, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_04_04_04_04_16_16_16_16_20_20_20_20(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_04_04_04_04_16_16_16_16_20_20_20_20:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_04_04_04_04_16_16_16_16_20_20_20_20:
; AVX2:       # BB#0:
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[0,0,0,0,4,5,6,7,8,8,8,8,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,4,4,4,4,8,9,10,11,12,12,12,12]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 16, i32 16, i32 16, i32 16, i32 20, i32 20, i32 20, i32 20>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_08_08_08_08_12_12_12_12_16_16_16_16_20_20_20_20(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_08_08_08_08_12_12_12_12_16_16_16_16_20_20_20_20:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_08_08_08_08_12_12_12_12_16_16_16_16_20_20_20_20:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm0 = ymm0[2,3],ymm1[0,1]
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[0,0,0,0,4,5,6,7,8,8,8,8,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,4,4,4,4,8,9,10,11,12,12,12,12]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 8, i32 8, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12, i32 16, i32 16, i32 16, i32 16, i32 20, i32 20, i32 20, i32 20>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_08_08_08_08_12_12_12_12_24_24_24_24_28_28_28_28(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_08_08_08_08_12_12_12_12_24_24_24_24_28_28_28_28:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm1
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_08_08_08_08_12_12_12_12_24_24_24_24_28_28_28_28:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm0 = ymm0[2,3],ymm1[2,3]
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[0,0,0,0,4,5,6,7,8,8,8,8,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,4,4,4,4,8,9,10,11,12,12,12,12]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 8, i32 8, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12, i32 24, i32 24, i32 24, i32 24, i32 28, i32 28, i32 28, i32 28>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_04_04_04_04_24_24_24_24_28_28_28_28(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_04_04_04_04_24_24_24_24_28_28_28_28:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm1
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_04_04_04_04_24_24_24_24_28_28_28_28:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[0,0,0,0,4,5,6,7,8,8,8,8,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,4,4,4,4,8,9,10,11,12,12,12,12]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 24, i32 24, i32 24, i32 24, i32 28, i32 28, i32 28, i32 28>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_16_01_17_02_18_03_19_04_20_05_21_06_22_07_23(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_16_01_17_02_18_03_19_04_20_05_21_06_22_07_23:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm2 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_16_01_17_02_18_03_19_04_20_05_21_06_22_07_23:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm2 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_zz_zz_zz_zz_zz_zz_zz_16_zz_zz_zz_zz_zz_zz_zz_24(<16 x i16> %a) {
; AVX1-LABEL: shuffle_v16i16_zz_zz_zz_zz_zz_zz_zz_16_zz_zz_zz_zz_zz_zz_zz_24:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpslldq {{.*#+}} xmm1 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm0[0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpslldq {{.*#+}} xmm0 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm0[0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_zz_zz_zz_zz_zz_zz_zz_16_zz_zz_zz_zz_zz_zz_zz_24:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpslldq {{.*#+}} ymm0 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,ymm0[0,1],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,ymm0[16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> zeroinitializer, <16 x i16> %a, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 24>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_17_18_19_20_21_22_23_zz_25_26_27_28_29_30_31_zz(<16 x i16> %a) {
; AVX1-LABEL: shuffle_v16i16_17_18_19_20_21_22_23_zz_25_26_27_28_29_30_31_zz:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpsrldq {{.*#+}} xmm1 = xmm0[2,3,4,5,6,7,8,9,10,11,12,13,14,15],zero,zero
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpsrldq {{.*#+}} xmm0 = xmm0[2,3,4,5,6,7,8,9,10,11,12,13,14,15],zero,zero
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_17_18_19_20_21_22_23_zz_25_26_27_28_29_30_31_zz:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsrldq {{.*#+}} ymm0 = ymm0[2,3,4,5,6,7,8,9,10,11,12,13,14,15],zero,zero,ymm0[18,19,20,21,22,23,24,25,26,27,28,29,30,31],zero,zero
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> zeroinitializer, <16 x i16> %a, <16 x i32> <i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0>
  ret <16 x i16> %shuffle
}

;
; Shuffle to logical bit shifts
;

define <16 x i16> @shuffle_v16i16_zz_00_zz_02_zz_04_zz_06_zz_08_zz_10_zz_12_zz_14(<16 x i16> %a) {
; AVX1-LABEL: shuffle_v16i16_zz_00_zz_02_zz_04_zz_06_zz_08_zz_10_zz_12_zz_14:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpslld $16, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpslld $16, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_zz_00_zz_02_zz_04_zz_06_zz_08_zz_10_zz_12_zz_14:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpslld $16, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> zeroinitializer, <16 x i32> <i32 16, i32 0, i32 16, i32 2, i32 16, i32 4, i32 16, i32 6, i32 16, i32 8, i32 16, i32 10, i32 16, i32 12, i32 16, i32 14>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_zz_zz_zz_00_zz_zz_zz_04_zz_zz_zz_08_zz_zz_zz_12(<16 x i16> %a) {
; AVX1-LABEL: shuffle_v16i16_zz_zz_zz_00_zz_zz_zz_04_zz_zz_zz_08_zz_zz_zz_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpsllq $48, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpsllq $48, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_zz_zz_zz_00_zz_zz_zz_04_zz_zz_zz_08_zz_zz_zz_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsllq $48, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> zeroinitializer, <16 x i32> <i32 16, i32 16, i32 16, i32 0, i32 16, i32 16, i32 16, i32 4, i32 16, i32 16, i32 16, i32 8, i32 16, i32 16, i32 16, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_01_zz_03_zz_05_zz_07_zz_09_zz_11_zz_13_zz_15_zz(<16 x i16> %a) {
; AVX1-LABEL: shuffle_v16i16_01_zz_03_zz_05_zz_07_zz_09_zz_11_zz_13_zz_15_zz:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpsrld $16, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpsrld $16, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_01_zz_03_zz_05_zz_07_zz_09_zz_11_zz_13_zz_15_zz:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsrld $16, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> zeroinitializer, <16 x i32> <i32 1, i32 16, i32 3, i32 16, i32 5, i32 16, i32 7, i32 16, i32 9, i32 16, i32 11, i32 16, i32 13, i32 16, i32 15, i32 16>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_02_03_zz_zz_06_07_zz_zz_10_11_zz_zz_14_15_zz_zz(<16 x i16> %a) {
; AVX1-LABEL: shuffle_v16i16_02_03_zz_zz_06_07_zz_zz_10_11_zz_zz_14_15_zz_zz:
; AVX1:       # BB#0:
; AVX1-NEXT:    vxorps %ymm1, %ymm1, %ymm1
; AVX1-NEXT:    vshufps {{.*#+}} ymm0 = ymm0[1,3],ymm1[1,3],ymm0[5,7],ymm1[5,7]
; AVX1-NEXT:    vshufps {{.*#+}} ymm0 = ymm0[0,2,1,3,4,6,5,7]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_02_03_zz_zz_06_07_zz_zz_10_11_zz_zz_14_15_zz_zz:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsrlq $32, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> zeroinitializer, <16 x i32> <i32 2, i32 3, i32 16, i32 16, i32 6, i32 7, i32 16, i32 16, i32 10, i32 11, i32 16, i32 16, i32 14, i32 15, i32 16, i32 16>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_16_zz_zz_zz_17_zz_zz_zz_18_zz_zz_zz_19_zz_zz_zz(<16 x i16> %a) {
; AVX1-LABEL: shuffle_v16i16_16_zz_zz_zz_17_zz_zz_zz_18_zz_zz_zz_19_zz_zz_zz:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmovzxwq {{.*#+}} xmm1 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,1,2,3]
; AVX1-NEXT:    vpmovzxwq {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_16_zz_zz_zz_17_zz_zz_zz_18_zz_zz_zz_19_zz_zz_zz:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxwq {{.*#+}} ymm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> zeroinitializer, <16 x i16> %a, <16 x i32> <i32 16, i32 0, i32 0, i32 0, i32 17, i32 0, i32 0, i32 0, i32 18, i32 0, i32 0, i32 0, i32 19, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_16_zz_17_zz_18_zz_19_zz_20_zz_21_zz_22_zz_22_zz(<16 x i16> %a) {
; AVX1-LABEL: shuffle_v16i16_16_zz_17_zz_18_zz_19_zz_20_zz_21_zz_22_zz_22_zz:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vpmovzxwd {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_16_zz_17_zz_18_zz_19_zz_20_zz_21_zz_22_zz_22_zz:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> zeroinitializer, <16 x i16> %a, <16 x i32> <i32 16, i32 0, i32 17, i32 0, i32 18, i32 0, i32 19, i32 0, i32 20, i32 0, i32 21, i32 0, i32 22, i32 0, i32 23, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_23_00_01_02_03_04_05_06_31_08_09_10_11_12_13_14(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_23_00_01_02_03_04_05_06_31_08_09_10_11_12_13_14:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[14,15],xmm3[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm1[14,15],xmm0[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_23_00_01_02_03_04_05_06_31_08_09_10_11_12_13_14:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm1[14,15],ymm0[0,1,2,3,4,5,6,7,8,9,10,11,12,13],ymm1[30,31],ymm0[16,17,18,19,20,21,22,23,24,25,26,27,28,29]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 31, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_01_02_03_04_05_06_07_16_09_10_11_12_13_14_15_24(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_01_02_03_04_05_06_07_16_09_10_11_12_13_14_15_24:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[2,3,4,5,6,7,8,9,10,11,12,13,14,15],xmm3[0,1]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[2,3,4,5,6,7,8,9,10,11,12,13,14,15],xmm1[0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_01_02_03_04_05_06_07_16_09_10_11_12_13_14_15_24:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm0[2,3,4,5,6,7,8,9,10,11,12,13,14,15],ymm1[0,1],ymm0[18,19,20,21,22,23,24,25,26,27,28,29,30,31],ymm1[16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 24>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_17_18_19_20_21_22_23_00_25_26_27_28_29_30_31_8(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_17_18_19_20_21_22_23_00_25_26_27_28_29_30_31_8:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[2,3,4,5,6,7,8,9,10,11,12,13,14,15],xmm3[0,1]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm1[2,3,4,5,6,7,8,9,10,11,12,13,14,15],xmm0[0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_17_18_19_20_21_22_23_00_25_26_27_28_29_30_31_8:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm1[2,3,4,5,6,7,8,9,10,11,12,13,14,15],ymm0[0,1],ymm1[18,19,20,21,22,23,24,25,26,27,28,29,30,31],ymm0[16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 00, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_16_17_18_19_20_21_22_15_24_25_26_27_28_29_30(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_07_16_17_18_19_20_21_22_15_24_25_26_27_28_29_30:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[14,15],xmm3[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[14,15],xmm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_07_16_17_18_19_20_21_22_15_24_25_26_27_28_29_30:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm0[14,15],ymm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13],ymm0[30,31],ymm1[16,17,18,19,20,21,22,23,24,25,26,27,28,29]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 15, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_01_02_03_04_05_06_07_00_17_18_19_20_21_22_23_16(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_01_02_03_04_05_06_07_00_17_18_19_20_21_22_23_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm1 = xmm1[2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_01_02_03_04_05_06_07_00_17_18_19_20_21_22_23_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm0[2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1,18,19,20,21,22,23,24,25,26,27,28,29,30,31,16,17]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 16>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_00_01_02_03_04_05_06_23_16_17_18_19_20_21_22(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_07_00_01_02_03_04_05_06_23_16_17_18_19_20_21_22:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm1 = xmm1[14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_07_00_01_02_03_04_05_06_23_16_17_18_19_20_21_22:
; AVX2:       # BB#0:
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm0[14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,30,31,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 23, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_01_00_01_02_03_02_11_08_09_08_09_10_11_10_11(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_01_00_01_02_03_02_11_08_09_08_09_10_11_10_11:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,2,0,2,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,6,4,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,0,1,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_01_00_01_02_03_02_11_08_09_08_09_10_11_10_11:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,2,0,2,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,6,4,7]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,0,1,1]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 2, i32 3, i32 2, i32 11, i32 8, i32 9, i32 8, i32 9, i32 10, i32 11, i32 10, i32 11>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_06_07_04_05_02_03_00_09_14_15_12_13_10_11_08_09(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_06_07_04_05_02_03_00_09_14_15_12_13_10_11_08_09:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2,3,4,5,6,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[3,2,1,0]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[3,2,1,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_06_07_04_05_02_03_00_09_14_15_12_13_10_11_08_09:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2,3,4,5,6,7]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[3,2,1,0]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[3,2,1,0]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 6, i32 7, i32 4, i32 5, i32 2, i32 3, i32 0, i32 9, i32 14, i32 15, i32 12, i32 13, i32 10, i32 11, i32 8, i32 9>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_05_06_07_16_17_18_27_12_13_14_15_24_25_26_27(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_05_06_07_16_17_18_27_12_13_14_15_24_25_26_27:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[8,9,10,11,12,13,14,15],xmm3[0,1,2,3,4,5,6,7]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,4,5,4,5,6,7,0,1,4,5,8,9,14,15]
; AVX1-NEXT:    vpunpckhqdq {{.*#+}} xmm0 = xmm0[1],xmm1[1]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_05_06_07_16_17_18_27_12_13_14_15_24_25_26_27:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3],ymm1[4,5],ymm0[6,7]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3],xmm0[4,5,6,7]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 27, i32 12, i32 13, i32 14, i32 15, i32 24, i32 25, i32 26, i32 27>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,3]
; AVX2-NEXT:    vpbroadcastw %xmm1, %xmm1
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_04_04_04_12_08_08_08_08_12_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_00_00_04_04_04_12_08_08_08_08_12_12_12_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_00_00_04_04_04_12_08_08_08_08_12_12_12_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,4,7]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,4,4]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 12, i32 8, i32 8, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_00_uu_01_uu_02_uu_11_uu_08_uu_09_uu_10_uu_11(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_00_uu_01_uu_02_uu_11_uu_08_uu_09_uu_10_uu_11:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm2 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,2,2,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_00_uu_01_uu_02_uu_11_uu_08_uu_09_uu_10_uu_11:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm2 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,2,2,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,6,7]
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 0, i32 undef, i32 1, i32 undef, i32 2, i32 undef, i32 11, i32 undef, i32 8, i32 undef, i32 9, i32 undef, i32 10, i32 undef, i32 11>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_04_uu_05_uu_06_uu_15_uu_12_uu_13_uu_14_uu_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_04_uu_05_uu_06_uu_15_uu_12_uu_13_uu_14_uu_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm2 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,2,2,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_04_uu_05_uu_06_uu_15_uu_12_uu_13_uu_14_uu_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm2 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,2,2,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,6,7]
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 4, i32 undef, i32 5, i32 undef, i32 6, i32 undef, i32 15, i32 undef, i32 12, i32 undef, i32 13, i32 undef, i32 14, i32 undef, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_03_01_02_00_06_07_04_13_11_09_10_08_14_15_12_13(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_03_01_02_00_06_07_04_13_11_09_10_08_14_15_12_13:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4],xmm1[5],xmm0[6,7]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[3,1,2,0,4,5,6,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,1,3,2]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[3,1,2,0,4,5,6,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,1,3,2]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_03_01_02_00_06_07_04_13_11_09_10_08_14_15_12_13:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4],xmm1[5],xmm0[6,7]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[3,1,2,0,4,5,6,7]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,1,3,2]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[3,1,2,0,4,5,6,7]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,1,3,2]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 3, i32 1, i32 2, i32 0, i32 6, i32 7, i32 4, i32 13, i32 11, i32 9, i32 10, i32 8, i32 14, i32 15, i32 12, i32 13>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_04_04_04_00_00_00_08_12_12_12_12_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_04_04_04_00_00_00_08_12_12_12_12_08_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpslldq {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm1[0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[8,9,8,9,8,9,8,9,0,1,0,1,0,1,14,15]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[8,9,8,9,8,9,8,9,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_04_04_04_00_00_00_08_12_12_12_12_08_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpbroadcastw %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[8,9,8,9,8,9,8,9,0,1,0,1,0,1,14,15]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[8,9,8,9,8,9,8,9,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 4, i32 4, i32 4, i32 0, i32 0, i32 0, i32 8, i32 12, i32 12, i32 12, i32 12, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_02_03_00_01_06_07_04_13_10_11_08_09_14_15_12_13(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_02_03_00_01_06_07_04_13_10_11_08_09_14_15_12_13:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4],xmm1[5],xmm0[6,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,0,3,2]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[1,0,3,2]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_02_03_00_01_06_07_04_13_10_11_08_09_14_15_12_13:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4],xmm1[5],xmm0[6,7]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,0,3,2]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[1,0,3,2]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 13, i32 10, i32 11, i32 8, i32 9, i32 14, i32 15, i32 12, i32 13>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_02_03_00_02_06_07_04_13_10_11_08_10_14_15_12_13(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_02_03_00_02_06_07_04_13_10_11_08_10_14_15_12_13:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4],xmm1[5],xmm0[6,7]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[2,3,0,2,4,5,6,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,1,3,2]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[2,3,0,2,4,5,6,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,1,3,2]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_02_03_00_02_06_07_04_13_10_11_08_10_14_15_12_13:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4],xmm1[5],xmm0[6,7]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[2,3,0,2,4,5,6,7]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,1,3,2]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[2,3,0,2,4,5,6,7]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,1,3,2]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 2, i32 3, i32 0, i32 2, i32 6, i32 7, i32 4, i32 13, i32 10, i32 11, i32 8, i32 10, i32 14, i32 15, i32 12, i32 13>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_02_03_00_01_06_07_04_15_10_11_08_09_14_15_12_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_02_03_00_01_06_07_04_15_10_11_08_09_14_15_12_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,0,3,2]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm1[7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[1,0,2,3]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,6,7,4,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_02_03_00_01_06_07_04_15_10_11_08_09_14_15_12_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,0,3,2]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm1[7]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[1,0,2,3]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,6,7,4,7]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 15, i32 10, i32 11, i32 8, i32 9, i32 14, i32 15, i32 12, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_05_06_04_03_01_02_08_15_13_14_12_11_09_10_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_07_05_06_04_03_01_02_08_15_13_14_12_11_09_10_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [14,15,10,11,12,13,8,9,6,7,2,3,4,5,0,1]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm1[0],xmm0[1,2,3,4,5,6,7]
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_07_05_06_04_03_01_02_08_15_13_14_12_11_09_10_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [14,15,10,11,12,13,8,9,6,7,2,3,4,5,0,1]
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm1[0],xmm0[1,2,3,4,5,6,7]
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 5, i32 6, i32 4, i32 3, i32 1, i32 2, i32 8, i32 15, i32 13, i32 14, i32 12, i32 11, i32 9, i32 10, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_01_00_05_04_05_04_01_08_09_08_13_12_13_12_09_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_01_00_05_04_05_04_01_08_09_08_13_12_13_12_09_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpslldq {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm1[0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[2,3,0,1,10,11,8,9,10,11,8,9,2,3,2,3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[2,3,0,1,10,11,8,9,10,11,8,9,2,3,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_01_00_05_04_05_04_01_08_09_08_13_12_13_12_09_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpbroadcastw %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[2,3,0,1,10,11,8,9,10,11,8,9,2,3,2,3]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[2,3,0,1,10,11,8,9,10,11,8,9,2,3,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 1, i32 0, i32 5, i32 4, i32 5, i32 4, i32 1, i32 8, i32 9, i32 8, i32 13, i32 12, i32 13, i32 12, i32 9, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_05_04_01_00_05_04_01_08_13_12_09_08_13_12_09_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_05_04_01_00_05_04_01_08_13_12_09_08_13_12_09_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpslldq {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm1[0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[10,11,8,9,2,3,0,1,10,11,8,9,2,3,2,3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[10,11,8,9,2,3,0,1,10,11,8,9,2,3,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_05_04_01_00_05_04_01_08_13_12_09_08_13_12_09_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpbroadcastw %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[10,11,8,9,2,3,0,1,10,11,8,9,2,3,2,3]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[10,11,8,9,2,3,0,1,10,11,8,9,2,3,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 5, i32 4, i32 1, i32 0, i32 5, i32 4, i32 1, i32 8, i32 13, i32 12, i32 9, i32 8, i32 13, i32 12, i32 9, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_05_04_01_00_01_00_05_12_13_12_09_08_09_08_13_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_05_04_01_00_01_00_05_12_13_12_09_08_09_08_13_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[10,11,8,9,2,3,0,1,2,3,0,1,10,11,2,3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[10,11,8,9,2,3,0,1,2,3,0,1,10,11,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_05_04_01_00_01_00_05_12_13_12_09_08_09_08_13_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[10,11,8,9,2,3,0,1,2,3,0,1,10,11,2,3]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[10,11,8,9,2,3,0,1,2,3,0,1,10,11,8,9]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 5, i32 4, i32 1, i32 0, i32 1, i32 0, i32 5, i32 12, i32 13, i32 12, i32 9, i32 8, i32 9, i32 8, i32 13, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_04_04_00_00_04_04_08_08_12_12_08_08_12_12_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_04_04_00_00_04_04_08_08_12_12_08_08_12_12_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpslldq {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm1[0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,8,9,8,9,0,1,0,1,8,9,8,9,2,3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,8,9,8,9,0,1,0,1,8,9,8,9,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_04_04_00_00_04_04_08_08_12_12_08_08_12_12_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpbroadcastw %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,8,9,8,9,0,1,0,1,8,9,8,9,2,3]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,8,9,8,9,0,1,0,1,8,9,8,9,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 4, i32 4, i32 0, i32 0, i32 4, i32 4, i32 8, i32 8, i32 12, i32 12, i32 8, i32 8, i32 12, i32 12, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_00_00_04_04_00_00_12_12_08_08_12_12_08_08_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_00_00_04_04_00_00_12_12_08_08_12_12_08_08_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[8,9,0,1,0,1,8,9,8,9,0,1,0,1,2,3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[8,9,0,1,0,1,8,9,8,9,0,1,0,1,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_00_00_04_04_00_00_12_12_08_08_12_12_08_08_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[8,9,0,1,0,1,8,9,8,9,0,1,0,1,2,3]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[8,9,0,1,0,1,8,9,8,9,0,1,0,1,8,9]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 0, i32 0, i32 4, i32 4, i32 0, i32 0, i32 12, i32 12, i32 8, i32 8, i32 12, i32 12, i32 8, i32 8, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_02_06_04_00_05_01_07_11_10_14_12_08_13_09_15_11(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_02_06_04_00_05_01_07_11_10_14_12_08_13_09_15_11:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [4,5,12,13,8,9,0,1,10,11,2,3,14,15,6,7]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3],xmm0[4,5,6,7]
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_02_06_04_00_05_01_07_11_10_14_12_08_13_09_15_11:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [4,5,12,13,8,9,0,1,10,11,2,3,14,15,6,7]
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3],xmm0[4,5,6,7]
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 2, i32 6, i32 4, i32 0, i32 5, i32 1, i32 7, i32 11, i32 10, i32 14, i32 12, i32 8, i32 13, i32 9, i32 15, i32 11>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_02_00_06_04_05_01_07_11_10_08_14_12_13_09_15_11(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_02_00_06_04_05_01_07_11_10_08_14_12_13_09_15_11:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [4,5,0,1,12,13,8,9,10,11,2,3,14,15,6,7]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3],xmm0[4,5,6,7]
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_02_00_06_04_05_01_07_11_10_08_14_12_13_09_15_11:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [4,5,0,1,12,13,8,9,10,11,2,3,14,15,6,7]
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3],xmm0[4,5,6,7]
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 2, i32 0, i32 6, i32 4, i32 5, i32 1, i32 7, i32 11, i32 10, i32 8, i32 14, i32 12, i32 13, i32 9, i32 15, i32 11>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_02_06_04_00_01_03_07_13_10_14_12_08_09_11_15_13(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_02_06_04_00_01_03_07_13_10_14_12_08_09_11_15_13:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [4,5,12,13,8,9,0,1,2,3,6,7,14,15,10,11]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4],xmm1[5],xmm0[6,7]
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_02_06_04_00_01_03_07_13_10_14_12_08_09_11_15_13:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [4,5,12,13,8,9,0,1,2,3,6,7,14,15,10,11]
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4],xmm1[5],xmm0[6,7]
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 2, i32 6, i32 4, i32 0, i32 1, i32 3, i32 7, i32 13, i32 10, i32 14, i32 12, i32 8, i32 9, i32 11, i32 15, i32 13>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_06_06_07_05_01_06_04_11_14_14_15_13_09_14_12_11(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_06_06_07_05_01_06_04_11_14_14_15_13_09_14_12_11:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [12,13,12,13,14,15,10,11,2,3,12,13,8,9,6,7]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5,6,7]
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_06_06_07_05_01_06_04_11_14_14_15_13_09_14_12_11:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [12,13,12,13,14,15,10,11,2,3,12,13,8,9,6,7]
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2,3]
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 6, i32 6, i32 7, i32 5, i32 1, i32 6, i32 4, i32 11, i32 14, i32 14, i32 15, i32 13, i32 9, i32 14, i32 12, i32 11>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_04_04_04_04_04_12_08_08_12_12_12_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_00_04_04_04_04_04_12_08_08_12_12_12_12_12_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,8,9,8,9,8,9,8,9,8,9,14,15]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,0,1,8,9,8,9,8,9,8,9,8,9,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_00_04_04_04_04_04_12_08_08_12_12_12_12_12_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,8,9,8,9,8,9,8,9,8,9,14,15]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,0,1,8,9,8,9,8,9,8,9,8,9,8,9]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 4, i32 12, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_04_00_00_04_04_04_12_12_12_08_08_12_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_04_00_00_04_04_04_12_12_12_08_08_12_12_12_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[8,9,8,9,0,1,0,1,8,9,8,9,8,9,14,15]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[8,9,8,9,0,1,0,1,8,9,8,9,8,9,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_04_00_00_04_04_04_12_12_12_08_08_12_12_12_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[8,9,8,9,0,1,0,1,8,9,8,9,8,9,14,15]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[8,9,8,9,0,1,0,1,8,9,8,9,8,9,8,9]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 4, i32 0, i32 0, i32 4, i32 4, i32 4, i32 12, i32 12, i32 12, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_04_04_00_04_04_04_12_08_12_12_08_12_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_04_04_00_04_04_04_12_08_12_12_08_12_12_12_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,8,9,8,9,0,1,8,9,8,9,8,9,14,15]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,8,9,8,9,0,1,8,9,8,9,8,9,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_04_04_00_04_04_04_12_08_12_12_08_12_12_12_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,8,9,8,9,0,1,8,9,8,9,8,9,14,15]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,8,9,8,9,0,1,8,9,8,9,8,9,8,9]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 4, i32 4, i32 0, i32 4, i32 4, i32 4, i32 12, i32 8, i32 12, i32 12, i32 8, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_04_04_00_00_00_00_08_08_12_12_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_04_04_00_00_00_00_08_08_12_12_08_08_08_08_08:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpslldq {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm1[0,1]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,8,9,8,9,0,1,0,1,0,1,0,1,14,15]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,8,9,8,9,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_04_04_00_00_00_00_08_08_12_12_08_08_08_08_08:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpbroadcastw %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,8,9,8,9,0,1,0,1,0,1,0,1,14,15]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,8,9,8,9,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 12, i32 12, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_04_04_00_04_05_06_15_08_12_12_08_12_13_14_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_04_04_00_04_05_06_15_08_12_12_08_12_13_14_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,2,2,0,4,5,6,7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm1[7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,2,2,3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,2,2,0,4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_04_04_00_04_05_06_15_08_12_12_08_12_13_14_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,2,2,0,4,5,6,7]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm1[7]
; AVX2-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,2,2,3]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,2,2,0,4,5,6,7]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 4, i32 4, i32 0, i32 4, i32 5, i32 6, i32 15, i32 8, i32 12, i32 12, i32 8, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_uu_04_04_04_04_04_12_08_uu_12_12_12_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_uu_04_04_04_04_04_12_08_uu_12_12_12_12_12_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,2,3,8,9,8,9,8,9,8,9,8,9,14,15]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,2,3,8,9,8,9,8,9,8,9,8,9,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_uu_04_04_04_04_04_12_08_uu_12_12_12_12_12_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,2,3,8,9,8,9,8,9,8,9,8,9,14,15]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,2,3,8,9,8,9,8,9,8,9,8,9,8,9]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 undef, i32 4, i32 4, i32 4, i32 4, i32 4, i32 12, i32 8, i32 undef, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_04_uu_00_04_04_04_12_12_12_uu_08_12_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_04_uu_00_04_04_04_12_12_12_uu_08_12_12_12_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[8,9,8,9,8,9,0,1,8,9,8,9,8,9,14,15]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[8,9,8,9,8,9,0,1,8,9,8,9,8,9,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_04_uu_00_04_04_04_12_12_12_uu_08_12_12_12_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[8,9,8,9,8,9,0,1,8,9,8,9,8,9,14,15]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[8,9,8,9,8,9,0,1,8,9,8,9,8,9,8,9]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 4, i32 undef, i32 0, i32 4, i32 4, i32 4, i32 12, i32 12, i32 12, i32 undef, i32 8, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_04_04_00_04_04_04_12_uu_12_12_08_12_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_04_04_00_04_04_04_12_uu_12_12_08_12_12_12_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,8,9,8,9,0,1,8,9,8,9,8,9,14,15]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,8,9,8,9,0,1,8,9,8,9,8,9,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_04_04_00_04_04_04_12_uu_12_12_08_12_12_12_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpsllq $48, %xmm1, %xmm2
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,8,9,8,9,0,1,8,9,8,9,8,9,14,15]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,8,9,8,9,0,1,8,9,8,9,8,9,8,9]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 4, i32 4, i32 0, i32 4, i32 4, i32 4, i32 12, i32 undef, i32 12, i32 12, i32 8, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_01_02_07_uu_uu_uu_uu_08_09_10_15_uu_uu_uu_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_01_02_07_uu_uu_uu_uu_08_09_10_15_uu_uu_uu_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,2,3,4,5,14,15,4,5,14,15,12,13,14,15]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_01_02_07_uu_uu_uu_uu_08_09_10_15_uu_uu_uu_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,2,3,4,5,14,15,4,5,14,15,12,13,14,15,16,17,18,19,20,21,30,31,20,21,30,31,28,29,30,31]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 8, i32 9, i32 10, i32 15, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_uu_uu_uu_04_05_06_11_uu_uu_uu_uu_12_13_14_11(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_uu_uu_uu_04_05_06_11_uu_uu_uu_uu_12_13_14_11:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm2 = xmm1[0,1,0,1]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[12,13,6,7,4,5,6,7,8,9,10,11,12,13,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_uu_uu_uu_04_05_06_11_uu_uu_uu_uu_12_13_14_11:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpbroadcastq %xmm1, %xmm2
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm2[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[12,13,6,7,4,5,6,7,8,9,10,11,12,13,6,7]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 4, i32 5, i32 6, i32 11, i32 undef, i32 undef, i32 undef, i32 undef, i32 12, i32 13, i32 14, i32 11>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_05_06_03_uu_uu_uu_uu_12_13_14_11_uu_uu_uu_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_05_06_03_uu_uu_uu_uu_12_13_14_11_uu_uu_uu_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [8,9,10,11,12,13,6,7,8,9,10,11,0,1,2,3]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_05_06_03_uu_uu_uu_uu_12_13_14_11_uu_uu_uu_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[8,9,10,11,12,13,6,7,8,9,10,11,0,1,2,3,24,25,26,27,28,29,22,23,24,25,26,27,16,17,18,19]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 5, i32 6, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 12, i32 13, i32 14, i32 11, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_01_02_07_04_05_06_11_08_09_10_15_12_13_14_11(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_01_02_07_04_05_06_11_08_09_10_15_12_13_14_11:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,2,3,4,5,14,15,8,9,10,11,12,13,6,7]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3],xmm0[4,5,6,7]
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_01_02_07_04_05_06_11_08_09_10_15_12_13_14_11:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,2,3,4,5,14,15,8,9,10,11,12,13,6,7]
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3],xmm0[4,5,6,7]
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 7, i32 4, i32 5, i32 6, i32 11, i32 8, i32 9, i32 10, i32 15, i32 12, i32 13, i32 14, i32 11>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_05_06_03_00_01_02_15_12_13_14_11_08_09_10_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_05_06_03_00_01_02_15_12_13_14_11_08_09_10_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[8,9,10,11,12,13,6,7,0,1,2,3,4,5,2,3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm1[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[8,9,10,11,12,13,6,7,0,1,2,3,4,5,14,15]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_05_06_03_00_01_02_15_12_13_14_11_08_09_10_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[8,9,10,11,12,13,6,7,0,1,2,3,4,5,2,3]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm1[7]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[8,9,10,11,12,13,6,7,0,1,2,3,4,5,14,15]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 5, i32 6, i32 3, i32 0, i32 1, i32 2, i32 15, i32 12, i32 13, i32 14, i32 11, i32 8, i32 9, i32 10, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_03_07_01_00_02_07_03_13_11_15_09_08_10_15_11_13(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_03_07_01_00_02_07_03_13_11_15_09_08_10_15_11_13:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [6,7,14,15,2,3,0,1,4,5,14,15,6,7,10,11]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5],xmm0[6,7]
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_03_07_01_00_02_07_03_13_11_15_09_08_10_15_11_13:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [6,7,14,15,2,3,0,1,4,5,14,15,6,7,10,11]
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0,1],xmm1[2],xmm0[3]
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 3, i32 7, i32 1, i32 0, i32 2, i32 7, i32 3, i32 13, i32 11, i32 15, i32 9, i32 8, i32 10, i32 15, i32 11, i32 13>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_16_01_17_02_18_03_27_08_24_09_25_10_26_11_27(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_16_01_17_02_18_03_27_08_24_09_25_10_26_11_27:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,4,5,8,9,14,15,14,15,8,9,12,13,14,15]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_16_01_17_02_18_03_27_08_24_09_25_10_26_11_27:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm3 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,2,2,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,6,7]
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm1, %ymm1
; AVX2-NEXT:    vpunpcklwd {{.*#+}} ymm0 = ymm0[0,0,1,1,2,2,3,3,8,8,9,9,10,10,11,11]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7],ymm0[8],ymm1[9],ymm0[10],ymm1[11],ymm0[12],ymm1[13],ymm0[14],ymm1[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 27, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_20_01_21_02_22_03_31_08_28_09_29_10_30_11_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_20_01_21_02_22_03_31_08_28_09_29_10_30_11_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpshufd {{.*#+}} xmm4 = xmm3[2,3,0,1]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1],xmm2[2],xmm4[2],xmm2[3],xmm4[3]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,4,5,8,9,14,15,14,15,8,9,12,13,14,15]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_20_01_21_02_22_03_31_08_28_09_29_10_30_11_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3],ymm0[4,5],ymm1[6,7]
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm1 = [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15]
; AVX2-NEXT:    vpshufb %xmm1, %xmm0, %xmm2
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpblendw {{.*#+}} xmm2 = xmm2[0,1,2,3,4,5,6],xmm0[7]
; AVX2-NEXT:    vpshufb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 20, i32 1, i32 21, i32 2, i32 22, i32 3, i32 31, i32 8, i32 28, i32 9, i32 29, i32 10, i32 30, i32 11, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_20_05_21_06_22_07_31_12_28_13_29_14_30_15_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_20_05_21_06_22_07_31_12_28_13_29_14_30_15_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm3 = xmm3[4],xmm2[4],xmm3[5],xmm2[5],xmm3[6],xmm2[6],xmm3[7],xmm2[7]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[4,5,0,1,4,5,4,5,0,1,4,5,8,9,14,15]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_20_05_21_06_22_07_31_12_28_13_29_14_30_15_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm3 = xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,2,2,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,4,6,7]
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm1, %ymm1
; AVX2-NEXT:    vpunpckhwd {{.*#+}} ymm0 = ymm0[4,4,5,5,6,6,7,7,12,12,13,13,14,14,15,15]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7],ymm0[8],ymm1[9],ymm0[10],ymm1[11],ymm0[12],ymm1[13],ymm0[14],ymm1[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 31, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_16_05_17_06_18_07_27_12_24_13_25_14_26_15_27(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_16_05_17_06_18_07_27_12_24_13_25_14_26_15_27:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpshufd {{.*#+}} xmm3 = xmm3[2,3,0,1]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[4,5,0,1,4,5,4,5,0,1,4,5,8,9,14,15]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_16_05_17_06_18_07_27_12_24_13_25_14_26_15_27:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3],ymm1[4,5],ymm0[6,7]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3],xmm0[4,5,6,7]
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [8,9,0,1,10,11,2,3,12,13,4,5,14,15,6,7]
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 16, i32 5, i32 17, i32 6, i32 18, i32 7, i32 27, i32 12, i32 24, i32 13, i32 25, i32 14, i32 26, i32 15, i32 27>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_16_01_17_06_22_07_31_08_24_09_25_14_30_15_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_16_01_17_06_22_07_31_08_24_09_25_14_30_15_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vpshufd {{.*#+}} xmm3 = xmm2[0,3,2,3]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm4
; AVX1-NEXT:    vpshufd {{.*#+}} xmm4 = xmm4[0,3,2,3]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm3 = xmm4[0],xmm3[0],xmm4[1],xmm3[1],xmm4[2],xmm3[2],xmm4[3],xmm3[3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,2,1,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,6,6,7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,5],xmm2[6,7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[2,3,6,7,10,11,14,15,14,15,10,11,12,13,14,15]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,3,2,3]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_16_01_17_06_22_07_31_08_24_09_25_14_30_15_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,2,1,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,6,6,7]
; AVX2-NEXT:    vpblendd {{.*#+}} xmm1 = xmm1[0,1,2],xmm2[3]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm2 = xmm2[0,0,2,1,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm2 = xmm2[0,1,2,3,4,6,6,7]
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[0,1,1,3,4,5,6,7,8,9,9,11,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,6,5,7,7,8,9,10,11,14,13,15,15]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7],ymm0[8],ymm1[9],ymm0[10],ymm1[11],ymm0[12],ymm1[13],ymm0[14],ymm1[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 6, i32 22, i32 7, i32 31, i32 8, i32 24, i32 9, i32 25, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_20_01_21_06_16_07_25_08_28_09_29_14_24_15_25(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_20_01_21_06_16_07_25_08_28_09_29_14_24_15_25:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vpshufd {{.*#+}} xmm3 = xmm2[2,0,2,3]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm4
; AVX1-NEXT:    vpshufd {{.*#+}} xmm4 = xmm4[0,3,2,3]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm3 = xmm4[0],xmm3[0],xmm4[1],xmm3[1],xmm4[2],xmm3[2],xmm4[3],xmm3[3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm1[0],xmm2[1],xmm1[2,3,4,5,6,7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[8,9,10,11,0,1,2,3,2,3,0,1,12,13,2,3]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,3,2,3]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_20_01_21_06_16_07_25_08_28_09_29_14_24_15_25:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm3 = [8,9,8,9,4,5,10,11,0,1,0,1,12,13,2,3]
; AVX2-NEXT:    vpshufb %xmm3, %xmm2, %xmm4
; AVX2-NEXT:    vpblendw {{.*#+}} xmm1 = xmm1[0],xmm2[1],xmm1[2,3,4,5,6,7]
; AVX2-NEXT:    vpshufb %xmm3, %xmm1, %xmm1
; AVX2-NEXT:    vinserti128 $1, %xmm4, %ymm1, %ymm1
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[0,1,1,3,4,5,6,7,8,9,9,11,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,6,5,7,7,8,9,10,11,14,13,15,15]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7],ymm0[8],ymm1[9],ymm0[10],ymm1[11],ymm0[12],ymm1[13],ymm0[14],ymm1[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 20, i32 1, i32 21, i32 6, i32 16, i32 7, i32 25, i32 8, i32 28, i32 9, i32 29, i32 14, i32 24, i32 15, i32 25>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_01_00_17_16_03_02_19_26_09_08_25_24_11_10_27_26(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_01_00_17_16_03_02_19_26_09_08_25_24_11_10_27_26:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[4,5,0,1,12,13,10,11,8,9,10,11,12,13,10,11]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm3 = xmm0[1,0,3,2,4,5,6,7]
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm1 = xmm3[0],xmm1[0],xmm3[1],xmm1[1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[2,0,3,1,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,6,4,7,5]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_01_00_17_16_03_02_19_26_09_08_25_24_11_10_27_26:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,1,2,0,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,5,6,5]
; AVX2-NEXT:    vpshufb {{.*#+}} xmm2 = xmm2[0,1,2,3,2,3,0,1,8,9,10,11,6,7,4,5]
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[2,3,0,1,4,5,6,7,6,7,4,5,4,5,6,7,18,19,16,17,20,21,22,23,22,23,20,21,20,21,22,23]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3],ymm0[4],ymm1[5],ymm0[6],ymm1[7]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 1, i32 0, i32 17, i32 16, i32 3, i32 2, i32 19, i32 26, i32 9, i32 8, i32 25, i32 24, i32 11, i32 10, i32 27, i32 26>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_16_00_17_01_18_02_19_11_24_08_25_09_26_10_27_11(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_16_00_17_01_18_02_19_11_24_08_25_09_26_10_27_11:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,4,5,8,9,14,15,14,15,8,9,12,13,14,15]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_16_00_17_01_18_02_19_11_24_08_25_09_26_10_27_11:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm2
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm3 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; AVX2-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,2,2,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,6,7]
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; AVX2-NEXT:    vpunpcklwd {{.*#+}} ymm1 = ymm1[0],ymm0[0],ymm1[1],ymm0[1],ymm1[2],ymm0[2],ymm1[3],ymm0[3],ymm1[8],ymm0[8],ymm1[9],ymm0[9],ymm1[10],ymm0[10],ymm1[11],ymm0[11]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm1[0],ymm0[1],ymm1[2],ymm0[3],ymm1[4],ymm0[5],ymm1[6],ymm0[7],ymm1[8],ymm0[9],ymm1[10],ymm0[11],ymm1[12],ymm0[13],ymm1[14],ymm0[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 16, i32 0, i32 17, i32 1, i32 18, i32 2, i32 19, i32 11, i32 24, i32 8, i32 25, i32 9, i32 26, i32 10, i32 27, i32 11>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_20_04_21_05_22_06_23_15_28_12_29_13_30_14_31_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_20_04_21_05_22_06_23_15_28_12_29_13_30_14_31_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm3 = xmm3[4],xmm2[4],xmm3[5],xmm2[5],xmm3[6],xmm2[6],xmm3[7],xmm2[7]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[4,5,0,1,4,5,4,5,0,1,4,5,8,9,14,15]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_20_04_21_05_22_06_23_15_28_12_29_13_30_14_31_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm2
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm3 = xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; AVX2-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,2,2,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,4,6,7]
; AVX2-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; AVX2-NEXT:    vpunpckhwd {{.*#+}} ymm1 = ymm1[4],ymm0[4],ymm1[5],ymm0[5],ymm1[6],ymm0[6],ymm1[7],ymm0[7],ymm1[12],ymm0[12],ymm1[13],ymm0[13],ymm1[14],ymm0[14],ymm1[15],ymm0[15]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm1[0],ymm0[1],ymm1[2],ymm0[3],ymm1[4],ymm0[5],ymm1[6],ymm0[7],ymm1[8],ymm0[9],ymm1[10],ymm0[11],ymm1[12],ymm0[13],ymm1[14],ymm0[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 20, i32 4, i32 21, i32 5, i32 22, i32 6, i32 23, i32 15, i32 28, i32 12, i32 29, i32 13, i32 30, i32 14, i32 31, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_02_01_03_20_22_21_31_08_10_09_11_28_30_29_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_02_01_03_20_22_21_31_08_10_09_11_28_30_29_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm2 = xmm2[0,2,1,3,4,5,6,7]
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpshufd {{.*#+}} xmm4 = xmm3[2,3,2,3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm4 = xmm4[0,2,1,3,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm4[0]
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,8,9,4,5,14,15,0,1,4,5,4,5,6,7]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,2,1,3,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_02_01_03_20_22_21_31_08_10_09_11_28_30_29_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3],ymm0[4,5],ymm1[6,7]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,2,1,3,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,6,5,7]
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm1[7]
; AVX2-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,2,1,3,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,4,6,5,7]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 2, i32 1, i32 3, i32 20, i32 22, i32 21, i32 31, i32 8, i32 10, i32 9, i32 11, i32 28, i32 30, i32 29, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_04_03_18_uu_uu_uu_uu_12_12_11_26_uu_uu_uu_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_04_04_03_18_uu_uu_uu_uu_12_12_11_26_uu_uu_uu_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm0[0,1],xmm1[2],xmm0[3,4,5,6,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm2 = xmm2[2,1,2,3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm2 = xmm2[0,0,3,2,4,5,6,7]
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2],xmm0[3,4,5,6,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,1,2,3]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,0,3,2,4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_04_04_03_18_uu_uu_uu_uu_12_12_11_26_uu_uu_uu_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0,1],ymm1[2],ymm0[3,4,5,6,7,8,9],ymm1[10],ymm0[11,12,13,14,15]
; AVX2-NEXT:    vpshufd {{.*#+}} ymm0 = ymm0[2,1,2,3,6,5,6,7]
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[0,0,3,2,4,5,6,7,8,8,11,10,12,13,14,15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 4, i32 3, i32 18, i32 undef, i32 undef, i32 undef, i32 undef, i32 12, i32 12, i32 11, i32 26, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_03_02_21_uu_uu_uu_uu_08_11_10_29_uu_uu_uu_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_03_02_21_uu_uu_uu_uu_08_11_10_29_uu_uu_uu_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm3[0,1,2,3],xmm2[4,5,6,7]
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [0,1,6,7,4,5,10,11,0,1,10,11,0,1,2,3]
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; AVX1-NEXT:    vpshufb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_03_02_21_uu_uu_uu_uu_08_11_10_29_uu_uu_uu_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3],ymm0[4,5],ymm1[6,7]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,6,7,4,5,10,11,0,1,10,11,0,1,2,3,16,17,22,23,20,21,26,27,16,17,26,27,16,17,18,19]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 3, i32 2, i32 21, i32 undef, i32 undef, i32 undef, i32 undef, i32 8, i32 11, i32 10, i32 29, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_uu_uu_21_uu_uu_uu_uu_uu_uu_uu_29_uu_uu_uu_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_uu_uu_21_uu_uu_uu_uu_uu_uu_uu_29_uu_uu_uu_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpermilps {{.*#+}} ymm0 = ymm1[0,2,2,3,4,6,6,7]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_uu_uu_21_uu_uu_uu_uu_uu_uu_uu_29_uu_uu_uu_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufd {{.*#+}} ymm0 = ymm1[0,2,2,3,4,6,6,7]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 21, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 29, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_01_02_21_uu_uu_uu_uu_08_09_10_29_uu_uu_uu_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_01_02_21_uu_uu_uu_uu_08_09_10_29_uu_uu_uu_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpshufd {{.*#+}} xmm3 = xmm3[2,2,3,3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm2[0,1,2],xmm3[3],xmm2[4,5,6,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[2,2,3,3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3],xmm0[4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_01_02_21_uu_uu_uu_uu_08_09_10_29_uu_uu_uu_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufd {{.*#+}} ymm1 = ymm1[0,2,2,3,4,6,6,7]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0,1,2],ymm1[3],ymm0[4,5,6,7,8,9,10],ymm1[11],ymm0[12,13,14,15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 21, i32 undef, i32 undef, i32 undef, i32 undef, i32 8, i32 9, i32 10, i32 29, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_uu_uu_uu_20_21_22_11_uu_uu_uu_uu_28_29_30_11(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_uu_uu_uu_20_21_22_11_uu_uu_uu_uu_28_29_30_11:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,1,0,1]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm2[0,1,2,3,4,5,6],xmm0[7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm1[0,1,2,3,4,5,6],xmm0[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_uu_uu_uu_20_21_22_11_uu_uu_uu_uu_28_29_30_11:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,2,2,2]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm1[0,1,2,3,4,5,6],ymm0[7],ymm1[8,9,10,11,12,13,14],ymm0[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 20, i32 21, i32 22, i32 11, i32 undef, i32 undef, i32 undef, i32 undef, i32 28, i32 29, i32 30, i32 11>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_20_21_22_03_uu_uu_uu_uu_28_29_30_11_uu_uu_uu_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_20_21_22_03_uu_uu_uu_uu_28_29_30_11_uu_uu_uu_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpshufd {{.*#+}} xmm3 = xmm3[2,3,0,1]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm3[0,1,2],xmm2[3],xmm3[4,5,6,7]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm1[0,1,2],xmm0[3],xmm1[4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_20_21_22_03_uu_uu_uu_uu_28_29_30_11_uu_uu_uu_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufd {{.*#+}} ymm1 = ymm1[2,3,2,3,6,7,6,7]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm1[0,1,2],ymm0[3],ymm1[4,5,6,7,8,9,10],ymm0[11],ymm1[12,13,14,15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 20, i32 21, i32 22, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 28, i32 29, i32 30, i32 11, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_01_02_21_20_21_22_11_08_09_10_29_28_29_30_11(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_01_02_21_20_21_22_11_08_09_10_29_28_29_30_11:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX1-NEXT:    vpshufd {{.*#+}} xmm3 = xmm1[0,2,2,3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm3[3,4,5,6],xmm0[7]
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm2[0,1,2,3],xmm1[4,5,6,7]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,2,3,4,5,10,11,8,9,10,11,12,13,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_01_02_21_20_21_22_11_08_09_10_29_28_29_30_11:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3],ymm0[4,5],ymm1[6,7]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3],xmm0[4,5,6,7]
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,1,2,3,4,5,10,11,8,9,10,11,12,13,6,7]
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 21, i32 20, i32 21, i32 22, i32 11, i32 8, i32 9, i32 10, i32 29, i32 28, i32 29, i32 30, i32 11>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_17_02_03_20_21_22_15_08_25_10_11_28_29_30_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_00_17_02_03_20_21_22_15_08_25_10_11_28_29_30_15:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm3[0],xmm2[1],xmm3[2,3],xmm2[4,5,6],xmm3[7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm3[4,5,6,7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2,3],xmm1[4,5,6],xmm0[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_00_17_02_03_20_21_22_15_08_25_10_11_28_29_30_15:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,3,2,3]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2,3],ymm1[4,5,6],ymm0[7,8],ymm1[9],ymm0[10,11],ymm1[12,13,14],ymm0[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 17, i32 2, i32 3, i32 20, i32 21, i32 22, i32 15, i32 8, i32 25, i32 10, i32 11, i32 28, i32 29, i32 30, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_uu_uu_01_uu_05_07_25_uu_uu_uu_09_uu_13_15_25(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_uu_uu_01_uu_05_07_25_uu_uu_uu_09_uu_13_15_25:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[0,1,2,0]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm2 = xmm2[0,1,2,1,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm2 = xmm2[0,1,2,3,4,5,7,7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm2[0,1,2,3,4,5,6],xmm1[7]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[0,1,2,1,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,7,7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_uu_uu_01_uu_05_07_25_uu_uu_uu_09_uu_13_15_25:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm2
; AVX2-NEXT:    vpermd %ymm1, %ymm2, %ymm1
; AVX2-NEXT:    vpshuflw {{.*#+}} ymm0 = ymm0[0,1,2,1,4,5,6,7,8,9,10,9,12,13,14,15]
; AVX2-NEXT:    vpshufhw {{.*#+}} ymm0 = ymm0[0,1,2,3,4,5,7,7,8,9,10,11,12,13,15,15]
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0,1,2,3,4,5,6],ymm1[7],ymm0[8,9,10,11,12,13,14],ymm1[15]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 1, i32 undef, i32 5, i32 7, i32 25, i32 undef, i32 undef, i32 undef, i32 9, i32 undef, i32 13, i32 15, i32 25>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_uu_04_uu_16_18_20_uu_uu_uu_12_uu_24_26_28_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_uu_04_uu_16_18_20_uu_uu_uu_12_uu_24_26_28_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [0,1,4,5,4,5,6,7,0,1,4,5,8,9,4,5]
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm4
; AVX1-NEXT:    vpshufd {{.*#+}} xmm4 = xmm4[2,2,3,3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm4[0,1,2,3],xmm2[4,5,6,7]
; AVX1-NEXT:    vpshufb %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,2,3,3]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_uu_04_uu_16_18_20_uu_uu_uu_12_uu_24_26_28_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[0,1,4,5,4,5,6,7,0,1,4,5,8,9,4,5,16,17,20,21,20,21,22,23,16,17,20,21,24,25,20,21]
; AVX2-NEXT:    vpshufd {{.*#+}} ymm0 = ymm0[0,2,2,3,4,6,6,7]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3],ymm0[4,5],ymm1[6,7]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 undef, i32 4, i32 undef, i32 16, i32 18, i32 20, i32 undef, i32 undef, i32 undef, i32 12, i32 undef, i32 24, i32 26, i32 28, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_21_22_23_00_01_02_03_12_29_30_31_08_09_10_11_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_21_22_23_00_01_02_03_12_29_30_31_08_09_10_11_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[10,11,12,13,14,15],xmm3[0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm3[4,5,6,7]
; AVX1-NEXT:    vpslldq {{.*#+}} xmm0 = zero,zero,zero,zero,zero,zero,xmm0[0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vpsrldq {{.*#+}} xmm1 = xmm1[10,11,12,13,14,15],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm1[0,1,2],xmm0[3,4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_21_22_23_00_01_02_03_12_29_30_31_08_09_10_11_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0,1,2,3,4],ymm1[5,6,7],ymm0[8,9,10,11,12],ymm1[13,14,15]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4],xmm0[5,6,7]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm1 = xmm1[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 12, i32 29, i32 30, i32 31, i32 8, i32 9, i32 10, i32 11, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_22_uu_uu_01_02_03_uu_uu_30_uu_uu_09_10_11_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_22_uu_uu_01_02_03_uu_uu_30_uu_uu_09_10_11_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[10,11,12,13,14,15],xmm3[0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm1[10,11,12,13,14,15],xmm0[0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_22_uu_uu_01_02_03_uu_uu_30_uu_uu_09_10_11_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm1[10,11,12,13,14,15],ymm0[0,1,2,3,4,5,6,7,8,9],ymm1[26,27,28,29,30,31],ymm0[16,17,18,19,20,21,22,23,24,25]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 22, i32 undef, i32 undef, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 30, i32 undef, i32 undef, i32 9, i32 10, i32 11, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_05_06_07_00_01_02_03_12_13_14_15_08_09_10_11_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_05_06_07_00_01_02_03_12_13_14_15_08_09_10_11_12:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4],xmm0[5,6,7]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm1 = xmm1[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_05_06_07_00_01_02_03_12_13_14_15_08_09_10_11_12:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4],xmm0[5,6,7]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm1 = xmm1[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_06_uu_uu_01_02_03_uu_uu_14_uu_uu_09_10_11_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_06_uu_uu_01_02_03_uu_uu_14_uu_uu_09_10_11_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpalignr {{.*#+}} xmm1 = xmm0[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_06_uu_uu_01_02_03_uu_uu_14_uu_uu_09_10_11_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm0[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,26,27,28,29,30,31,16,17,18,19,20,21,22,23,24,25]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 6, i32 undef, i32 undef, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 14, i32 undef, i32 undef, i32 9, i32 10, i32 11, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_uu_uu_uu_01_02_03_uu_uu_uu_uu_uu_09_10_11_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_uu_uu_uu_01_02_03_uu_uu_uu_uu_uu_09_10_11_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpslldq {{.*#+}} xmm1 = zero,zero,zero,zero,zero,zero,xmm0[0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpslldq {{.*#+}} xmm0 = zero,zero,zero,zero,zero,zero,xmm0[0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_uu_uu_uu_01_02_03_uu_uu_uu_uu_uu_09_10_11_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpslldq {{.*#+}} ymm0 = zero,zero,zero,zero,zero,zero,ymm0[0,1,2,3,4,5,6,7,8,9],zero,zero,zero,zero,zero,zero,ymm0[16,17,18,19,20,21,22,23,24,25]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 9, i32 10, i32 11, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_19_20_21_22_23_00_01_10_27_28_29_30_31_08_09_10(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_19_20_21_22_23_00_01_10_27_28_29_30_31_08_09_10:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[6,7,8,9,10,11,12,13,14,15],xmm3[0,1,2,3,4,5]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,4,5,4,5,6,7,8,9,0,1,4,5,10,11]
; AVX1-NEXT:    vpsrldq {{.*#+}} xmm1 = xmm1[6,7,8,9,10,11,12,13,14,15],zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm1[0,1,2,3,4],xmm0[5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_19_20_21_22_23_00_01_10_27_28_29_30_31_08_09_10:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0,1,2],ymm1[3,4,5,6,7],ymm0[8,9,10],ymm1[11,12,13,14,15]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2],xmm0[3,4,5,6,7]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm1 = xmm1[6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 10, i32 27, i32 28, i32 29, i32 30, i32 31, i32 8, i32 9, i32 10>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_20_21_22_uu_uu_01_uu_uu_28_29_30_uu_uu_09_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_20_21_22_uu_uu_01_uu_uu_28_29_30_uu_uu_09_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[6,7,8,9,10,11,12,13,14,15],xmm3[0,1,2,3,4,5]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm1[6,7,8,9,10,11,12,13,14,15],xmm0[0,1,2,3,4,5]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_20_21_22_uu_uu_01_uu_uu_28_29_30_uu_uu_09_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm1[6,7,8,9,10,11,12,13,14,15],ymm0[0,1,2,3,4,5],ymm1[22,23,24,25,26,27,28,29,30,31],ymm0[16,17,18,19,20,21]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 20, i32 21, i32 22, i32 undef, i32 undef, i32 1, i32 undef, i32 undef, i32 28, i32 29, i32 30, i32 undef, i32 undef, i32 9, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_03_04_05_06_07_00_01_10_11_12_13_14_15_08_09_10(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_03_04_05_06_07_00_01_10_11_12_13_14_15_08_09_10:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2],xmm0[3,4,5,6,7]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm1 = xmm1[6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_03_04_05_06_07_00_01_10_11_12_13_14_15_08_09_10:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2],xmm0[3,4,5,6,7]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm1 = xmm1[6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_04_05_06_uu_uu_01_uu_uu_12_13_14_uu_uu_09_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_04_05_06_uu_uu_01_uu_uu_12_13_14_uu_uu_09_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpalignr {{.*#+}} xmm1 = xmm0[6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_04_05_06_uu_uu_01_uu_uu_12_13_14_uu_uu_09_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm0[6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5,22,23,24,25,26,27,28,29,30,31,16,17,18,19,20,21]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 4, i32 5, i32 6, i32 undef, i32 undef, i32 1, i32 undef, i32 undef, i32 12, i32 13, i32 14, i32 undef, i32 undef, i32 9, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_04_05_06_uu_uu_uu_uu_uu_12_13_14_uu_uu_uu_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_04_05_06_uu_uu_uu_uu_uu_12_13_14_uu_uu_uu_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpsrldq {{.*#+}} xmm1 = xmm0[6,7,8,9,10,11,12,13,14,15],zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpsrldq {{.*#+}} xmm0 = xmm0[6,7,8,9,10,11,12,13,14,15],zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_04_05_06_uu_uu_uu_uu_uu_12_13_14_uu_uu_uu_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsrldq {{.*#+}} ymm0 = ymm0[6,7,8,9,10,11,12,13,14,15],zero,zero,zero,zero,zero,zero,ymm0[22,23,24,25,26,27,28,29,30,31],zero,zero,zero,zero,zero,zero
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 4, i32 5, i32 6, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 12, i32 13, i32 14, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_03_04_05_06_07_16_17_26_11_12_13_14_15_24_25_26(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_03_04_05_06_07_16_17_26_11_12_13_14_15_24_25_26:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[6,7,8,9,10,11,12,13,14,15],xmm3[0,1,2,3,4,5]
; AVX1-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1,4,5,4,5,6,7,8,9,0,1,4,5,10,11]
; AVX1-NEXT:    vpsrldq {{.*#+}} xmm0 = xmm0[6,7,8,9,10,11,12,13,14,15],zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4],xmm1[5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_03_04_05_06_07_16_17_26_11_12_13_14_15_24_25_26:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm1[0,1,2],ymm0[3,4,5,6,7],ymm1[8,9,10],ymm0[11,12,13,14,15]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2],xmm0[3,4,5,6,7]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm1 = xmm1[6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 26, i32 11, i32 12, i32 13, i32 14, i32 15, i32 24, i32 25, i32 26>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_04_05_06_uu_uu_17_uu_uu_12_13_14_uu_uu_25_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_04_05_06_uu_uu_17_uu_uu_12_13_14_uu_uu_25_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[6,7,8,9,10,11,12,13,14,15],xmm3[0,1,2,3,4,5]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[6,7,8,9,10,11,12,13,14,15],xmm1[0,1,2,3,4,5]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_04_05_06_uu_uu_17_uu_uu_12_13_14_uu_uu_25_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm0[6,7,8,9,10,11,12,13,14,15],ymm1[0,1,2,3,4,5],ymm0[22,23,24,25,26,27,28,29,30,31],ymm1[16,17,18,19,20,21]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 4, i32 5, i32 6, i32 undef, i32 undef, i32 17, i32 undef, i32 undef, i32 12, i32 13, i32 14, i32 undef, i32 undef, i32 25, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_05_06_07_16_17_18_19_28_13_14_15_24_25_26_27_28(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_05_06_07_16_17_18_19_28_13_14_15_24_25_26_27_28:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[10,11,12,13,14,15],xmm3[0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm1[0,1,2,3],xmm3[4,5,6,7]
; AVX1-NEXT:    vpslldq {{.*#+}} xmm1 = zero,zero,zero,zero,zero,zero,xmm1[0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vpsrldq {{.*#+}} xmm0 = xmm0[10,11,12,13,14,15],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2],xmm1[3,4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_05_06_07_16_17_18_19_28_13_14_15_24_25_26_27_28:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm1[0,1,2,3,4],ymm0[5,6,7],ymm1[8,9,10,11,12],ymm0[13,14,15]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4],xmm0[5,6,7]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9]
; AVX2-NEXT:    vpalignr {{.*#+}} xmm1 = xmm1[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 28, i32 13, i32 14, i32 15, i32 24, i32 25, i32 26, i32 27, i32 28>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_uu_06_uu_uu_17_18_19_uu_uu_14_uu_uu_25_26_27_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_uu_06_uu_uu_17_18_19_uu_uu_14_uu_uu_25_26_27_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpalignr {{.*#+}} xmm2 = xmm2[10,11,12,13,14,15],xmm3[0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vpalignr {{.*#+}} xmm0 = xmm0[10,11,12,13,14,15],xmm1[0,1,2,3,4,5,6,7,8,9]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_uu_06_uu_uu_17_18_19_uu_uu_14_uu_uu_25_26_27_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpalignr {{.*#+}} ymm0 = ymm0[10,11,12,13,14,15],ymm1[0,1,2,3,4,5,6,7,8,9],ymm0[26,27,28,29,30,31],ymm1[16,17,18,19,20,21,22,23,24,25]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 undef, i32 6, i32 undef, i32 undef, i32 17, i32 18, i32 19, i32 undef, i32 undef, i32 14, i32 undef, i32 undef, i32 25, i32 26, i32 27, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_23_uu_03_uu_20_20_05_uu_31_uu_11_uu_28_28_13_uu(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: shuffle_v16i16_23_uu_03_uu_20_20_05_uu_31_uu_11_uu_28_28_13_uu:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpslldq {{.*#+}} xmm2 = zero,zero,xmm2[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm3 = xmm3[0,1,2,3,7,5,4,4]
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm2 = xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; AVX1-NEXT:    vpslldq {{.*#+}} xmm0 = zero,zero,xmm0[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
; AVX1-NEXT:    vpshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,7,5,4,4]
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm0 = xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v16i16_23_uu_03_uu_20_20_05_uu_31_uu_11_uu_28_28_13_uu:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0,1,2,3],ymm1[4],ymm0[5,6],ymm1[7],ymm0[8,9,10,11],ymm1[12],ymm0[13,14],ymm1[15]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[14,15,14,15,6,7,6,7,8,9,8,9,10,11,14,15,30,31,30,31,22,23,22,23,24,25,24,25,26,27,30,31]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 23, i32 undef, i32 3, i32 undef, i32 20, i32 20, i32 5, i32 undef, i32 31, i32 undef, i32 11, i32 undef, i32 28, i32 28, i32 13, i32 undef>
  ret <16 x i16> %shuffle
}

define <16 x i16> @insert_v16i16_0elt_into_zero_vector(i16* %ptr) {
; ALL-LABEL: insert_v16i16_0elt_into_zero_vector:
; ALL:       # BB#0:
; ALL-NEXT:    movzwl (%rdi), %eax
; ALL-NEXT:    vmovd %eax, %xmm0
; ALL-NEXT:    retq
  %val = load i16, i16* %ptr
  %i0 = insertelement <16 x i16> zeroinitializer, i16 %val, i32 0
  ret <16 x i16> %i0
}

define <16 x i16> @concat_v16i16_0_1_2_3_4_5_6_7_24_25_26_27_28_29_30_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: concat_v16i16_0_1_2_3_4_5_6_7_24_25_26_27_28_29_30_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: concat_v16i16_0_1_2_3_4_5_6_7_24_25_26_27_28_29_30_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; AVX2-NEXT:    retq
  %alo = shufflevector <16 x i16> %a, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %bhi = shufflevector <16 x i16> %b, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuf = shufflevector <8 x i16> %alo, <8 x i16> %bhi, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i16> %shuf
}

define <16 x i16> @concat_v16i16_8_9_10_11_12_13_14_15_24_25_26_27_28_29_30_31_bc(<16 x i16> %a, <16 x i16> %b) {
; ALL-LABEL: concat_v16i16_8_9_10_11_12_13_14_15_24_25_26_27_28_29_30_31_bc:
; ALL:       # BB#0:
; ALL-NEXT:    vperm2f128 {{.*#+}} ymm0 = ymm0[2,3],ymm1[2,3]
; ALL-NEXT:    retq
  %ahi = shufflevector <16 x i16> %a, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %bhi = shufflevector <16 x i16> %b, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %bc0hi = bitcast <8 x i16> %ahi to <16 x i8>
  %bc1hi = bitcast <8 x i16> %bhi to <16 x i8>
  %shuffle8 = shufflevector <16 x i8> %bc0hi, <16 x i8> %bc1hi, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %shuffle16 = bitcast <32 x i8> %shuffle8 to <16 x i16>
  ret <16 x i16> %shuffle16
}

define <16 x i16> @insert_dup_mem_v16i16_i32(i32* %ptr) {
; AVX1-LABEL: insert_dup_mem_v16i16_i32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vmovd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: insert_dup_mem_v16i16_i32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastw (%rdi), %ymm0
; AVX2-NEXT:    retq
  %tmp = load i32, i32* %ptr, align 4
  %tmp1 = insertelement <4 x i32> zeroinitializer, i32 %tmp, i32 0
  %tmp2 = bitcast <4 x i32> %tmp1 to <8 x i16>
  %tmp3 = shufflevector <8 x i16> %tmp2, <8 x i16> undef, <16 x i32> zeroinitializer
  ret <16 x i16> %tmp3
}

define <16 x i16> @insert_dup_mem_v16i16_sext_i16(i16* %ptr) {
; AVX1-LABEL: insert_dup_mem_v16i16_sext_i16:
; AVX1:       # BB#0:
; AVX1-NEXT:    movswl (%rdi), %eax
; AVX1-NEXT:    vmovd %eax, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: insert_dup_mem_v16i16_sext_i16:
; AVX2:       # BB#0:
; AVX2-NEXT:    movswl (%rdi), %eax
; AVX2-NEXT:    vmovd %eax, %xmm0
; AVX2-NEXT:    vpbroadcastw %xmm0, %ymm0
; AVX2-NEXT:    retq
  %tmp = load i16, i16* %ptr, align 2
  %tmp1 = sext i16 %tmp to i32
  %tmp2 = insertelement <4 x i32> zeroinitializer, i32 %tmp1, i32 0
  %tmp3 = bitcast <4 x i32> %tmp2 to <8 x i16>
  %tmp4 = shufflevector <8 x i16> %tmp3, <8 x i16> undef, <16 x i32> zeroinitializer
  ret <16 x i16> %tmp4
}
