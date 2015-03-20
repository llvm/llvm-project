; RUN: llc < %s -mcpu=x86-64 -mattr=+avx512f -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx512bw -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512BW

target triple = "x86_64-unknown-unknown"

define <8 x double> @shuffle_v8f64_00000000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00000000:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastsd %xmm0, %zmm0 
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00000010(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00000010:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm1
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[0,0,1,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00000200(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00000200:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm1
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[0,2,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00003000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00003000:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm1
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[3,0,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00040000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00040000:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vbroadcastsd %xmm1, %ymm1
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm0
; ALL-NEXT:    vblendpd {{.*#+}} ymm1 = ymm0[0,1,2],ymm1[3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00500000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00500000:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm1[0,1,1,3]
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm0
; ALL-NEXT:    vblendpd {{.*#+}} ymm1 = ymm0[0,1],ymm1[2],ymm0[3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_06000000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_06000000:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm1[0,2,2,3]
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm0
; ALL-NEXT:    vblendpd {{.*#+}} ymm1 = ymm0[0],ymm1[1],ymm0[2,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_70000000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_70000000:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm1[3,1,2,3]
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm0
; ALL-NEXT:    vblendpd {{.*#+}} ymm1 = ymm1[0],ymm0[1,2,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01014545(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_01014545:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm1, %ymm1
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 4, i32 5, i32 4, i32 5>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00112233(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00112233:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[0,0,1,1]
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[2,2,3,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00001111(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00001111:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm1
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[1,1,1,1]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_81a3c5e7(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_81a3c5e7:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm2 
; ALL-NEXT:    vextractf64x4 $1, %zmm1, %ymm3 
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm3[0],ymm2[1],ymm3[2],ymm2[3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm1[0],ymm0[1],ymm1[2],ymm0[3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08080808(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_08080808:
; ALL:       # BB#0:
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vbroadcastsd %xmm1, %ymm1
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 0, i32 8, i32 0, i32 8>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08084c4c(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_08084c4c:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm2 
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm2, %ymm2
; ALL-NEXT:    vextractf64x4 $1, %zmm1, %ymm3 
; ALL-NEXT:    vbroadcastsd %xmm3, %ymm3
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm2[0],ymm3[1],ymm2[2],ymm3[3]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vbroadcastsd %xmm1, %ymm1
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 4, i32 12, i32 4, i32 12>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_8823cc67(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_8823cc67:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm2 
; ALL-NEXT:    vextractf64x4 $1, %zmm1, %ymm3 
; ALL-NEXT:    vbroadcastsd %xmm3, %ymm3
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm3[0,1],ymm2[2,3]
; ALL-NEXT:    vbroadcastsd %xmm1, %ymm1
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 8, i32 8, i32 2, i32 3, i32 12, i32 12, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_9832dc76(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_9832dc76:
; ALL:       # BB#0:
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm1[0,1],ymm0[2,3]
; ALL-NEXT:    vpermilpd {{.*#+}} ymm2 = ymm2[1,0,3,2]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0
; ALL-NEXT:    vextractf64x4 $1, %zmm1, %ymm1
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3]
; ALL-NEXT:    vpermilpd {{.*#+}} ymm0 = ymm0[1,0,3,2]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 9, i32 8, i32 3, i32 2, i32 13, i32 12, i32 7, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_9810dc54(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_9810dc54:
; ALL:       # BB#0:
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm2
; ALL-NEXT:    vpermilpd {{.*#+}} ymm2 = ymm2[1,0,3,2]
; ALL-NEXT:    vextractf64x4 $1, %zmm1, %ymm1
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    vpermilpd {{.*#+}} ymm0 = ymm0[1,0,3,2]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 9, i32 8, i32 1, i32 0, i32 13, i32 12, i32 5, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08194c5d(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_08194c5d:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm1, %ymm2 
; ALL-NEXT:    vpermpd {{.*#+}} ymm2 = ymm2[0,0,2,1]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm3 
; ALL-NEXT:    vpermpd {{.*#+}} ymm3 = ymm3[0,1,1,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm3[0],ymm2[1],ymm3[2],ymm2[3]
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm1[0,0,2,1]
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[0,1,1,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_2a3b6e7f(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_2a3b6e7f:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm1, %ymm2 
; ALL-NEXT:    vpermpd {{.*#+}} ymm2 = ymm2[0,2,2,3]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm3 
; ALL-NEXT:    vpermpd {{.*#+}} ymm3 = ymm3[2,1,3,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm3[0],ymm2[1],ymm3[2],ymm2[3]
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm1[0,2,2,3]
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[2,1,3,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08192a3b(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_08192a3b:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm2 = ymm1[0,2,2,3]
; ALL-NEXT:    vpermpd {{.*#+}} ymm3 = ymm0[2,1,3,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm3[0],ymm2[1],ymm3[2],ymm2[3]
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm1[0,0,2,1]
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[0,1,1,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08991abb(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_08991abb:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm2 = ymm0[1,0,2,2]
; ALL-NEXT:    vpermpd {{.*#+}} ymm3 = ymm1[0,2,3,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm2[0],ymm3[1,2,3]
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm1[0,0,1,1]
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 9, i32 9, i32 1, i32 10, i32 11, i32 11>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_091b2d3f(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_091b2d3f:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm1, %ymm2 
; ALL-NEXT:    vpermpd {{.*#+}} ymm3 = ymm0[2,1,3,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm3[0],ymm2[1],ymm3[2],ymm2[3]
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[0,1,1,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 9, i32 1, i32 11, i32 2, i32 13, i32 3, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_09ab1def(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_09ab1def:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm1, %ymm2 
; ALL-NEXT:    vpermilpd {{.*#+}} ymm3 = ymm0[1,0,2,2]
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm3[0],ymm2[1,2,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 1, i32 13, i32 14, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00014445(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00014445:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[0,0,0,1]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[0,0,0,1]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 4, i32 4, i32 4, i32 5>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00204464(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00204464:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[0,0,2,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[0,0,2,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 4, i32 6, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_03004744(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_03004744:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[0,3,0,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[0,3,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 7, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10005444(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10005444:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[1,0,0,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[1,0,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_22006644(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_22006644:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[2,2,0,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[2,2,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_33307774(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_33307774:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[3,3,3,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[3,3,3,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 7, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_32107654(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_32107654:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[3,2,1,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[3,2,1,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00234467(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00234467:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[0,0,2,3]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermilpd {{.*#+}} ymm0 = ymm0[0,0,2,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 4, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00224466(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00224466:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[0,0,2,2]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermilpd {{.*#+}} ymm0 = ymm0[0,0,2,2]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10325476(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10325476:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[1,0,3,2]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermilpd {{.*#+}} ymm0 = ymm0[1,0,3,2]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_11335577(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_11335577:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[1,1,3,3]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermilpd {{.*#+}} ymm0 = ymm0[1,1,3,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10235467(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10235467:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[1,0,2,3]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermilpd {{.*#+}} ymm0 = ymm0[1,0,2,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10225466(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10225466:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[1,0,2,2]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermilpd {{.*#+}} ymm0 = ymm0[1,0,2,2]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 2, i32 5, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00015444(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00015444:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[0,0,0,1]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[1,0,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 5, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00204644(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00204644:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[0,0,2,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[0,2,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_03004474(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_03004474:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[0,3,0,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[0,0,3,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 4, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10004444(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10004444:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[1,0,0,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_22006446(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_22006446:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[2,2,0,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[2,0,0,2]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 4, i32 4, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_33307474(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_33307474:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[3,3,3,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[3,0,3,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 4, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_32104567(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_32104567:
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm0[3,2,1,0]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00236744(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00236744:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[0,0,2,3]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[2,3,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 6, i32 7, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00226644(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00226644:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[0,0,2,2]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[2,2,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 6, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10324567(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10324567:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[1,0,3,2]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_11334567(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_11334567:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[1,1,3,3]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01235467(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_01235467:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm1[1,0,2,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01235466(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_01235466:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm1[1,0,2,2]
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_002u6u44(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_002u6u44:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[0,0,2,2]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[2,1,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00uu66uu(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00uu66uu:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm1
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[2,2,2,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 undef, i32 undef, i32 6, i32 6, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_103245uu(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_103245uu:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[1,0,3,2]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_1133uu67(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_1133uu67:
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm0[1,1,3,3]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 undef, i32 undef, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_0uu354uu(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_0uu354uu:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm1[1,0,2,2]
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 undef, i32 undef, i32 3, i32 5, i32 4, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_uuu3uu66(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_uuu3uu66:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermilpd {{.*#+}} ymm1 = ymm1[0,0,2,2]
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_c348cda0(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_c348cda0:
; ALL:       # BB#0:
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm2 
; ALL-NEXT:    vperm2f128 {{.*#+}} ymm2 = ymm0[2,3],ymm2[0,1]
; ALL-NEXT:    vextractf64x4 $1, %zmm1, %ymm3 
; ALL-NEXT:    vbroadcastsd %xmm1, %ymm4
; ALL-NEXT:    vblendpd {{.*#+}} ymm4 = ymm3[0,1,2],ymm4[3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm4[0],ymm2[1,2],ymm4[3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm1 = ymm3[0,1],ymm1[2],ymm3[3]
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm0
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm1[0,1,2],ymm0[3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 12, i32 3, i32 4, i32 8, i32 12, i32 13, i32 10, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_f511235a(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_f511235a:
; ALL:       # BB#0:
; ALL-NEXT:    vperm2f128 {{.*#+}} ymm2 = ymm0[2,3,0,1]
; ALL-NEXT:    vextractf64x4 $1, %zmm0, %ymm3 
; ALL-NEXT:    vpermpd {{.*#+}} ymm4 = ymm3[0,1,1,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm2[0,1],ymm4[2],ymm2[3]
; ALL-NEXT:    vpermilpd {{.*#+}} ymm4 = ymm1[0,0,2,2]
; ALL-NEXT:    vblendpd {{.*#+}} ymm2 = ymm2[0,1,2],ymm4[3]
; ALL-NEXT:    vpermpd {{.*#+}} ymm0 = ymm0[0,1,1,1]
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm3[1],ymm0[2,3]
; ALL-NEXT:    vextractf64x4 $1, %zmm1, %ymm1 
; ALL-NEXT:    vpermpd {{.*#+}} ymm1 = ymm1[3,1,2,3]
; ALL-NEXT:    vblendpd {{.*#+}} ymm0 = ymm1[0],ymm0[1,2,3]
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 15, i32 5, i32 1, i32 1, i32 2, i32 3, i32 5, i32 10>
  ret <8 x double> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00000000:
; ALL:       # BB#0:
; ALL-NEXT:    vpbroadcastq %xmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000010(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00000010:
; ALL:       # BB#0:
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm1
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,0,1,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000200(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00000200:
; ALL:       # BB#0:
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm1
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,2,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00003000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00003000:
; ALL:       # BB#0:
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm1
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[3,0,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00040000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00040000:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpbroadcastq %xmm1, %ymm1
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm0
; ALL-NEXT:    vpblendd {{.*#+}} ymm1 = ymm0[0,1,2,3,4,5],ymm1[6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00500000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00500000:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm1[0,1,1,3]
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm0
; ALL-NEXT:    vpblendd {{.*#+}} ymm1 = ymm0[0,1,2,3],ymm1[4,5],ymm0[6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_06000000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_06000000:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm1[0,2,2,3]
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm0
; ALL-NEXT:    vpblendd {{.*#+}} ymm1 = ymm0[0,1],ymm1[2,3],ymm0[4,5,6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_70000000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_70000000:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm1[3,1,2,3]
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm0
; ALL-NEXT:    vpblendd {{.*#+}} ymm1 = ymm1[0,1],ymm0[2,3,4,5,6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01014545(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_01014545:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm1, %ymm1
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 4, i32 5, i32 4, i32 5>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00112233(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00112233:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[0,0,1,1]
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[2,2,3,3]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00001111(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00001111:
; ALL:       # BB#0:
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm1
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[1,1,1,1]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_81a3c5e7(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_81a3c5e7:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm2 
; ALL-NEXT:    vextracti64x4 $1, %zmm1, %ymm3 
; ALL-NEXT:    vpblendd {{.*#+}} ymm2 = ymm3[0,1],ymm2[2,3],ymm3[4,5],ymm2[6,7]
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3],ymm1[4,5],ymm0[6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08080808(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_08080808:
; ALL:       # BB#0:
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vpbroadcastq %xmm1, %ymm1
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3],ymm0[4,5],ymm1[6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 0, i32 8, i32 0, i32 8>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08084c4c(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_08084c4c:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm2 
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm2, %ymm2
; ALL-NEXT:    vextracti64x4 $1, %zmm1, %ymm3 
; ALL-NEXT:    vpbroadcastq %xmm3, %ymm3
; ALL-NEXT:    vpblendd {{.*#+}} ymm2 = ymm2[0,1],ymm3[2,3],ymm2[4,5],ymm3[6,7]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vpbroadcastq %xmm1, %ymm1
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3],ymm0[4,5],ymm1[6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 4, i32 12, i32 4, i32 12>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_8823cc67(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_8823cc67:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm2 
; ALL-NEXT:    vextracti64x4 $1, %zmm1, %ymm3 
; ALL-NEXT:    vpbroadcastq %xmm3, %ymm3
; ALL-NEXT:    vpblendd {{.*#+}} ymm2 = ymm3[0,1,2,3],ymm2[4,5,6,7]
; ALL-NEXT:    vpbroadcastq %xmm1, %ymm1
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 8, i32 8, i32 2, i32 3, i32 12, i32 12, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_9832dc76(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_9832dc76:
; ALL:       # BB#0:
; ALL-NEXT:    vpblendd {{.*#+}} ymm2 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    vpshufd {{.*#+}} ymm2 = ymm2[2,3,0,1,6,7,4,5]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0
; ALL-NEXT:    vextracti64x4 $1, %zmm1, %ymm1
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    vpshufd {{.*#+}} ymm0 = ymm0[2,3,0,1,6,7,4,5]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 9, i32 8, i32 3, i32 2, i32 13, i32 12, i32 7, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_9810dc54(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_9810dc54:
; ALL:       # BB#0:
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm2
; ALL-NEXT:    vpshufd {{.*#+}} ymm2 = ymm2[2,3,0,1,6,7,4,5]
; ALL-NEXT:    vextracti64x4 $1, %zmm1, %ymm1
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    vpshufd {{.*#+}} ymm0 = ymm0[2,3,0,1,6,7,4,5]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 9, i32 8, i32 1, i32 0, i32 13, i32 12, i32 5, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08194c5d(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_08194c5d:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm1, %ymm2 
; ALL-NEXT:    vpermq {{.*#+}} ymm2 = ymm2[0,0,2,1]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm3 
; ALL-NEXT:    vpermq {{.*#+}} ymm3 = ymm3[0,1,1,3]
; ALL-NEXT:    vpblendd {{.*#+}} ymm2 = ymm3[0,1],ymm2[2,3],ymm3[4,5],ymm2[6,7]
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm1[0,0,2,1]
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,1,1,3]
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3],ymm0[4,5],ymm1[6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_2a3b6e7f(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_2a3b6e7f:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm1, %ymm2 
; ALL-NEXT:    vpermq {{.*#+}} ymm2 = ymm2[0,2,2,3]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm3 
; ALL-NEXT:    vpermq {{.*#+}} ymm3 = ymm3[2,1,3,3]
; ALL-NEXT:    vpblendd {{.*#+}} ymm2 = ymm3[0,1],ymm2[2,3],ymm3[4,5],ymm2[6,7]
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm1[0,2,2,3]
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[2,1,3,3]
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3],ymm0[4,5],ymm1[6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08192a3b(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_08192a3b:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm2 = ymm1[0,2,2,3]
; ALL-NEXT:    vpermq {{.*#+}} ymm3 = ymm0[2,1,3,3]
; ALL-NEXT:    vpblendd {{.*#+}} ymm2 = ymm3[0,1],ymm2[2,3],ymm3[4,5],ymm2[6,7]
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm1[0,0,2,1]
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,1,1,3]
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3],ymm0[4,5],ymm1[6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08991abb(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_08991abb:
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*#+}} ymm2 = ymm0[2,3,2,3,6,7,6,7]
; ALL-NEXT:    vpermq {{.*#+}} ymm3 = ymm1[0,2,3,3]
; ALL-NEXT:    vpblendd {{.*#+}} ymm2 = ymm2[0,1],ymm3[2,3,4,5,6,7]
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm1[0,0,1,1]
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3,4,5,6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 9, i32 9, i32 1, i32 10, i32 11, i32 11>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_091b2d3f(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_091b2d3f:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm1, %ymm2 
; ALL-NEXT:    vpermq {{.*#+}} ymm3 = ymm0[2,1,3,3]
; ALL-NEXT:    vpblendd {{.*#+}} ymm2 = ymm3[0,1],ymm2[2,3],ymm3[4,5],ymm2[6,7]
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,1,1,3]
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3],ymm0[4,5],ymm1[6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 9, i32 1, i32 11, i32 2, i32 13, i32 3, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_09ab1def(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_09ab1def:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm1, %ymm2 
; ALL-NEXT:    vpshufd {{.*#+}} ymm3 = ymm0[2,3,2,3,6,7,6,7]
; ALL-NEXT:    vpblendd {{.*#+}} ymm2 = ymm3[0,1],ymm2[2,3,4,5,6,7]
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3,4,5,6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 1, i32 13, i32 14, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00014445(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00014445:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[0,0,0,1]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,0,0,1]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 4, i32 4, i32 4, i32 5>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00204464(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00204464:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[0,0,2,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,0,2,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 4, i32 6, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_03004744(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_03004744:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[0,3,0,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,3,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 7, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10005444(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10005444:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[1,0,0,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[1,0,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_22006644(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_22006644:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[2,2,0,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[2,2,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_33307774(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_33307774:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[3,3,3,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[3,3,3,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 7, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_32107654(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_32107654:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[3,2,1,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[3,2,1,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00234467(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00234467:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[0,0,2,3]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,0,2,3]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 4, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00224466(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00224466:
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm0[0,1,0,1,4,5,4,5]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpshufd {{.*#+}} ymm0 = ymm0[0,1,0,1,4,5,4,5]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10325476(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10325476:
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm0[2,3,0,1,6,7,4,5]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpshufd {{.*#+}} ymm0 = ymm0[2,3,0,1,6,7,4,5]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_11335577(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_11335577:
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm0[2,3,2,3,6,7,6,7]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpshufd {{.*#+}} ymm0 = ymm0[2,3,2,3,6,7,6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10235467(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10235467:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[1,0,2,3]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[1,0,2,3]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10225466(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10225466:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[1,0,2,2]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[1,0,2,2]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 2, i32 5, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00015444(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00015444:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[0,0,0,1]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[1,0,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 5, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00204644(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00204644:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[0,0,2,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,2,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_03004474(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_03004474:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[0,3,0,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,0,3,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 4, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10004444(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10004444:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[1,0,0,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_22006446(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_22006446:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[2,2,0,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[2,0,0,2]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 4, i32 4, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_33307474(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_33307474:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[3,3,3,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[3,0,3,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 4, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_32104567(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_32104567:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[3,2,1,0]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00236744(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00236744:
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm0[0,0,2,3]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[2,3,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 6, i32 7, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00226644(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00226644:
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm0[0,1,0,1,4,5,4,5]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[2,2,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 6, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10324567(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10324567:
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm0[2,3,0,1,6,7,4,5]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_11334567(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_11334567:
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm0[2,3,2,3,6,7,6,7]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01235467(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_01235467:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm1[1,0,2,3]
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01235466(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_01235466:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpermq {{.*#+}} ymm1 = ymm1[1,0,2,2]
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_002u6u44(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_002u6u44:
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm0[0,1,0,1,4,5,4,5]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[2,1,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00uu66uu(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00uu66uu:
; ALL:       # BB#0:
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm1
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[2,2,2,3]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 undef, i32 undef, i32 6, i32 6, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_103245uu(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_103245uu:
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm0[2,3,0,1,6,7,4,5]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_1133uu67(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_1133uu67:
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm0[2,3,2,3,6,7,6,7]
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 undef, i32 undef, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_0uu354uu(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_0uu354uu:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm1[2,3,0,1,6,7,4,5]
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 undef, i32 undef, i32 3, i32 5, i32 4, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_uuu3uu66(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_uuu3uu66:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm1 
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm1[0,1,0,1,4,5,4,5]
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_6caa87e5(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_6caa87e5:
; ALL:       # BB#0:
; ALL-NEXT:    vextracti64x4 $1, %zmm0, %ymm0 
; ALL-NEXT:    vperm2i128 {{.*#+}} ymm0 = ymm0[2,3,0,1]
; ALL-NEXT:    vextracti64x4 $1, %zmm1, %ymm2
; ALL-NEXT:    vpblendd {{.*#+}} ymm3 = ymm1[0,1,2,3],ymm2[4,5],ymm1[6,7]
; ALL-NEXT:    vpblendd {{.*#+}} ymm3 = ymm3[0,1],ymm0[2,3],ymm3[4,5],ymm0[6,7]
; ALL-NEXT:    vpblendd {{.*#+}} ymm1 = ymm2[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    vpshufd {{.*#+}} ymm1 = ymm1[0,1,0,1,4,5,4,5]
; ALL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3,4,5,6,7]
; ALL-NEXT:    vinserti64x4 $1, %ymm3, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 6, i32 12, i32 10, i32 10, i32 8, i32 7, i32 14, i32 5>
  ret <8 x i64> %shuffle
}

define <8 x double> @shuffle_v8f64_082a4c6e(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_082a4c6e:
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklpd {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32><i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x double> %shuffle
}

define <8 x i64> @shuffle_v8i64_082a4c6e(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_082a4c6e:
; ALL:       # BB#0:
; ALL-NEXT:    vpunpcklqdq {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32><i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i64> %shuffle
}

define <8 x double> @shuffle_v8f64_193b5d7f(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_193b5d7f:
; ALL:       # BB#0:
; ALL-NEXT:    vunpckhpd {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32><i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x double> %shuffle
}

define <8 x i64> @shuffle_v8i64_193b5d7f(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_193b5d7f:
; ALL:       # BB#0:
; ALL-NEXT:    vpunpckhqdq {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32><i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i64> %shuffle
}
