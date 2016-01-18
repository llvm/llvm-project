; RUN: llc < %s -mcpu=x86-64 -mattr=+avx512f | FileCheck %s --check-prefix=ALL --check-prefix=AVX512F
; RUN: llc < %s -mtriple=i386-unknown-linux-gnu -mattr=+avx512f | FileCheck %s --check-prefix=ALL --check-prefix=AVX512F-32

target triple = "x86_64-unknown-unknown"

define <8 x double> @shuffle_v8f64_00000000(<8 x double> %a, <8 x double> %b) {
; AVX512F-LABEL: shuffle_v8f64_00000000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vbroadcastsd %xmm0, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00000000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vbroadcastsd %xmm0, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00000010(<8 x double> %a, <8 x double> %b) {
; AVX512F-LABEL: shuffle_v8f64_00000010:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,1,0]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00000010:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00000200(<8 x double> %a, <8 x double> %b) {
; AVX512F-LABEL: shuffle_v8f64_00000200:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,2,0,0]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00000200:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00003000(<8 x double> %a, <8 x double> %b) {
; AVX512F-LABEL: shuffle_v8f64_00003000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,3,0,0,0]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00003000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00040000(<8 x double> %a, <8 x double> %b) {
; AVX512F-LABEL: shuffle_v8f64_00040000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,4,0,0,0,0]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00040000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00500000(<8 x double> %a, <8 x double> %b) {
; AVX512F-LABEL: shuffle_v8f64_00500000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,5,0,0,0,0,0]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00500000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_06000000(<8 x double> %a, <8 x double> %b) {
; AVX512F-LABEL: shuffle_v8f64_06000000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,6,0,0,0,0,0,0]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_06000000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_70000000(<8 x double> %a, <8 x double> %b) {
; AVX512F-LABEL: shuffle_v8f64_70000000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; AVX512F-NEXT:    movl $7, %eax
; AVX512F-NEXT:    vpinsrq $0, %rax, %xmm1, %xmm2
; AVX512F-NEXT:    vinserti32x4 $0, %xmm2, %zmm1, %zmm1
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_70000000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX512F-32-NEXT:    movl $7, %eax
; AVX512F-32-NEXT:    vpinsrd $0, %eax, %xmm1, %xmm1
; AVX512F-32-NEXT:    vpxord %zmm2, %zmm2, %zmm2
; AVX512F-32-NEXT:    vinserti32x4 $0, %xmm1, %zmm2, %zmm1
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01014545(<8 x double> %a, <8 x double> %b) {
; AVX512F-LABEL: shuffle_v8f64_01014545:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,0,1,4,5,4,5]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_01014545:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,0,1,4,5,4,5]
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 4, i32 5, i32 4, i32 5>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00112233(<8 x double> %a, <8 x double> %b) {
; AVX512F-LABEL: shuffle_v8f64_00112233:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,1,1,2,2,3,3]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00112233:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,1,0,1,0,2,0,2,0,3,0,3,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00001111(<8 x double> %a, <8 x double> %b) {
; AVX512F-LABEL: shuffle_v8f64_00001111:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,1,1,1,1]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00001111:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_81a3c5e7(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_81a3c5e7:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,9,2,11,4,13,6,15]
; AVX512F-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_81a3c5e7:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,9,0,2,0,11,0,4,0,13,0,6,0,15,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08080808(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_08080808:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,0,8,0,8,0,8]
; AVX512F-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_08080808:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,8,0,0,0,8,0,0,0,8,0,0,0,8,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 0, i32 8, i32 0, i32 8>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08084c4c(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_08084c4c:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,0,8,4,12,4,12]
; AVX512F-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_08084c4c:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,8,0,0,0,8,0,4,0,12,0,4,0,12,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 4, i32 12, i32 4, i32 12>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_8823cc67(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_8823cc67:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,10,11,4,4,14,15]
; AVX512F-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_8823cc67:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,0,0,10,0,11,0,4,0,4,0,14,0,15,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 8, i32 8, i32 2, i32 3, i32 12, i32 12, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_9832dc76(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_9832dc76:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,11,10,5,4,15,14]
; AVX512F-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_9832dc76:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,0,0,11,0,10,0,5,0,4,0,15,0,14,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 9, i32 8, i32 3, i32 2, i32 13, i32 12, i32 7, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_9810dc54(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_9810dc54:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,9,8,5,4,13,12]
; AVX512F-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_9810dc54:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,0,0,9,0,8,0,5,0,4,0,13,0,12,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 9, i32 8, i32 1, i32 0, i32 13, i32 12, i32 5, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08194c5d(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_08194c5d:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,1,9,4,12,5,13]
; AVX512F-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_08194c5d:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,8,0,1,0,9,0,4,0,12,0,5,0,13,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_2a3b6e7f(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_2a3b6e7f:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [2,10,3,11,6,14,7,15]
; AVX512F-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_2a3b6e7f:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [2,0,10,0,3,0,11,0,6,0,14,0,7,0,15,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08192a3b(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_08192a3b:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,1,9,2,10,3,11]
; AVX512F-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_08192a3b:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,8,0,1,0,9,0,2,0,10,0,3,0,11,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08991abb(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_08991abb:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,0,1,1,9,2,3,3]
; AVX512F-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_08991abb:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,0,0,0,1,0,1,0,9,0,2,0,3,0,3,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 9, i32 9, i32 1, i32 10, i32 11, i32 11>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_091b2d3f(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_091b2d3f:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,9,1,11,2,13,3,15]
; AVX512F-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_091b2d3f:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,9,0,1,0,11,0,2,0,13,0,3,0,15,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 9, i32 1, i32 11, i32 2, i32 13, i32 3, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_09ab1def(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_09ab1def:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,1,2,3,9,5,6,7]
; AVX512F-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_09ab1def:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,0,1,0,2,0,3,0,9,0,5,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 1, i32 13, i32 14, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00014445(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_00014445:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,1,4,4,4,5]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00014445:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,1,0,4,0,4,0,4,0,5,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 4, i32 4, i32 4, i32 5>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00204464(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_00204464:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,0,4,4,6,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00204464:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,0,0,4,0,4,0,6,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 4, i32 6, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_03004744(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_03004744:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,3,0,0,4,7,4,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_03004744:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,3,0,0,0,0,0,4,0,7,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 7, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10005444(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_10005444:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,5,4,4,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_10005444:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,0,0,0,0,5,0,4,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_22006644(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_22006644:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,2,0,0,6,6,4,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_22006644:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,0,2,0,0,0,0,0,6,0,6,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_33307774(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_33307774:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,3,3,0,7,7,7,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_33307774:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,0,3,0,3,0,0,0,7,0,7,0,7,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 7, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_32107654(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_32107654:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,2,1,0,7,6,5,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_32107654:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,0,2,0,1,0,0,0,7,0,6,0,5,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00234467(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_00234467:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,3,4,4,6,7]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00234467:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,3,0,4,0,4,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 4, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00224466(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_00224466:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,2,4,4,6,6]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00224466:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,2,0,4,0,4,0,6,0,6,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10325476(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_10325476:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,3,2,5,4,7,6]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_10325476:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,3,0,2,0,5,0,4,0,7,0,6,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_11335577(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_11335577:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,1,3,3,5,5,7,7]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_11335577:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,1,0,3,0,3,0,5,0,5,0,7,0,7,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10235467(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_10235467:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,2,3,5,4,6,7]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_10235467:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,2,0,3,0,5,0,4,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10225466(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_10225466:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,2,2,5,4,6,6]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_10225466:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,2,0,2,0,5,0,4,0,6,0,6,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 2, i32 5, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00015444(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_00015444:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,1,5,4,4,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00015444:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,1,0,5,0,4,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 5, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00204644(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_00204644:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,0,4,6,4,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00204644:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,0,0,4,0,6,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_03004474(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_03004474:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,3,0,0,4,4,7,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_03004474:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,3,0,0,0,0,0,4,0,4,0,7,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 4, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10004444(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_10004444:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,4,4,4,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_10004444:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,0,0,0,0,4,0,4,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_22006446(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_22006446:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,2,0,0,6,4,4,6]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_22006446:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,0,2,0,0,0,0,0,6,0,4,0,4,0,6,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 4, i32 4, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_33307474(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_33307474:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,3,3,0,7,4,7,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_33307474:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,0,3,0,3,0,0,0,7,0,4,0,7,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 4, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_32104567(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_32104567:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,2,1,0,4,5,6,7]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_32104567:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,0,2,0,1,0,0,0,4,0,5,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00236744(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_00236744:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,3,6,7,4,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00236744:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,3,0,6,0,7,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 6, i32 7, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00226644(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_00226644:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,2,6,6,4,4]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00226644:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,2,0,6,0,6,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 6, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10324567(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_10324567:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,3,2,4,5,6,7]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_10324567:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,3,0,2,0,4,0,5,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_11334567(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_11334567:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,1,3,3,4,5,6,7]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_11334567:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,1,0,3,0,3,0,4,0,5,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01235467(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_01235467:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,1,2,3,5,4,6,7]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_01235467:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,1,0,2,0,3,0,5,0,4,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01235466(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_01235466:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,1,2,3,5,4,6,6]
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_01235466:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,1,0,2,0,3,0,5,0,4,0,6,0,6,0]
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_002u6u44(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_002u6u44:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,2,u,6,u,4,4>
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_002u6u44:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,0,0,2,0,u,u,6,0,u,u,4,0,4,0>
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00uu66uu(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_00uu66uu:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,u,u,6,6,u,u>
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_00uu66uu:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,0,0,u,u,u,u,6,0,6,0,u,u,u,u>
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 undef, i32 undef, i32 6, i32 6, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_103245uu(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_103245uu:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,0,3,2,4,5,u,u>
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_103245uu:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,0,0,0,3,0,2,0,4,0,5,0,u,u,u,u>
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_1133uu67(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_1133uu67:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,1,3,3,u,u,6,7>
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_1133uu67:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,0,1,0,3,0,3,0,u,u,u,u,6,0,7,0>
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 undef, i32 undef, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_0uu354uu(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_0uu354uu:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,u,u,3,5,4,u,u>
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_0uu354uu:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,u,u,u,u,3,0,5,0,4,0,u,u,u,u>
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 undef, i32 undef, i32 3, i32 5, i32 4, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_uuu3uu66(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_uuu3uu66:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <u,u,u,3,u,u,6,6>
; AVX512F-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_uuu3uu66:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <u,u,u,u,u,u,3,0,u,u,u,u,6,0,6,0>
; AVX512F-32-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_c348cda0(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_c348cda0:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [4,11,12,0,4,5,2,8]
; AVX512F-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_c348cda0:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [4,0,11,0,12,0,0,0,4,0,5,0,2,0,8,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 12, i32 3, i32 4, i32 8, i32 12, i32 13, i32 10, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_f511235a(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_f511235a:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [15,5,1,1,2,3,5,10]
; AVX512F-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_f511235a:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [15,0,5,0,1,0,1,0,2,0,3,0,5,0,10,0]
; AVX512F-32-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 15, i32 5, i32 1, i32 1, i32 2, i32 3, i32 5, i32 10>
  ret <8 x double> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000000(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00000000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpbroadcastq %xmm0, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00000000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpbroadcastq %xmm0, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000010(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00000010:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,1,0]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00000010:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000200(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00000200:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,2,0,0]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00000200:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00003000(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00003000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,3,0,0,0]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00003000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00040000(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00040000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,4,0,0,0,0]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00040000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00500000(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00500000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,5,0,0,0,0,0]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00500000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_06000000(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_06000000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,6,0,0,0,0,0,0]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_06000000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_70000000(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_70000000:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; AVX512F-NEXT:    movl $7, %eax
; AVX512F-NEXT:    vpinsrq $0, %rax, %xmm1, %xmm2
; AVX512F-NEXT:    vinserti32x4 $0, %xmm2, %zmm1, %zmm1
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_70000000:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX512F-32-NEXT:    movl $7, %eax
; AVX512F-32-NEXT:    vpinsrd $0, %eax, %xmm1, %xmm1
; AVX512F-32-NEXT:    vpxord %zmm2, %zmm2, %zmm2
; AVX512F-32-NEXT:    vinserti32x4 $0, %xmm1, %zmm2, %zmm1
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01014545(<8 x i64> %a, <8 x i64> %b) {
; AVX512F-LABEL: shuffle_v8i64_01014545:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vshufi64x2 {{.*#+}} zmm0 = zmm0[0,1,0,1,4,5,4,5]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_01014545:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vshufi64x2 {{.*#+}} zmm0 = zmm0[0,1,0,1,4,5,4,5]
; AVX512F-32-NEXT:    retl

  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 4, i32 5, i32 4, i32 5>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00112233(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00112233:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,1,1,2,2,3,3]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00112233:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,1,0,1,0,2,0,2,0,3,0,3,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00001111(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00001111:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,1,1,1,1]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00001111:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_81a3c5e7(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_81a3c5e7:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,9,2,11,4,13,6,15]
; AVX512F-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_81a3c5e7:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,9,0,2,0,11,0,4,0,13,0,6,0,15,0]
; AVX512F-32-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08080808(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_08080808:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,0,8,0,8,0,8]
; AVX512F-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_08080808:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,8,0,0,0,8,0,0,0,8,0,0,0,8,0]
; AVX512F-32-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 0, i32 8, i32 0, i32 8>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08084c4c(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_08084c4c:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,0,8,4,12,4,12]
; AVX512F-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_08084c4c:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,8,0,0,0,8,0,4,0,12,0,4,0,12,0]
; AVX512F-32-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 4, i32 12, i32 4, i32 12>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_8823cc67(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_8823cc67:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,10,11,4,4,14,15]
; AVX512F-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_8823cc67:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,0,0,10,0,11,0,4,0,4,0,14,0,15,0]
; AVX512F-32-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 8, i32 8, i32 2, i32 3, i32 12, i32 12, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_9832dc76(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_9832dc76:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,11,10,5,4,15,14]
; AVX512F-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_9832dc76:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,0,0,11,0,10,0,5,0,4,0,15,0,14,0]
; AVX512F-32-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 9, i32 8, i32 3, i32 2, i32 13, i32 12, i32 7, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_9810dc54(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_9810dc54:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,9,8,5,4,13,12]
; AVX512F-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_9810dc54:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,0,0,9,0,8,0,5,0,4,0,13,0,12,0]
; AVX512F-32-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 9, i32 8, i32 1, i32 0, i32 13, i32 12, i32 5, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08194c5d(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_08194c5d:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,1,9,4,12,5,13]
; AVX512F-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_08194c5d:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,8,0,1,0,9,0,4,0,12,0,5,0,13,0]
; AVX512F-32-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_2a3b6e7f(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_2a3b6e7f:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [2,10,3,11,6,14,7,15]
; AVX512F-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_2a3b6e7f:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [2,0,10,0,3,0,11,0,6,0,14,0,7,0,15,0]
; AVX512F-32-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08192a3b(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_08192a3b:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,1,9,2,10,3,11]
; AVX512F-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_08192a3b:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,8,0,1,0,9,0,2,0,10,0,3,0,11,0]
; AVX512F-32-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08991abb(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_08991abb:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,0,1,1,9,2,3,3]
; AVX512F-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_08991abb:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,0,0,0,1,0,1,0,9,0,2,0,3,0,3,0]
; AVX512F-32-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 9, i32 9, i32 1, i32 10, i32 11, i32 11>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_091b2d3f(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_091b2d3f:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,9,1,11,2,13,3,15]
; AVX512F-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_091b2d3f:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,9,0,1,0,11,0,2,0,13,0,3,0,15,0]
; AVX512F-32-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 9, i32 1, i32 11, i32 2, i32 13, i32 3, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_09ab1def(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_09ab1def:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,1,2,3,9,5,6,7]
; AVX512F-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_09ab1def:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,0,1,0,2,0,3,0,9,0,5,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 1, i32 13, i32 14, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00014445(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00014445:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,1,4,4,4,5]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00014445:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,1,0,4,0,4,0,4,0,5,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 4, i32 4, i32 4, i32 5>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00204464(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00204464:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,0,4,4,6,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00204464:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,0,0,4,0,4,0,6,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 4, i32 6, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_03004744(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_03004744:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,3,0,0,4,7,4,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_03004744:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,3,0,0,0,0,0,4,0,7,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 7, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10005444(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_10005444:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,5,4,4,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_10005444:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,0,0,0,0,5,0,4,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_22006644(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_22006644:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,2,0,0,6,6,4,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_22006644:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,0,2,0,0,0,0,0,6,0,6,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_33307774(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_33307774:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,3,3,0,7,7,7,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_33307774:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,0,3,0,3,0,0,0,7,0,7,0,7,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 7, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_32107654(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_32107654:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,2,1,0,7,6,5,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_32107654:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,0,2,0,1,0,0,0,7,0,6,0,5,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00234467(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00234467:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,3,4,4,6,7]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00234467:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,3,0,4,0,4,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 4, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00224466(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00224466:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,2,4,4,6,6]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00224466:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,2,0,4,0,4,0,6,0,6,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10325476(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_10325476:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,3,2,5,4,7,6]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_10325476:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,3,0,2,0,5,0,4,0,7,0,6,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_11335577(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_11335577:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,1,3,3,5,5,7,7]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_11335577:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,1,0,3,0,3,0,5,0,5,0,7,0,7,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10235467(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_10235467:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,2,3,5,4,6,7]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_10235467:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,2,0,3,0,5,0,4,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10225466(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_10225466:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,2,2,5,4,6,6]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_10225466:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,2,0,2,0,5,0,4,0,6,0,6,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 2, i32 5, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00015444(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00015444:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,1,5,4,4,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00015444:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,1,0,5,0,4,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 5, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00204644(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00204644:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,0,4,6,4,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00204644:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,0,0,4,0,6,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_03004474(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_03004474:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,3,0,0,4,4,7,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_03004474:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,3,0,0,0,0,0,4,0,4,0,7,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 4, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10004444(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_10004444:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,4,4,4,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_10004444:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,0,0,0,0,4,0,4,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_22006446(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_22006446:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,2,0,0,6,4,4,6]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_22006446:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,0,2,0,0,0,0,0,6,0,4,0,4,0,6,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 4, i32 4, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_33307474(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_33307474:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,3,3,0,7,4,7,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_33307474:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,0,3,0,3,0,0,0,7,0,4,0,7,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 4, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_32104567(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_32104567:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,2,1,0,4,5,6,7]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_32104567:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,0,2,0,1,0,0,0,4,0,5,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00236744(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00236744:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,3,6,7,4,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00236744:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,3,0,6,0,7,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 6, i32 7, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00226644(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00226644:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,2,6,6,4,4]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00226644:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,2,0,2,0,6,0,6,0,4,0,4,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 6, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10324567(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_10324567:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,3,2,4,5,6,7]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_10324567:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,3,0,2,0,4,0,5,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_11334567(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_11334567:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,1,3,3,4,5,6,7]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_11334567:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,1,0,3,0,3,0,4,0,5,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01235467(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_01235467:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,1,2,3,5,4,6,7]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_01235467:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,1,0,2,0,3,0,5,0,4,0,6,0,7,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01235466(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_01235466:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,1,2,3,5,4,6,6]
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_01235466:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,1,0,2,0,3,0,5,0,4,0,6,0,6,0]
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_002u6u44(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_002u6u44:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,2,u,6,u,4,4>
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_002u6u44:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,0,0,2,0,u,u,6,0,u,u,4,0,4,0>
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00uu66uu(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_00uu66uu:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,u,u,6,6,u,u>
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_00uu66uu:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,0,0,u,u,u,u,6,0,6,0,u,u,u,u>
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 undef, i32 undef, i32 6, i32 6, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_103245uu(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_103245uu:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,0,3,2,4,5,u,u>
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_103245uu:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,0,0,0,3,0,2,0,4,0,5,0,u,u,u,u>
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_1133uu67(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_1133uu67:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,1,3,3,u,u,6,7>
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_1133uu67:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,0,1,0,3,0,3,0,u,u,u,u,6,0,7,0>
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 undef, i32 undef, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_0uu354uu(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_0uu354uu:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,u,u,3,5,4,u,u>
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_0uu354uu:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,u,u,u,u,3,0,5,0,4,0,u,u,u,u>
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 undef, i32 undef, i32 3, i32 5, i32 4, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_uuu3uu66(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_uuu3uu66:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <u,u,u,3,u,u,6,6>
; AVX512F-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_uuu3uu66:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <u,u,u,u,u,u,3,0,u,u,u,u,6,0,6,0>
; AVX512F-32-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_6caa87e5(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_6caa87e5:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [14,4,2,2,0,15,6,13]
; AVX512F-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_6caa87e5:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [14,0,4,0,2,0,2,0,0,0,15,0,6,0,13,0]
; AVX512F-32-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; AVX512F-32-NEXT:    vmovaps %zmm1, %zmm0
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 6, i32 12, i32 10, i32 10, i32 8, i32 7, i32 14, i32 5>
  ret <8 x i64> %shuffle
}

define <8 x double> @shuffle_v8f64_082a4c6e(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_082a4c6e:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vunpcklpd {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_082a4c6e:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vunpcklpd {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32><i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_0z2z4z6z(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_0z2z4z6z:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; AVX512F-NEXT:    vunpcklpd {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_0z2z4z6z:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; AVX512F-32-NEXT:    vunpcklpd {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> zeroinitializer, <8 x i32><i32 0, i32 8, i32 2, i32 8, i32 4, i32 8, i32 6, i32 8>
  ret <8 x double> %shuffle
}

define <8 x i64> @shuffle_v8i64_082a4c6e(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_082a4c6e:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpunpcklqdq {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_082a4c6e:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpunpcklqdq {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32><i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_z8zazcze(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_z8zazcze:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpxord %zmm0, %zmm0, %zmm0
; AVX512F-NEXT:    vpunpcklqdq {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_z8zazcze:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpxord %zmm0, %zmm0, %zmm0
; AVX512F-32-NEXT:    vpunpcklqdq {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> zeroinitializer, <8 x i64> %b, <8 x i32><i32 7, i32 8, i32 5, i32 10, i32 3, i32 12, i32 1, i32 14>
  ret <8 x i64> %shuffle
}

define <8 x double> @shuffle_v8f64_193b5d7f(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_193b5d7f:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vunpckhpd {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_193b5d7f:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vunpckhpd {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32><i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_z9zbzdzf(<8 x double> %a, <8 x double> %b) {
;
; AVX512F-LABEL: shuffle_v8f64_z9zbzdzf:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpxord %zmm0, %zmm0, %zmm0
; AVX512F-NEXT:    vunpckhpd {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8f64_z9zbzdzf:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpxord %zmm0, %zmm0, %zmm0
; AVX512F-32-NEXT:    vunpckhpd {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x double> zeroinitializer, <8 x double> %b, <8 x i32><i32 0, i32 9, i32 0, i32 11, i32 0, i32 13, i32 0, i32 15>
  ret <8 x double> %shuffle
}

define <8 x i64> @shuffle_v8i64_193b5d7f(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_193b5d7f:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpunpckhqdq {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_193b5d7f:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpunpckhqdq {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32><i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_1z3z5z7z(<8 x i64> %a, <8 x i64> %b) {
;
; AVX512F-LABEL: shuffle_v8i64_1z3z5z7z:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; AVX512F-NEXT:    vpunpckhqdq {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: shuffle_v8i64_1z3z5z7z:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; AVX512F-32-NEXT:    vpunpckhqdq {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; AVX512F-32-NEXT:    retl
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> zeroinitializer, <8 x i32><i32 1, i32 8, i32 3, i32 15, i32 5, i32 8, i32 7, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x double> @test_vshuff64x2_512(<8 x double> %x, <8 x double> %x1) nounwind {
; AVX512F-LABEL: test_vshuff64x2_512:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],zmm1[2,3,0,1]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: test_vshuff64x2_512:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],zmm1[2,3,0,1]
; AVX512F-32-NEXT:    retl
  %res = shufflevector <8 x double> %x, <8 x double> %x1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 10, i32 11, i32 8, i32 9>
  ret <8 x double> %res
}

define <8 x double> @test_vshuff64x2_512_maskz(<8 x double> %x, <8 x double> %x1, <8 x i1> %mask) nounwind {
; AVX512F-LABEL: test_vshuff64x2_512_maskz:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpmovsxwq %xmm2, %zmm2
; AVX512F-NEXT:    vpandq {{.*}}(%rip){1to8}, %zmm2, %zmm2
; AVX512F-NEXT:    vptestmq %zmm2, %zmm2, %k1
; AVX512F-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],zmm1[2,3,0,1]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: test_vshuff64x2_512_maskz:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpmovsxwq %xmm2, %zmm2
; AVX512F-32-NEXT:    vpandq .LCPI122_0, %zmm2, %zmm2
; AVX512F-32-NEXT:    vptestmq %zmm2, %zmm2, %k1
; AVX512F-32-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],zmm1[2,3,0,1]
; AVX512F-32-NEXT:    retl
  %y = shufflevector <8 x double> %x, <8 x double> %x1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 10, i32 11, i32 8, i32 9>
  %res = select <8 x i1> %mask, <8 x double> %y, <8 x double> zeroinitializer
  ret <8 x double> %res
}

define <8 x i64> @test_vshufi64x2_512_mask(<8 x i64> %x, <8 x i64> %x1, <8 x i1> %mask) nounwind {
; AVX512F-LABEL: test_vshufi64x2_512_mask:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpmovsxwq %xmm2, %zmm2
; AVX512F-NEXT:    vpandq {{.*}}(%rip){1to8}, %zmm2, %zmm2
; AVX512F-NEXT:    vptestmq %zmm2, %zmm2, %k1
; AVX512F-NEXT:    vshufi64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],zmm1[2,3,0,1]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: test_vshufi64x2_512_mask:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpmovsxwq %xmm2, %zmm2
; AVX512F-32-NEXT:    vpandq .LCPI123_0, %zmm2, %zmm2
; AVX512F-32-NEXT:    vptestmq %zmm2, %zmm2, %k1
; AVX512F-32-NEXT:    vshufi64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],zmm1[2,3,0,1]
; AVX512F-32-NEXT:    retl
  %y = shufflevector <8 x i64> %x, <8 x i64> %x1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 10, i32 11, i32 8, i32 9>
  %res = select <8 x i1> %mask, <8 x i64> %y, <8 x i64> %x
  ret <8 x i64> %res
}

define <8 x double> @test_vshuff64x2_512_mem(<8 x double> %x, <8 x double> *%ptr) nounwind {
; AVX512F-LABEL: test_vshuff64x2_512_mem:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],mem[2,3,0,1]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: test_vshuff64x2_512_mem:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; AVX512F-32-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],mem[2,3,0,1]
; AVX512F-32-NEXT:    retl
  %x1   = load <8 x double>,<8 x double> *%ptr,align 1
  %res = shufflevector <8 x double> %x, <8 x double> %x1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 10, i32 11, i32 8, i32 9>
  ret <8 x double> %res
}

define <8 x double> @test_vshuff64x2_512_mem_mask(<8 x double> %x, <8 x double> *%ptr, <8 x i1> %mask) nounwind {
; AVX512F-LABEL: test_vshuff64x2_512_mem_mask:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpmovsxwq %xmm1, %zmm1
; AVX512F-NEXT:    vpandq {{.*}}(%rip){1to8}, %zmm1, %zmm1
; AVX512F-NEXT:    vptestmq %zmm1, %zmm1, %k1
; AVX512F-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],mem[2,3,0,1]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: test_vshuff64x2_512_mem_mask:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpmovsxwq %xmm1, %zmm1
; AVX512F-32-NEXT:    vpandq .LCPI125_0, %zmm1, %zmm1
; AVX512F-32-NEXT:    vptestmq %zmm1, %zmm1, %k1
; AVX512F-32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; AVX512F-32-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],mem[2,3,0,1]
; AVX512F-32-NEXT:    retl
  %x1 = load <8 x double>,<8 x double> *%ptr,align 1
  %y = shufflevector <8 x double> %x, <8 x double> %x1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 10, i32 11, i32 8, i32 9>
  %res = select <8 x i1> %mask, <8 x double> %y, <8 x double> %x
  ret <8 x double> %res
}

define <8 x double> @test_vshuff64x2_512_mem_maskz(<8 x double> %x, <8 x double> *%ptr, <8 x i1> %mask) nounwind {
; AVX512F-LABEL: test_vshuff64x2_512_mem_maskz:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vpmovsxwq %xmm1, %zmm1
; AVX512F-NEXT:    vpandq {{.*}}(%rip){1to8}, %zmm1, %zmm1
; AVX512F-NEXT:    vptestmq %zmm1, %zmm1, %k1
; AVX512F-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],mem[2,3,0,1]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: test_vshuff64x2_512_mem_maskz:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vpmovsxwq %xmm1, %zmm1
; AVX512F-32-NEXT:    vpandq .LCPI126_0, %zmm1, %zmm1
; AVX512F-32-NEXT:    vptestmq %zmm1, %zmm1, %k1
; AVX512F-32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; AVX512F-32-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,4,5],mem[2,3,0,1]
; AVX512F-32-NEXT:    retl
  %x1 = load <8 x double>,<8 x double> *%ptr,align 1
  %y = shufflevector <8 x double> %x, <8 x double> %x1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 10, i32 11, i32 8, i32 9>
  %res = select <8 x i1> %mask, <8 x double> %y, <8 x double> zeroinitializer
  ret <8 x double> %res
}

define <16 x float> @test_vshuff32x4_512(<16 x float> %x, <16 x float> %x1) nounwind {
; AVX512F-LABEL: test_vshuff32x4_512:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,2,3],zmm1[2,3,0,1]
; AVX512F-NEXT:    retq
;
; AVX512F-32-LABEL: test_vshuff32x4_512:
; AVX512F-32:       # BB#0:
; AVX512F-32-NEXT:    vshuff64x2 {{.*#+}} zmm0 = zmm0[0,1,2,3],zmm1[2,3,0,1]
; AVX512F-32-NEXT:    retl
  %res = shufflevector <16 x float> %x, <16 x float> %x1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 20, i32 21, i32 22, i32 23, i32 16, i32 17, i32 18, i32 19>
  ret <16 x float> %res
}
