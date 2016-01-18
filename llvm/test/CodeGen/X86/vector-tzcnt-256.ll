; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2

define <4 x i64> @testv4i64(<4 x i64> %in) nounwind {
; AVX1-LABEL: testv4i64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpsubq %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpsubq %xmm0, %xmm2, %xmm3
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm3, %ymm1
; AVX1-NEXT:    vandps %ymm1, %ymm0, %ymm0
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [1,1]
; AVX1-NEXT:    vpsubq %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm4 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vpand %xmm4, %xmm1, %xmm5
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm6 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm5, %xmm6, %xmm5
; AVX1-NEXT:    vpsrlw $4, %xmm1, %xmm1
; AVX1-NEXT:    vpand %xmm4, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm6, %xmm1
; AVX1-NEXT:    vpaddb %xmm5, %xmm1, %xmm1
; AVX1-NEXT:    vpsadbw %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpsubq %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm4, %xmm0, %xmm3
; AVX1-NEXT:    vpshufb %xmm3, %xmm6, %xmm3
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm4, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm6, %xmm0
; AVX1-NEXT:    vpaddb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpsadbw %xmm0, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv4i64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %ymm1, %ymm1, %ymm1
; AVX2-NEXT:    vpsubq %ymm0, %ymm1, %ymm2
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpbroadcastq {{.*}}(%rip), %ymm2
; AVX2-NEXT:    vpsubq %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm3
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm4 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm3, %ymm4, %ymm3
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm4, %ymm0
; AVX2-NEXT:    vpaddb %ymm3, %ymm0, %ymm0
; AVX2-NEXT:    vpsadbw %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %out = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> %in, i1 0)
  ret <4 x i64> %out
}

define <4 x i64> @testv4i64u(<4 x i64> %in) nounwind {
; AVX1-LABEL: testv4i64u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpsubq %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpsubq %xmm0, %xmm2, %xmm3
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm3, %ymm1
; AVX1-NEXT:    vandps %ymm1, %ymm0, %ymm0
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [1,1]
; AVX1-NEXT:    vpsubq %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm4 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vpand %xmm4, %xmm1, %xmm5
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm6 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm5, %xmm6, %xmm5
; AVX1-NEXT:    vpsrlw $4, %xmm1, %xmm1
; AVX1-NEXT:    vpand %xmm4, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm6, %xmm1
; AVX1-NEXT:    vpaddb %xmm5, %xmm1, %xmm1
; AVX1-NEXT:    vpsadbw %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpsubq %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm4, %xmm0, %xmm3
; AVX1-NEXT:    vpshufb %xmm3, %xmm6, %xmm3
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm4, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm6, %xmm0
; AVX1-NEXT:    vpaddb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpsadbw %xmm0, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv4i64u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %ymm1, %ymm1, %ymm1
; AVX2-NEXT:    vpsubq %ymm0, %ymm1, %ymm2
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpbroadcastq {{.*}}(%rip), %ymm2
; AVX2-NEXT:    vpsubq %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm3
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm4 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm3, %ymm4, %ymm3
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm4, %ymm0
; AVX2-NEXT:    vpaddb %ymm3, %ymm0, %ymm0
; AVX2-NEXT:    vpsadbw %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %out = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> %in, i1 -1)
  ret <4 x i64> %out
}

define <8 x i32> @testv8i32(<8 x i32> %in) nounwind {
; AVX1-LABEL: testv8i32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpsubd %xmm2, %xmm1, %xmm2
; AVX1-NEXT:    vpsubd %xmm0, %xmm1, %xmm3
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm3, %ymm2
; AVX1-NEXT:    vandps %ymm2, %ymm0, %ymm0
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [1,1,1,1]
; AVX1-NEXT:    vpsubd %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm4 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vpand %xmm4, %xmm2, %xmm5
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm6 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm5, %xmm6, %xmm5
; AVX1-NEXT:    vpsrlw $4, %xmm2, %xmm2
; AVX1-NEXT:    vpand %xmm4, %xmm2, %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm6, %xmm2
; AVX1-NEXT:    vpaddb %xmm5, %xmm2, %xmm2
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm5 = xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; AVX1-NEXT:    vpsadbw %xmm5, %xmm1, %xmm5
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; AVX1-NEXT:    vpsadbw %xmm2, %xmm1, %xmm2
; AVX1-NEXT:    vpackuswb %xmm5, %xmm2, %xmm2
; AVX1-NEXT:    vpsubd %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm4, %xmm0, %xmm3
; AVX1-NEXT:    vpshufb %xmm3, %xmm6, %xmm3
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm4, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm6, %xmm0
; AVX1-NEXT:    vpaddb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm3 = xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpsadbw %xmm3, %xmm1, %xmm3
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; AVX1-NEXT:    vpsadbw %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    vpackuswb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv8i32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %ymm1, %ymm1, %ymm1
; AVX2-NEXT:    vpsubd %ymm0, %ymm1, %ymm2
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm2
; AVX2-NEXT:    vpsubd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm3
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm4 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm3, %ymm4, %ymm3
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm4, %ymm0
; AVX2-NEXT:    vpaddb %ymm3, %ymm0, %ymm0
; AVX2-NEXT:    vpunpckhdq {{.*#+}} ymm2 = ymm0[2],ymm1[2],ymm0[3],ymm1[3],ymm0[6],ymm1[6],ymm0[7],ymm1[7]
; AVX2-NEXT:    vpsadbw %ymm2, %ymm1, %ymm2
; AVX2-NEXT:    vpunpckldq {{.*#+}} ymm0 = ymm0[0],ymm1[0],ymm0[1],ymm1[1],ymm0[4],ymm1[4],ymm0[5],ymm1[5]
; AVX2-NEXT:    vpsadbw %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    vpackuswb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> %in, i1 0)
  ret <8 x i32> %out
}

define <8 x i32> @testv8i32u(<8 x i32> %in) nounwind {
; AVX1-LABEL: testv8i32u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpsubd %xmm2, %xmm1, %xmm2
; AVX1-NEXT:    vpsubd %xmm0, %xmm1, %xmm3
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm3, %ymm2
; AVX1-NEXT:    vandps %ymm2, %ymm0, %ymm0
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [1,1,1,1]
; AVX1-NEXT:    vpsubd %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm4 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vpand %xmm4, %xmm2, %xmm5
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm6 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm5, %xmm6, %xmm5
; AVX1-NEXT:    vpsrlw $4, %xmm2, %xmm2
; AVX1-NEXT:    vpand %xmm4, %xmm2, %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm6, %xmm2
; AVX1-NEXT:    vpaddb %xmm5, %xmm2, %xmm2
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm5 = xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; AVX1-NEXT:    vpsadbw %xmm5, %xmm1, %xmm5
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; AVX1-NEXT:    vpsadbw %xmm2, %xmm1, %xmm2
; AVX1-NEXT:    vpackuswb %xmm5, %xmm2, %xmm2
; AVX1-NEXT:    vpsubd %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm4, %xmm0, %xmm3
; AVX1-NEXT:    vpshufb %xmm3, %xmm6, %xmm3
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm4, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm6, %xmm0
; AVX1-NEXT:    vpaddb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm3 = xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpsadbw %xmm3, %xmm1, %xmm3
; AVX1-NEXT:    vpunpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; AVX1-NEXT:    vpsadbw %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    vpackuswb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv8i32u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %ymm1, %ymm1, %ymm1
; AVX2-NEXT:    vpsubd %ymm0, %ymm1, %ymm2
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm2
; AVX2-NEXT:    vpsubd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm3
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm4 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm3, %ymm4, %ymm3
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm4, %ymm0
; AVX2-NEXT:    vpaddb %ymm3, %ymm0, %ymm0
; AVX2-NEXT:    vpunpckhdq {{.*#+}} ymm2 = ymm0[2],ymm1[2],ymm0[3],ymm1[3],ymm0[6],ymm1[6],ymm0[7],ymm1[7]
; AVX2-NEXT:    vpsadbw %ymm2, %ymm1, %ymm2
; AVX2-NEXT:    vpunpckldq {{.*#+}} ymm0 = ymm0[0],ymm1[0],ymm0[1],ymm1[1],ymm0[4],ymm1[4],ymm0[5],ymm1[5]
; AVX2-NEXT:    vpsadbw %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    vpackuswb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> %in, i1 -1)
  ret <8 x i32> %out
}

define <16 x i16> @testv16i16(<16 x i16> %in) nounwind {
; AVX1-LABEL: testv16i16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpsubw %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpsubw %xmm0, %xmm2, %xmm2
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; AVX1-NEXT:    vandps %ymm1, %ymm0, %ymm0
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm1 = [1,1,1,1,1,1,1,1]
; AVX1-NEXT:    vpsubw %xmm1, %xmm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vpand %xmm3, %xmm2, %xmm4
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm5 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm4, %xmm5, %xmm4
; AVX1-NEXT:    vpsrlw $4, %xmm2, %xmm2
; AVX1-NEXT:    vpand %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm5, %xmm2
; AVX1-NEXT:    vpaddb %xmm4, %xmm2, %xmm2
; AVX1-NEXT:    vpsllw $8, %xmm2, %xmm4
; AVX1-NEXT:    vpaddb %xmm2, %xmm4, %xmm2
; AVX1-NEXT:    vpsrlw $8, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpsubw %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm3, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm5, %xmm1
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm5, %xmm0
; AVX1-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpsllw $8, %xmm0, %xmm1
; AVX1-NEXT:    vpaddb %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    vpsrlw $8, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv16i16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %ymm1, %ymm1, %ymm1
; AVX2-NEXT:    vpsubw %ymm0, %ymm1, %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpsubw {{.*}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm1 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm2
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm3 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm2, %ymm3, %ymm2
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm3, %ymm0
; AVX2-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpsllw $8, %ymm0, %ymm1
; AVX2-NEXT:    vpaddb %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    vpsrlw $8, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> %in, i1 0)
  ret <16 x i16> %out
}

define <16 x i16> @testv16i16u(<16 x i16> %in) nounwind {
; AVX1-LABEL: testv16i16u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpsubw %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpsubw %xmm0, %xmm2, %xmm2
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; AVX1-NEXT:    vandps %ymm1, %ymm0, %ymm0
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm1 = [1,1,1,1,1,1,1,1]
; AVX1-NEXT:    vpsubw %xmm1, %xmm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vpand %xmm3, %xmm2, %xmm4
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm5 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm4, %xmm5, %xmm4
; AVX1-NEXT:    vpsrlw $4, %xmm2, %xmm2
; AVX1-NEXT:    vpand %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm5, %xmm2
; AVX1-NEXT:    vpaddb %xmm4, %xmm2, %xmm2
; AVX1-NEXT:    vpsllw $8, %xmm2, %xmm4
; AVX1-NEXT:    vpaddb %xmm2, %xmm4, %xmm2
; AVX1-NEXT:    vpsrlw $8, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpsubw %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm3, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm5, %xmm1
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm5, %xmm0
; AVX1-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpsllw $8, %xmm0, %xmm1
; AVX1-NEXT:    vpaddb %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    vpsrlw $8, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv16i16u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %ymm1, %ymm1, %ymm1
; AVX2-NEXT:    vpsubw %ymm0, %ymm1, %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpsubw {{.*}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm1 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm2
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm3 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm2, %ymm3, %ymm2
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm3, %ymm0
; AVX2-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpsllw $8, %ymm0, %ymm1
; AVX2-NEXT:    vpaddb %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    vpsrlw $8, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> %in, i1 -1)
  ret <16 x i16> %out
}

define <32 x i8> @testv32i8(<32 x i8> %in) nounwind {
; AVX1-LABEL: testv32i8:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpsubb %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpsubb %xmm0, %xmm2, %xmm2
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; AVX1-NEXT:    vandps %ymm1, %ymm0, %ymm0
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
; AVX1-NEXT:    vpsubb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vpand %xmm3, %xmm1, %xmm4
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm5 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm4, %xmm5, %xmm4
; AVX1-NEXT:    vpsrlw $4, %xmm1, %xmm1
; AVX1-NEXT:    vpand %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm5, %xmm1
; AVX1-NEXT:    vpaddb %xmm4, %xmm1, %xmm1
; AVX1-NEXT:    vpsubb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm3, %xmm0, %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm5, %xmm2
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm5, %xmm0
; AVX1-NEXT:    vpaddb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv32i8:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %ymm1, %ymm1, %ymm1
; AVX2-NEXT:    vpsubb %ymm0, %ymm1, %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpsubb {{.*}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm1 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm2
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm3 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm2, %ymm3, %ymm2
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm3, %ymm0
; AVX2-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> %in, i1 0)
  ret <32 x i8> %out
}

define <32 x i8> @testv32i8u(<32 x i8> %in) nounwind {
; AVX1-LABEL: testv32i8u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpsubb %xmm1, %xmm2, %xmm1
; AVX1-NEXT:    vpsubb %xmm0, %xmm2, %xmm2
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; AVX1-NEXT:    vandps %ymm1, %ymm0, %ymm0
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
; AVX1-NEXT:    vpsubb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vpand %xmm3, %xmm1, %xmm4
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm5 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX1-NEXT:    vpshufb %xmm4, %xmm5, %xmm4
; AVX1-NEXT:    vpsrlw $4, %xmm1, %xmm1
; AVX1-NEXT:    vpand %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm5, %xmm1
; AVX1-NEXT:    vpaddb %xmm4, %xmm1, %xmm1
; AVX1-NEXT:    vpsubb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm3, %xmm0, %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm5, %xmm2
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm0, %xmm5, %xmm0
; AVX1-NEXT:    vpaddb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv32i8u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %ymm1, %ymm1, %ymm1
; AVX2-NEXT:    vpsubb %ymm0, %ymm1, %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpsubb {{.*}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm1 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm2
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm3 = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]
; AVX2-NEXT:    vpshufb %ymm2, %ymm3, %ymm2
; AVX2-NEXT:    vpsrlw $4, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm0, %ymm3, %ymm0
; AVX2-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> %in, i1 -1)
  ret <32 x i8> %out
}

define <4 x i64> @foldv4i64() nounwind {
; ALL-LABEL: foldv4i64:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,64,0]
; ALL-NEXT:    retq
  %out = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> <i64 256, i64 -1, i64 0, i64 255>, i1 0)
  ret <4 x i64> %out
}

define <4 x i64> @foldv4i64u() nounwind {
; ALL-LABEL: foldv4i64u:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,64,0]
; ALL-NEXT:    retq
  %out = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> <i64 256, i64 -1, i64 0, i64 255>, i1 -1)
  ret <4 x i64> %out
}

define <8 x i32> @foldv8i32() nounwind {
; ALL-LABEL: foldv8i32:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,32,0,16,0,3,3]
; ALL-NEXT:    retq
  %out = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> <i32 256, i32 -1, i32 0, i32 255, i32 -65536, i32 7, i32 24, i32 88>, i1 0)
  ret <8 x i32> %out
}

define <8 x i32> @foldv8i32u() nounwind {
; ALL-LABEL: foldv8i32u:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,32,0,16,0,3,3]
; ALL-NEXT:    retq
  %out = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> <i32 256, i32 -1, i32 0, i32 255, i32 -65536, i32 7, i32 24, i32 88>, i1 -1)
  ret <8 x i32> %out
}

define <16 x i16> @foldv16i16() nounwind {
; ALL-LABEL: foldv16i16:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,16,0,16,0,3,3,1,1,0,1,2,3,4,5]
; ALL-NEXT:    retq
  %out = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> <i16 256, i16 -1, i16 0, i16 255, i16 -65536, i16 7, i16 24, i16 88, i16 -2, i16 254, i16 1, i16 2, i16 4, i16 8, i16 16, i16 32>, i1 0)
  ret <16 x i16> %out
}

define <16 x i16> @foldv16i16u() nounwind {
; ALL-LABEL: foldv16i16u:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,16,0,16,0,3,3,1,1,0,1,2,3,4,5]
; ALL-NEXT:    retq
  %out = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> <i16 256, i16 -1, i16 0, i16 255, i16 -65536, i16 7, i16 24, i16 88, i16 -2, i16 254, i16 1, i16 2, i16 4, i16 8, i16 16, i16 32>, i1 -1)
  ret <16 x i16> %out
}

define <32 x i8> @foldv32i8() nounwind {
; ALL-LABEL: foldv32i8:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,8,0,8,0,3,3,1,1,0,1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1,0,0,0,0,0]
; ALL-NEXT:    retq
  %out = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> <i8 256, i8 -1, i8 0, i8 255, i8 -65536, i8 7, i8 24, i8 88, i8 -2, i8 254, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 128, i8 256, i8 -256, i8 -128, i8 -64, i8 -32, i8 -16, i8 -8, i8 -4, i8 -2, i8 -1, i8 3, i8 5, i8 7, i8 127>, i1 0)
  ret <32 x i8> %out
}

define <32 x i8> @foldv32i8u() nounwind {
; ALL-LABEL: foldv32i8u:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,8,0,8,0,3,3,1,1,0,1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1,0,0,0,0,0]
; ALL-NEXT:    retq
  %out = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> <i8 256, i8 -1, i8 0, i8 255, i8 -65536, i8 7, i8 24, i8 88, i8 -2, i8 254, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 128, i8 256, i8 -256, i8 -128, i8 -64, i8 -32, i8 -16, i8 -8, i8 -4, i8 -2, i8 -1, i8 3, i8 5, i8 7, i8 127>, i1 -1)
  ret <32 x i8> %out
}

declare <4 x i64> @llvm.cttz.v4i64(<4 x i64>, i1)
declare <8 x i32> @llvm.cttz.v8i32(<8 x i32>, i1)
declare <16 x i16> @llvm.cttz.v16i16(<16 x i16>, i1)
declare <32 x i8> @llvm.cttz.v32i8(<32 x i8>, i1)
