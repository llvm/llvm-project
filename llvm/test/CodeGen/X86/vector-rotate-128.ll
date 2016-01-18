; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+xop,+avx | FileCheck %s --check-prefix=ALL --check-prefix=XOP --check-prefix=XOPAVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+xop,+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=XOP --check-prefix=XOPAVX2
;
; Just one 32-bit run to make sure we do reasonable things for i64 rotates.
; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefix=ALL --check-prefix=X32-SSE --check-prefix=X32-SSE2

;
; Variable Rotates
;

define <2 x i64> @var_rotate_v2i64(<2 x i64> %a, <2 x i64> %b) nounwind {
; SSE2-LABEL: var_rotate_v2i64:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa {{.*#+}} xmm2 = [64,64]
; SSE2-NEXT:    psubq %xmm1, %xmm2
; SSE2-NEXT:    pshufd {{.*#+}} xmm3 = xmm1[2,3,0,1]
; SSE2-NEXT:    movdqa %xmm0, %xmm4
; SSE2-NEXT:    psllq %xmm3, %xmm4
; SSE2-NEXT:    movdqa %xmm0, %xmm3
; SSE2-NEXT:    psllq %xmm1, %xmm3
; SSE2-NEXT:    movsd {{.*#+}} xmm4 = xmm3[0],xmm4[1]
; SSE2-NEXT:    pshufd {{.*#+}} xmm3 = xmm2[2,3,0,1]
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    psrlq %xmm3, %xmm1
; SSE2-NEXT:    psrlq %xmm2, %xmm0
; SSE2-NEXT:    movsd {{.*#+}} xmm1 = xmm0[0],xmm1[1]
; SSE2-NEXT:    orpd %xmm4, %xmm1
; SSE2-NEXT:    movapd %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: var_rotate_v2i64:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm2 = [64,64]
; SSE41-NEXT:    psubq %xmm1, %xmm2
; SSE41-NEXT:    movdqa %xmm0, %xmm3
; SSE41-NEXT:    psllq %xmm1, %xmm3
; SSE41-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; SSE41-NEXT:    movdqa %xmm0, %xmm4
; SSE41-NEXT:    psllq %xmm1, %xmm4
; SSE41-NEXT:    pblendw {{.*#+}} xmm4 = xmm3[0,1,2,3],xmm4[4,5,6,7]
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    psrlq %xmm2, %xmm1
; SSE41-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[2,3,0,1]
; SSE41-NEXT:    psrlq %xmm2, %xmm0
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; SSE41-NEXT:    por %xmm4, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: var_rotate_v2i64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [64,64]
; AVX1-NEXT:    vpsubq %xmm1, %xmm2, %xmm2
; AVX1-NEXT:    vpsllq %xmm1, %xmm0, %xmm3
; AVX1-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[2,3,0,1]
; AVX1-NEXT:    vpsllq %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm3[0,1,2,3],xmm1[4,5,6,7]
; AVX1-NEXT:    vpsrlq %xmm2, %xmm0, %xmm3
; AVX1-NEXT:    vpshufd {{.*#+}} xmm2 = xmm2[2,3,0,1]
; AVX1-NEXT:    vpsrlq %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm3[0,1,2,3],xmm0[4,5,6,7]
; AVX1-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: var_rotate_v2i64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [64,64]
; AVX2-NEXT:    vpsubq %xmm1, %xmm2, %xmm2
; AVX2-NEXT:    vpsllvq %xmm1, %xmm0, %xmm1
; AVX2-NEXT:    vpsrlvq %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    retq
;
; XOP-LABEL: var_rotate_v2i64:
; XOP:       # BB#0:
; XOP-NEXT:    vprotq %xmm1, %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: var_rotate_v2i64:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm2 = [64,0,64,0]
; X32-SSE-NEXT:    psubq %xmm1, %xmm2
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm1[2,3,0,1]
; X32-SSE-NEXT:    movdqa %xmm0, %xmm4
; X32-SSE-NEXT:    psllq %xmm3, %xmm4
; X32-SSE-NEXT:    movq {{.*#+}} xmm1 = xmm1[0],zero
; X32-SSE-NEXT:    movdqa %xmm0, %xmm3
; X32-SSE-NEXT:    psllq %xmm1, %xmm3
; X32-SSE-NEXT:    movsd {{.*#+}} xmm4 = xmm3[0],xmm4[1]
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm2[2,3,0,1]
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    psrlq %xmm3, %xmm1
; X32-SSE-NEXT:    movq {{.*#+}} xmm2 = xmm2[0],zero
; X32-SSE-NEXT:    psrlq %xmm2, %xmm0
; X32-SSE-NEXT:    movsd {{.*#+}} xmm1 = xmm0[0],xmm1[1]
; X32-SSE-NEXT:    orpd %xmm4, %xmm1
; X32-SSE-NEXT:    movapd %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %b64 = sub <2 x i64> <i64 64, i64 64>, %b
  %shl = shl <2 x i64> %a, %b
  %lshr = lshr <2 x i64> %a, %b64
  %or = or <2 x i64> %shl, %lshr
  ret <2 x i64> %or
}

define <4 x i32> @var_rotate_v4i32(<4 x i32> %a, <4 x i32> %b) nounwind {
; SSE2-LABEL: var_rotate_v4i32:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa {{.*#+}} xmm2 = [32,32,32,32]
; SSE2-NEXT:    psubd %xmm1, %xmm2
; SSE2-NEXT:    pslld $23, %xmm1
; SSE2-NEXT:    paddd {{.*}}(%rip), %xmm1
; SSE2-NEXT:    cvttps2dq %xmm1, %xmm1
; SSE2-NEXT:    pshufd {{.*#+}} xmm3 = xmm1[1,1,3,3]
; SSE2-NEXT:    pmuludq %xmm0, %xmm1
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[0,2,2,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[1,1,3,3]
; SSE2-NEXT:    pmuludq %xmm3, %xmm4
; SSE2-NEXT:    pshufd {{.*#+}} xmm3 = xmm4[0,2,2,3]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1]
; SSE2-NEXT:    movdqa %xmm2, %xmm3
; SSE2-NEXT:    psrldq {{.*#+}} xmm3 = xmm3[12,13,14,15],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; SSE2-NEXT:    movdqa %xmm0, %xmm4
; SSE2-NEXT:    psrld %xmm3, %xmm4
; SSE2-NEXT:    movdqa %xmm2, %xmm3
; SSE2-NEXT:    psrlq $32, %xmm3
; SSE2-NEXT:    movdqa %xmm0, %xmm5
; SSE2-NEXT:    psrld %xmm3, %xmm5
; SSE2-NEXT:    movsd {{.*#+}} xmm4 = xmm5[0],xmm4[1]
; SSE2-NEXT:    pshufd {{.*#+}} xmm3 = xmm4[1,3,2,3]
; SSE2-NEXT:    pxor %xmm4, %xmm4
; SSE2-NEXT:    movdqa %xmm2, %xmm5
; SSE2-NEXT:    punpckhdq {{.*#+}} xmm5 = xmm5[2],xmm4[2],xmm5[3],xmm4[3]
; SSE2-NEXT:    movdqa %xmm0, %xmm6
; SSE2-NEXT:    psrld %xmm5, %xmm6
; SSE2-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1]
; SSE2-NEXT:    psrld %xmm2, %xmm0
; SSE2-NEXT:    movsd {{.*#+}} xmm6 = xmm0[0],xmm6[1]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm6[0,2,2,3]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1]
; SSE2-NEXT:    por %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: var_rotate_v4i32:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm2 = [32,32,32,32]
; SSE41-NEXT:    psubd %xmm1, %xmm2
; SSE41-NEXT:    pslld $23, %xmm1
; SSE41-NEXT:    paddd {{.*}}(%rip), %xmm1
; SSE41-NEXT:    cvttps2dq %xmm1, %xmm1
; SSE41-NEXT:    pmulld %xmm0, %xmm1
; SSE41-NEXT:    movdqa %xmm2, %xmm3
; SSE41-NEXT:    psrldq {{.*#+}} xmm3 = xmm3[12,13,14,15],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; SSE41-NEXT:    movdqa %xmm0, %xmm4
; SSE41-NEXT:    psrld %xmm3, %xmm4
; SSE41-NEXT:    movdqa %xmm2, %xmm3
; SSE41-NEXT:    psrlq $32, %xmm3
; SSE41-NEXT:    movdqa %xmm0, %xmm5
; SSE41-NEXT:    psrld %xmm3, %xmm5
; SSE41-NEXT:    pblendw {{.*#+}} xmm5 = xmm5[0,1,2,3],xmm4[4,5,6,7]
; SSE41-NEXT:    pxor %xmm3, %xmm3
; SSE41-NEXT:    pmovzxdq {{.*#+}} xmm4 = xmm2[0],zero,xmm2[1],zero
; SSE41-NEXT:    punpckhdq {{.*#+}} xmm2 = xmm2[2],xmm3[2],xmm2[3],xmm3[3]
; SSE41-NEXT:    movdqa %xmm0, %xmm3
; SSE41-NEXT:    psrld %xmm2, %xmm3
; SSE41-NEXT:    psrld %xmm4, %xmm0
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm3[4,5,6,7]
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm5[2,3],xmm0[4,5],xmm5[6,7]
; SSE41-NEXT:    por %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: var_rotate_v4i32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [32,32,32,32]
; AVX1-NEXT:    vpsubd %xmm1, %xmm2, %xmm2
; AVX1-NEXT:    vpslld $23, %xmm1, %xmm1
; AVX1-NEXT:    vpaddd {{.*}}(%rip), %xmm1, %xmm1
; AVX1-NEXT:    vcvttps2dq %xmm1, %xmm1
; AVX1-NEXT:    vpmulld %xmm0, %xmm1, %xmm1
; AVX1-NEXT:    vpsrldq {{.*#+}} xmm3 = xmm2[12,13,14,15],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpsrld %xmm3, %xmm0, %xmm3
; AVX1-NEXT:    vpsrlq $32, %xmm2, %xmm4
; AVX1-NEXT:    vpsrld %xmm4, %xmm0, %xmm4
; AVX1-NEXT:    vpblendw {{.*#+}} xmm3 = xmm4[0,1,2,3],xmm3[4,5,6,7]
; AVX1-NEXT:    vpxor %xmm4, %xmm4, %xmm4
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm4 = xmm2[2],xmm4[2],xmm2[3],xmm4[3]
; AVX1-NEXT:    vpsrld %xmm4, %xmm0, %xmm4
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm2 = xmm2[0],zero,xmm2[1],zero
; AVX1-NEXT:    vpsrld %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm4[4,5,6,7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm3[2,3],xmm0[4,5],xmm3[6,7]
; AVX1-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: var_rotate_v4i32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm2
; AVX2-NEXT:    vpsubd %xmm1, %xmm2, %xmm2
; AVX2-NEXT:    vpsllvd %xmm1, %xmm0, %xmm1
; AVX2-NEXT:    vpsrlvd %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    retq
;
; XOP-LABEL: var_rotate_v4i32:
; XOP:       # BB#0:
; XOP-NEXT:    vprotd %xmm1, %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: var_rotate_v4i32:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm2 = [32,32,32,32]
; X32-SSE-NEXT:    psubd %xmm1, %xmm2
; X32-SSE-NEXT:    pslld $23, %xmm1
; X32-SSE-NEXT:    paddd .LCPI1_1, %xmm1
; X32-SSE-NEXT:    cvttps2dq %xmm1, %xmm1
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm1[1,1,3,3]
; X32-SSE-NEXT:    pmuludq %xmm0, %xmm1
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[0,2,2,3]
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[1,1,3,3]
; X32-SSE-NEXT:    pmuludq %xmm3, %xmm4
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm4[0,2,2,3]
; X32-SSE-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1]
; X32-SSE-NEXT:    movdqa %xmm2, %xmm3
; X32-SSE-NEXT:    psrldq {{.*#+}} xmm3 = xmm3[12,13,14,15],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; X32-SSE-NEXT:    movdqa %xmm0, %xmm4
; X32-SSE-NEXT:    psrld %xmm3, %xmm4
; X32-SSE-NEXT:    movdqa %xmm2, %xmm3
; X32-SSE-NEXT:    psrlq $32, %xmm3
; X32-SSE-NEXT:    movdqa %xmm0, %xmm5
; X32-SSE-NEXT:    psrld %xmm3, %xmm5
; X32-SSE-NEXT:    movsd {{.*#+}} xmm4 = xmm5[0],xmm4[1]
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm4[1,3,2,3]
; X32-SSE-NEXT:    pxor %xmm4, %xmm4
; X32-SSE-NEXT:    movdqa %xmm2, %xmm5
; X32-SSE-NEXT:    punpckhdq {{.*#+}} xmm5 = xmm5[2],xmm4[2],xmm5[3],xmm4[3]
; X32-SSE-NEXT:    movdqa %xmm0, %xmm6
; X32-SSE-NEXT:    psrld %xmm5, %xmm6
; X32-SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1]
; X32-SSE-NEXT:    psrld %xmm2, %xmm0
; X32-SSE-NEXT:    movsd {{.*#+}} xmm6 = xmm0[0],xmm6[1]
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm6[0,2,2,3]
; X32-SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1]
; X32-SSE-NEXT:    por %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %b32 = sub <4 x i32> <i32 32, i32 32, i32 32, i32 32>, %b
  %shl = shl <4 x i32> %a, %b
  %lshr = lshr <4 x i32> %a, %b32
  %or = or <4 x i32> %shl, %lshr
  ret <4 x i32> %or
}

define <8 x i16> @var_rotate_v8i16(<8 x i16> %a, <8 x i16> %b) nounwind {
; SSE2-LABEL: var_rotate_v8i16:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa {{.*#+}} xmm3 = [16,16,16,16,16,16,16,16]
; SSE2-NEXT:    psubw %xmm1, %xmm3
; SSE2-NEXT:    psllw $12, %xmm1
; SSE2-NEXT:    movdqa %xmm1, %xmm2
; SSE2-NEXT:    psraw $15, %xmm2
; SSE2-NEXT:    movdqa %xmm0, %xmm4
; SSE2-NEXT:    psllw $8, %xmm4
; SSE2-NEXT:    pand %xmm2, %xmm4
; SSE2-NEXT:    pandn %xmm0, %xmm2
; SSE2-NEXT:    por %xmm4, %xmm2
; SSE2-NEXT:    paddw %xmm1, %xmm1
; SSE2-NEXT:    movdqa %xmm1, %xmm4
; SSE2-NEXT:    psraw $15, %xmm4
; SSE2-NEXT:    movdqa %xmm4, %xmm5
; SSE2-NEXT:    pandn %xmm2, %xmm5
; SSE2-NEXT:    psllw $4, %xmm2
; SSE2-NEXT:    pand %xmm4, %xmm2
; SSE2-NEXT:    por %xmm5, %xmm2
; SSE2-NEXT:    paddw %xmm1, %xmm1
; SSE2-NEXT:    movdqa %xmm1, %xmm4
; SSE2-NEXT:    psraw $15, %xmm4
; SSE2-NEXT:    movdqa %xmm4, %xmm5
; SSE2-NEXT:    pandn %xmm2, %xmm5
; SSE2-NEXT:    psllw $2, %xmm2
; SSE2-NEXT:    pand %xmm4, %xmm2
; SSE2-NEXT:    por %xmm5, %xmm2
; SSE2-NEXT:    paddw %xmm1, %xmm1
; SSE2-NEXT:    psraw $15, %xmm1
; SSE2-NEXT:    movdqa %xmm1, %xmm4
; SSE2-NEXT:    pandn %xmm2, %xmm4
; SSE2-NEXT:    psllw $1, %xmm2
; SSE2-NEXT:    pand %xmm1, %xmm2
; SSE2-NEXT:    psllw $12, %xmm3
; SSE2-NEXT:    movdqa %xmm3, %xmm1
; SSE2-NEXT:    psraw $15, %xmm1
; SSE2-NEXT:    movdqa %xmm1, %xmm5
; SSE2-NEXT:    pandn %xmm0, %xmm5
; SSE2-NEXT:    psrlw $8, %xmm0
; SSE2-NEXT:    pand %xmm1, %xmm0
; SSE2-NEXT:    por %xmm5, %xmm0
; SSE2-NEXT:    paddw %xmm3, %xmm3
; SSE2-NEXT:    movdqa %xmm3, %xmm1
; SSE2-NEXT:    psraw $15, %xmm1
; SSE2-NEXT:    movdqa %xmm1, %xmm5
; SSE2-NEXT:    pandn %xmm0, %xmm5
; SSE2-NEXT:    psrlw $4, %xmm0
; SSE2-NEXT:    pand %xmm1, %xmm0
; SSE2-NEXT:    por %xmm5, %xmm0
; SSE2-NEXT:    paddw %xmm3, %xmm3
; SSE2-NEXT:    movdqa %xmm3, %xmm1
; SSE2-NEXT:    psraw $15, %xmm1
; SSE2-NEXT:    movdqa %xmm1, %xmm5
; SSE2-NEXT:    pandn %xmm0, %xmm5
; SSE2-NEXT:    psrlw $2, %xmm0
; SSE2-NEXT:    pand %xmm1, %xmm0
; SSE2-NEXT:    por %xmm5, %xmm0
; SSE2-NEXT:    paddw %xmm3, %xmm3
; SSE2-NEXT:    psraw $15, %xmm3
; SSE2-NEXT:    movdqa %xmm3, %xmm1
; SSE2-NEXT:    pandn %xmm0, %xmm1
; SSE2-NEXT:    psrlw $1, %xmm0
; SSE2-NEXT:    pand %xmm3, %xmm0
; SSE2-NEXT:    por %xmm1, %xmm0
; SSE2-NEXT:    por %xmm4, %xmm0
; SSE2-NEXT:    por %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: var_rotate_v8i16:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa %xmm0, %xmm3
; SSE41-NEXT:    movdqa {{.*#+}} xmm2 = [16,16,16,16,16,16,16,16]
; SSE41-NEXT:    psubw %xmm1, %xmm2
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    psllw $12, %xmm0
; SSE41-NEXT:    psllw $4, %xmm1
; SSE41-NEXT:    por %xmm0, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm4
; SSE41-NEXT:    paddw %xmm4, %xmm4
; SSE41-NEXT:    movdqa %xmm3, %xmm6
; SSE41-NEXT:    psllw $8, %xmm6
; SSE41-NEXT:    movdqa %xmm3, %xmm5
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    pblendvb %xmm6, %xmm5
; SSE41-NEXT:    movdqa %xmm5, %xmm1
; SSE41-NEXT:    psllw $4, %xmm1
; SSE41-NEXT:    movdqa %xmm4, %xmm0
; SSE41-NEXT:    pblendvb %xmm1, %xmm5
; SSE41-NEXT:    movdqa %xmm5, %xmm1
; SSE41-NEXT:    psllw $2, %xmm1
; SSE41-NEXT:    paddw %xmm4, %xmm4
; SSE41-NEXT:    movdqa %xmm4, %xmm0
; SSE41-NEXT:    pblendvb %xmm1, %xmm5
; SSE41-NEXT:    movdqa %xmm5, %xmm1
; SSE41-NEXT:    psllw $1, %xmm1
; SSE41-NEXT:    paddw %xmm4, %xmm4
; SSE41-NEXT:    movdqa %xmm4, %xmm0
; SSE41-NEXT:    pblendvb %xmm1, %xmm5
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    psllw $12, %xmm0
; SSE41-NEXT:    psllw $4, %xmm2
; SSE41-NEXT:    por %xmm0, %xmm2
; SSE41-NEXT:    movdqa %xmm2, %xmm1
; SSE41-NEXT:    paddw %xmm1, %xmm1
; SSE41-NEXT:    movdqa %xmm3, %xmm4
; SSE41-NEXT:    psrlw $8, %xmm4
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    pblendvb %xmm4, %xmm3
; SSE41-NEXT:    movdqa %xmm3, %xmm2
; SSE41-NEXT:    psrlw $4, %xmm2
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    pblendvb %xmm2, %xmm3
; SSE41-NEXT:    movdqa %xmm3, %xmm2
; SSE41-NEXT:    psrlw $2, %xmm2
; SSE41-NEXT:    paddw %xmm1, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    pblendvb %xmm2, %xmm3
; SSE41-NEXT:    movdqa %xmm3, %xmm2
; SSE41-NEXT:    psrlw $1, %xmm2
; SSE41-NEXT:    paddw %xmm1, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    pblendvb %xmm2, %xmm3
; SSE41-NEXT:    por %xmm5, %xmm3
; SSE41-NEXT:    movdqa %xmm3, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: var_rotate_v8i16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [16,16,16,16,16,16,16,16]
; AVX1-NEXT:    vpsubw %xmm1, %xmm2, %xmm2
; AVX1-NEXT:    vpsllw $12, %xmm1, %xmm3
; AVX1-NEXT:    vpsllw $4, %xmm1, %xmm1
; AVX1-NEXT:    vpor %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpaddw %xmm1, %xmm1, %xmm3
; AVX1-NEXT:    vpsllw $8, %xmm0, %xmm4
; AVX1-NEXT:    vpblendvb %xmm1, %xmm4, %xmm0, %xmm1
; AVX1-NEXT:    vpsllw $4, %xmm1, %xmm4
; AVX1-NEXT:    vpblendvb %xmm3, %xmm4, %xmm1, %xmm1
; AVX1-NEXT:    vpsllw $2, %xmm1, %xmm4
; AVX1-NEXT:    vpaddw %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpblendvb %xmm3, %xmm4, %xmm1, %xmm1
; AVX1-NEXT:    vpsllw $1, %xmm1, %xmm4
; AVX1-NEXT:    vpaddw %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpblendvb %xmm3, %xmm4, %xmm1, %xmm1
; AVX1-NEXT:    vpsllw $12, %xmm2, %xmm3
; AVX1-NEXT:    vpsllw $4, %xmm2, %xmm2
; AVX1-NEXT:    vpor %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vpaddw %xmm2, %xmm2, %xmm3
; AVX1-NEXT:    vpsrlw $8, %xmm0, %xmm4
; AVX1-NEXT:    vpblendvb %xmm2, %xmm4, %xmm0, %xmm0
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm2
; AVX1-NEXT:    vpblendvb %xmm3, %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpsrlw $2, %xmm0, %xmm2
; AVX1-NEXT:    vpaddw %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpblendvb %xmm3, %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpsrlw $1, %xmm0, %xmm2
; AVX1-NEXT:    vpaddw %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpblendvb %xmm3, %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: var_rotate_v8i16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpsubw %xmm1, %xmm2, %xmm2
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm1 = xmm1[0],zero,xmm1[1],zero,xmm1[2],zero,xmm1[3],zero,xmm1[4],zero,xmm1[5],zero,xmm1[6],zero,xmm1[7],zero
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vpsllvd %ymm1, %ymm0, %ymm1
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm3 = [0,1,4,5,8,9,12,13,128,128,128,128,128,128,128,128,0,1,4,5,8,9,12,13,128,128,128,128,128,128,128,128]
; AVX2-NEXT:    vpshufb %ymm3, %ymm1, %ymm1
; AVX2-NEXT:    vpermq {{.*#+}} ymm1 = ymm1[0,2,2,3]
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm2 = xmm2[0],zero,xmm2[1],zero,xmm2[2],zero,xmm2[3],zero,xmm2[4],zero,xmm2[5],zero,xmm2[6],zero,xmm2[7],zero
; AVX2-NEXT:    vpsrlvd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb %ymm3, %ymm0, %ymm0
; AVX2-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,2,2,3]
; AVX2-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; XOP-LABEL: var_rotate_v8i16:
; XOP:       # BB#0:
; XOP-NEXT:    vprotw %xmm1, %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: var_rotate_v8i16:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm3 = [16,16,16,16,16,16,16,16]
; X32-SSE-NEXT:    psubw %xmm1, %xmm3
; X32-SSE-NEXT:    psllw $12, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm2
; X32-SSE-NEXT:    psraw $15, %xmm2
; X32-SSE-NEXT:    movdqa %xmm0, %xmm4
; X32-SSE-NEXT:    psllw $8, %xmm4
; X32-SSE-NEXT:    pand %xmm2, %xmm4
; X32-SSE-NEXT:    pandn %xmm0, %xmm2
; X32-SSE-NEXT:    por %xmm4, %xmm2
; X32-SSE-NEXT:    paddw %xmm1, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm4
; X32-SSE-NEXT:    psraw $15, %xmm4
; X32-SSE-NEXT:    movdqa %xmm4, %xmm5
; X32-SSE-NEXT:    pandn %xmm2, %xmm5
; X32-SSE-NEXT:    psllw $4, %xmm2
; X32-SSE-NEXT:    pand %xmm4, %xmm2
; X32-SSE-NEXT:    por %xmm5, %xmm2
; X32-SSE-NEXT:    paddw %xmm1, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm4
; X32-SSE-NEXT:    psraw $15, %xmm4
; X32-SSE-NEXT:    movdqa %xmm4, %xmm5
; X32-SSE-NEXT:    pandn %xmm2, %xmm5
; X32-SSE-NEXT:    psllw $2, %xmm2
; X32-SSE-NEXT:    pand %xmm4, %xmm2
; X32-SSE-NEXT:    por %xmm5, %xmm2
; X32-SSE-NEXT:    paddw %xmm1, %xmm1
; X32-SSE-NEXT:    psraw $15, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm4
; X32-SSE-NEXT:    pandn %xmm2, %xmm4
; X32-SSE-NEXT:    psllw $1, %xmm2
; X32-SSE-NEXT:    pand %xmm1, %xmm2
; X32-SSE-NEXT:    psllw $12, %xmm3
; X32-SSE-NEXT:    movdqa %xmm3, %xmm1
; X32-SSE-NEXT:    psraw $15, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm5
; X32-SSE-NEXT:    pandn %xmm0, %xmm5
; X32-SSE-NEXT:    psrlw $8, %xmm0
; X32-SSE-NEXT:    pand %xmm1, %xmm0
; X32-SSE-NEXT:    por %xmm5, %xmm0
; X32-SSE-NEXT:    paddw %xmm3, %xmm3
; X32-SSE-NEXT:    movdqa %xmm3, %xmm1
; X32-SSE-NEXT:    psraw $15, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm5
; X32-SSE-NEXT:    pandn %xmm0, %xmm5
; X32-SSE-NEXT:    psrlw $4, %xmm0
; X32-SSE-NEXT:    pand %xmm1, %xmm0
; X32-SSE-NEXT:    por %xmm5, %xmm0
; X32-SSE-NEXT:    paddw %xmm3, %xmm3
; X32-SSE-NEXT:    movdqa %xmm3, %xmm1
; X32-SSE-NEXT:    psraw $15, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm5
; X32-SSE-NEXT:    pandn %xmm0, %xmm5
; X32-SSE-NEXT:    psrlw $2, %xmm0
; X32-SSE-NEXT:    pand %xmm1, %xmm0
; X32-SSE-NEXT:    por %xmm5, %xmm0
; X32-SSE-NEXT:    paddw %xmm3, %xmm3
; X32-SSE-NEXT:    psraw $15, %xmm3
; X32-SSE-NEXT:    movdqa %xmm3, %xmm1
; X32-SSE-NEXT:    pandn %xmm0, %xmm1
; X32-SSE-NEXT:    psrlw $1, %xmm0
; X32-SSE-NEXT:    pand %xmm3, %xmm0
; X32-SSE-NEXT:    por %xmm1, %xmm0
; X32-SSE-NEXT:    por %xmm4, %xmm0
; X32-SSE-NEXT:    por %xmm2, %xmm0
; X32-SSE-NEXT:    retl
  %b16 = sub <8 x i16> <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>, %b
  %shl = shl <8 x i16> %a, %b
  %lshr = lshr <8 x i16> %a, %b16
  %or = or <8 x i16> %shl, %lshr
  ret <8 x i16> %or
}

define <16 x i8> @var_rotate_v16i8(<16 x i8> %a, <16 x i8> %b) nounwind {
; SSE2-LABEL: var_rotate_v16i8:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa {{.*#+}} xmm4 = [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
; SSE2-NEXT:    psubb %xmm1, %xmm4
; SSE2-NEXT:    psllw $5, %xmm1
; SSE2-NEXT:    pxor %xmm3, %xmm3
; SSE2-NEXT:    pxor %xmm2, %xmm2
; SSE2-NEXT:    pcmpgtb %xmm1, %xmm2
; SSE2-NEXT:    movdqa %xmm0, %xmm5
; SSE2-NEXT:    psllw $4, %xmm5
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm5
; SSE2-NEXT:    pand %xmm2, %xmm5
; SSE2-NEXT:    pandn %xmm0, %xmm2
; SSE2-NEXT:    por %xmm5, %xmm2
; SSE2-NEXT:    paddb %xmm1, %xmm1
; SSE2-NEXT:    pxor %xmm5, %xmm5
; SSE2-NEXT:    pcmpgtb %xmm1, %xmm5
; SSE2-NEXT:    movdqa %xmm5, %xmm6
; SSE2-NEXT:    pandn %xmm2, %xmm6
; SSE2-NEXT:    psllw $2, %xmm2
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm2
; SSE2-NEXT:    pand %xmm5, %xmm2
; SSE2-NEXT:    por %xmm6, %xmm2
; SSE2-NEXT:    paddb %xmm1, %xmm1
; SSE2-NEXT:    pxor %xmm5, %xmm5
; SSE2-NEXT:    pcmpgtb %xmm1, %xmm5
; SSE2-NEXT:    movdqa %xmm5, %xmm1
; SSE2-NEXT:    pandn %xmm2, %xmm1
; SSE2-NEXT:    paddb %xmm2, %xmm2
; SSE2-NEXT:    pand %xmm5, %xmm2
; SSE2-NEXT:    psllw $5, %xmm4
; SSE2-NEXT:    pxor %xmm5, %xmm5
; SSE2-NEXT:    pcmpgtb %xmm4, %xmm5
; SSE2-NEXT:    movdqa %xmm5, %xmm6
; SSE2-NEXT:    pandn %xmm0, %xmm6
; SSE2-NEXT:    psrlw $4, %xmm0
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE2-NEXT:    pand %xmm5, %xmm0
; SSE2-NEXT:    por %xmm6, %xmm0
; SSE2-NEXT:    paddb %xmm4, %xmm4
; SSE2-NEXT:    pxor %xmm5, %xmm5
; SSE2-NEXT:    pcmpgtb %xmm4, %xmm5
; SSE2-NEXT:    movdqa %xmm5, %xmm6
; SSE2-NEXT:    pandn %xmm0, %xmm6
; SSE2-NEXT:    psrlw $2, %xmm0
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE2-NEXT:    pand %xmm5, %xmm0
; SSE2-NEXT:    por %xmm6, %xmm0
; SSE2-NEXT:    paddb %xmm4, %xmm4
; SSE2-NEXT:    pcmpgtb %xmm4, %xmm3
; SSE2-NEXT:    movdqa %xmm3, %xmm4
; SSE2-NEXT:    pandn %xmm0, %xmm4
; SSE2-NEXT:    psrlw $1, %xmm0
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE2-NEXT:    pand %xmm3, %xmm0
; SSE2-NEXT:    por %xmm4, %xmm0
; SSE2-NEXT:    por %xmm1, %xmm0
; SSE2-NEXT:    por %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: var_rotate_v16i8:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa %xmm1, %xmm3
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    movdqa {{.*#+}} xmm2 = [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
; SSE41-NEXT:    psubb %xmm3, %xmm2
; SSE41-NEXT:    psllw $5, %xmm3
; SSE41-NEXT:    movdqa %xmm1, %xmm5
; SSE41-NEXT:    psllw $4, %xmm5
; SSE41-NEXT:    pand {{.*}}(%rip), %xmm5
; SSE41-NEXT:    movdqa %xmm1, %xmm4
; SSE41-NEXT:    movdqa %xmm3, %xmm0
; SSE41-NEXT:    pblendvb %xmm5, %xmm4
; SSE41-NEXT:    movdqa %xmm4, %xmm5
; SSE41-NEXT:    psllw $2, %xmm5
; SSE41-NEXT:    pand {{.*}}(%rip), %xmm5
; SSE41-NEXT:    paddb %xmm3, %xmm3
; SSE41-NEXT:    movdqa %xmm3, %xmm0
; SSE41-NEXT:    pblendvb %xmm5, %xmm4
; SSE41-NEXT:    movdqa %xmm4, %xmm5
; SSE41-NEXT:    paddb %xmm5, %xmm5
; SSE41-NEXT:    paddb %xmm3, %xmm3
; SSE41-NEXT:    movdqa %xmm3, %xmm0
; SSE41-NEXT:    pblendvb %xmm5, %xmm4
; SSE41-NEXT:    psllw $5, %xmm2
; SSE41-NEXT:    movdqa %xmm2, %xmm3
; SSE41-NEXT:    paddb %xmm3, %xmm3
; SSE41-NEXT:    movdqa %xmm1, %xmm5
; SSE41-NEXT:    psrlw $4, %xmm5
; SSE41-NEXT:    pand {{.*}}(%rip), %xmm5
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    pblendvb %xmm5, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm2
; SSE41-NEXT:    psrlw $2, %xmm2
; SSE41-NEXT:    pand {{.*}}(%rip), %xmm2
; SSE41-NEXT:    movdqa %xmm3, %xmm0
; SSE41-NEXT:    pblendvb %xmm2, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm2
; SSE41-NEXT:    psrlw $1, %xmm2
; SSE41-NEXT:    pand {{.*}}(%rip), %xmm2
; SSE41-NEXT:    paddb %xmm3, %xmm3
; SSE41-NEXT:    movdqa %xmm3, %xmm0
; SSE41-NEXT:    pblendvb %xmm2, %xmm1
; SSE41-NEXT:    por %xmm4, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: var_rotate_v16i8:
; AVX:       # BB#0:
; AVX-NEXT:    vmovdqa {{.*#+}} xmm2 = [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
; AVX-NEXT:    vpsubb %xmm1, %xmm2, %xmm2
; AVX-NEXT:    vpsllw $5, %xmm1, %xmm1
; AVX-NEXT:    vpsllw $4, %xmm0, %xmm3
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm3, %xmm3
; AVX-NEXT:    vpblendvb %xmm1, %xmm3, %xmm0, %xmm3
; AVX-NEXT:    vpsllw $2, %xmm3, %xmm4
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm4, %xmm4
; AVX-NEXT:    vpaddb %xmm1, %xmm1, %xmm1
; AVX-NEXT:    vpblendvb %xmm1, %xmm4, %xmm3, %xmm3
; AVX-NEXT:    vpaddb %xmm3, %xmm3, %xmm4
; AVX-NEXT:    vpaddb %xmm1, %xmm1, %xmm1
; AVX-NEXT:    vpblendvb %xmm1, %xmm4, %xmm3, %xmm1
; AVX-NEXT:    vpsllw $5, %xmm2, %xmm2
; AVX-NEXT:    vpaddb %xmm2, %xmm2, %xmm3
; AVX-NEXT:    vpsrlw $4, %xmm0, %xmm4
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm4, %xmm4
; AVX-NEXT:    vpblendvb %xmm2, %xmm4, %xmm0, %xmm0
; AVX-NEXT:    vpsrlw $2, %xmm0, %xmm2
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm2, %xmm2
; AVX-NEXT:    vpblendvb %xmm3, %xmm2, %xmm0, %xmm0
; AVX-NEXT:    vpsrlw $1, %xmm0, %xmm2
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm2, %xmm2
; AVX-NEXT:    vpaddb %xmm3, %xmm3, %xmm3
; AVX-NEXT:    vpblendvb %xmm3, %xmm2, %xmm0, %xmm0
; AVX-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
;
; XOP-LABEL: var_rotate_v16i8:
; XOP:       # BB#0:
; XOP-NEXT:    vprotb %xmm1, %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: var_rotate_v16i8:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm4 = [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
; X32-SSE-NEXT:    psubb %xmm1, %xmm4
; X32-SSE-NEXT:    psllw $5, %xmm1
; X32-SSE-NEXT:    pxor %xmm3, %xmm3
; X32-SSE-NEXT:    pxor %xmm2, %xmm2
; X32-SSE-NEXT:    pcmpgtb %xmm1, %xmm2
; X32-SSE-NEXT:    movdqa %xmm0, %xmm5
; X32-SSE-NEXT:    psllw $4, %xmm5
; X32-SSE-NEXT:    pand .LCPI3_1, %xmm5
; X32-SSE-NEXT:    pand %xmm2, %xmm5
; X32-SSE-NEXT:    pandn %xmm0, %xmm2
; X32-SSE-NEXT:    por %xmm5, %xmm2
; X32-SSE-NEXT:    paddb %xmm1, %xmm1
; X32-SSE-NEXT:    pxor %xmm5, %xmm5
; X32-SSE-NEXT:    pcmpgtb %xmm1, %xmm5
; X32-SSE-NEXT:    movdqa %xmm5, %xmm6
; X32-SSE-NEXT:    pandn %xmm2, %xmm6
; X32-SSE-NEXT:    psllw $2, %xmm2
; X32-SSE-NEXT:    pand .LCPI3_2, %xmm2
; X32-SSE-NEXT:    pand %xmm5, %xmm2
; X32-SSE-NEXT:    por %xmm6, %xmm2
; X32-SSE-NEXT:    paddb %xmm1, %xmm1
; X32-SSE-NEXT:    pxor %xmm5, %xmm5
; X32-SSE-NEXT:    pcmpgtb %xmm1, %xmm5
; X32-SSE-NEXT:    movdqa %xmm5, %xmm1
; X32-SSE-NEXT:    pandn %xmm2, %xmm1
; X32-SSE-NEXT:    paddb %xmm2, %xmm2
; X32-SSE-NEXT:    pand %xmm5, %xmm2
; X32-SSE-NEXT:    psllw $5, %xmm4
; X32-SSE-NEXT:    pxor %xmm5, %xmm5
; X32-SSE-NEXT:    pcmpgtb %xmm4, %xmm5
; X32-SSE-NEXT:    movdqa %xmm5, %xmm6
; X32-SSE-NEXT:    pandn %xmm0, %xmm6
; X32-SSE-NEXT:    psrlw $4, %xmm0
; X32-SSE-NEXT:    pand .LCPI3_3, %xmm0
; X32-SSE-NEXT:    pand %xmm5, %xmm0
; X32-SSE-NEXT:    por %xmm6, %xmm0
; X32-SSE-NEXT:    paddb %xmm4, %xmm4
; X32-SSE-NEXT:    pxor %xmm5, %xmm5
; X32-SSE-NEXT:    pcmpgtb %xmm4, %xmm5
; X32-SSE-NEXT:    movdqa %xmm5, %xmm6
; X32-SSE-NEXT:    pandn %xmm0, %xmm6
; X32-SSE-NEXT:    psrlw $2, %xmm0
; X32-SSE-NEXT:    pand .LCPI3_4, %xmm0
; X32-SSE-NEXT:    pand %xmm5, %xmm0
; X32-SSE-NEXT:    por %xmm6, %xmm0
; X32-SSE-NEXT:    paddb %xmm4, %xmm4
; X32-SSE-NEXT:    pcmpgtb %xmm4, %xmm3
; X32-SSE-NEXT:    movdqa %xmm3, %xmm4
; X32-SSE-NEXT:    pandn %xmm0, %xmm4
; X32-SSE-NEXT:    psrlw $1, %xmm0
; X32-SSE-NEXT:    pand .LCPI3_5, %xmm0
; X32-SSE-NEXT:    pand %xmm3, %xmm0
; X32-SSE-NEXT:    por %xmm4, %xmm0
; X32-SSE-NEXT:    por %xmm1, %xmm0
; X32-SSE-NEXT:    por %xmm2, %xmm0
; X32-SSE-NEXT:    retl
  %b8 = sub <16 x i8> <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>, %b
  %shl = shl <16 x i8> %a, %b
  %lshr = lshr <16 x i8> %a, %b8
  %or = or <16 x i8> %shl, %lshr
  ret <16 x i8> %or
}

;
; Constant Rotates
;

define <2 x i64> @constant_rotate_v2i64(<2 x i64> %a) nounwind {
; SSE2-LABEL: constant_rotate_v2i64:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa %xmm0, %xmm2
; SSE2-NEXT:    psllq $14, %xmm2
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    psllq $4, %xmm1
; SSE2-NEXT:    movsd {{.*#+}} xmm2 = xmm1[0],xmm2[1]
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    psrlq $50, %xmm1
; SSE2-NEXT:    psrlq $60, %xmm0
; SSE2-NEXT:    movsd {{.*#+}} xmm1 = xmm0[0],xmm1[1]
; SSE2-NEXT:    orpd %xmm2, %xmm1
; SSE2-NEXT:    movapd %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: constant_rotate_v2i64:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    psllq $14, %xmm1
; SSE41-NEXT:    movdqa %xmm0, %xmm2
; SSE41-NEXT:    psllq $4, %xmm2
; SSE41-NEXT:    pblendw {{.*#+}} xmm2 = xmm2[0,1,2,3],xmm1[4,5,6,7]
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    psrlq $50, %xmm1
; SSE41-NEXT:    psrlq $60, %xmm0
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; SSE41-NEXT:    por %xmm2, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: constant_rotate_v2i64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpsllq $14, %xmm0, %xmm1
; AVX1-NEXT:    vpsllq $4, %xmm0, %xmm2
; AVX1-NEXT:    vpblendw {{.*#+}} xmm1 = xmm2[0,1,2,3],xmm1[4,5,6,7]
; AVX1-NEXT:    vpsrlq $50, %xmm0, %xmm2
; AVX1-NEXT:    vpsrlq $60, %xmm0, %xmm0
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm2[4,5,6,7]
; AVX1-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: constant_rotate_v2i64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsllvq {{.*}}(%rip), %xmm0, %xmm1
; AVX2-NEXT:    vpsrlvq {{.*}}(%rip), %xmm0, %xmm0
; AVX2-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    retq
;
; XOPAVX1-LABEL: constant_rotate_v2i64:
; XOPAVX1:       # BB#0:
; XOPAVX1-NEXT:    vpshlq {{.*}}(%rip), %xmm0, %xmm1
; XOPAVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; XOPAVX1-NEXT:    vpsubq {{.*}}(%rip), %xmm2, %xmm2
; XOPAVX1-NEXT:    vpshlq %xmm2, %xmm0, %xmm0
; XOPAVX1-NEXT:    vpor %xmm0, %xmm1, %xmm0
; XOPAVX1-NEXT:    retq
;
; XOPAVX2-LABEL: constant_rotate_v2i64:
; XOPAVX2:       # BB#0:
; XOPAVX2-NEXT:    vpsllvq {{.*}}(%rip), %xmm0, %xmm1
; XOPAVX2-NEXT:    vpsrlvq {{.*}}(%rip), %xmm0, %xmm0
; XOPAVX2-NEXT:    vpor %xmm0, %xmm1, %xmm0
; XOPAVX2-NEXT:    retq
;
; X32-SSE-LABEL: constant_rotate_v2i64:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa %xmm0, %xmm2
; X32-SSE-NEXT:    psllq $14, %xmm2
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    psllq $4, %xmm1
; X32-SSE-NEXT:    movsd {{.*#+}} xmm2 = xmm1[0],xmm2[1]
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    psrlq $50, %xmm1
; X32-SSE-NEXT:    psrlq $60, %xmm0
; X32-SSE-NEXT:    movsd {{.*#+}} xmm1 = xmm0[0],xmm1[1]
; X32-SSE-NEXT:    orpd %xmm2, %xmm1
; X32-SSE-NEXT:    movapd %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <2 x i64> %a, <i64 4, i64 14>
  %lshr = lshr <2 x i64> %a, <i64 60, i64 50>
  %or = or <2 x i64> %shl, %lshr
  ret <2 x i64> %or
}

define <4 x i32> @constant_rotate_v4i32(<4 x i32> %a) nounwind {
; SSE2-LABEL: constant_rotate_v4i32:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa {{.*#+}} xmm1 = [16,32,64,128]
; SSE2-NEXT:    movdqa %xmm0, %xmm2
; SSE2-NEXT:    pmuludq %xmm1, %xmm2
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[0,2,2,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[1,1,3,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm3 = xmm0[1,1,3,3]
; SSE2-NEXT:    pmuludq %xmm1, %xmm3
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm3[0,2,2,3]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    psrld $25, %xmm1
; SSE2-NEXT:    movdqa %xmm0, %xmm3
; SSE2-NEXT:    psrld $27, %xmm3
; SSE2-NEXT:    movsd {{.*#+}} xmm1 = xmm3[0],xmm1[1]
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[1,3,2,3]
; SSE2-NEXT:    movdqa %xmm0, %xmm3
; SSE2-NEXT:    psrld $26, %xmm3
; SSE2-NEXT:    psrld $28, %xmm0
; SSE2-NEXT:    movsd {{.*#+}} xmm3 = xmm0[0],xmm3[1]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm3[0,2,2,3]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE2-NEXT:    por %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: constant_rotate_v4i32:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm1 = [16,32,64,128]
; SSE41-NEXT:    pmulld %xmm0, %xmm1
; SSE41-NEXT:    movdqa %xmm0, %xmm2
; SSE41-NEXT:    psrld $25, %xmm2
; SSE41-NEXT:    movdqa %xmm0, %xmm3
; SSE41-NEXT:    psrld $27, %xmm3
; SSE41-NEXT:    pblendw {{.*#+}} xmm3 = xmm3[0,1,2,3],xmm2[4,5,6,7]
; SSE41-NEXT:    movdqa %xmm0, %xmm2
; SSE41-NEXT:    psrld $26, %xmm2
; SSE41-NEXT:    psrld $28, %xmm0
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm2[4,5,6,7]
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm3[2,3],xmm0[4,5],xmm3[6,7]
; SSE41-NEXT:    por %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: constant_rotate_v4i32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmulld {{.*}}(%rip), %xmm0, %xmm1
; AVX1-NEXT:    vpsrld $25, %xmm0, %xmm2
; AVX1-NEXT:    vpsrld $27, %xmm0, %xmm3
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm3[0,1,2,3],xmm2[4,5,6,7]
; AVX1-NEXT:    vpsrld $26, %xmm0, %xmm3
; AVX1-NEXT:    vpsrld $28, %xmm0, %xmm0
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm3[4,5,6,7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm2[2,3],xmm0[4,5],xmm2[6,7]
; AVX1-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: constant_rotate_v4i32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsllvd {{.*}}(%rip), %xmm0, %xmm1
; AVX2-NEXT:    vpsrlvd {{.*}}(%rip), %xmm0, %xmm0
; AVX2-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    retq
;
; XOPAVX1-LABEL: constant_rotate_v4i32:
; XOPAVX1:       # BB#0:
; XOPAVX1-NEXT:    vpshld {{.*}}(%rip), %xmm0, %xmm1
; XOPAVX1-NEXT:    vpshld {{.*}}(%rip), %xmm0, %xmm0
; XOPAVX1-NEXT:    vpor %xmm0, %xmm1, %xmm0
; XOPAVX1-NEXT:    retq
;
; XOPAVX2-LABEL: constant_rotate_v4i32:
; XOPAVX2:       # BB#0:
; XOPAVX2-NEXT:    vpsllvd {{.*}}(%rip), %xmm0, %xmm1
; XOPAVX2-NEXT:    vpsrlvd {{.*}}(%rip), %xmm0, %xmm0
; XOPAVX2-NEXT:    vpor %xmm0, %xmm1, %xmm0
; XOPAVX2-NEXT:    retq
;
; X32-SSE-LABEL: constant_rotate_v4i32:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm1 = [16,32,64,128]
; X32-SSE-NEXT:    movdqa %xmm0, %xmm2
; X32-SSE-NEXT:    pmuludq %xmm1, %xmm2
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[0,2,2,3]
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[1,1,3,3]
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm0[1,1,3,3]
; X32-SSE-NEXT:    pmuludq %xmm1, %xmm3
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm3[0,2,2,3]
; X32-SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    psrld $25, %xmm1
; X32-SSE-NEXT:    movdqa %xmm0, %xmm3
; X32-SSE-NEXT:    psrld $27, %xmm3
; X32-SSE-NEXT:    movsd {{.*#+}} xmm1 = xmm3[0],xmm1[1]
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[1,3,2,3]
; X32-SSE-NEXT:    movdqa %xmm0, %xmm3
; X32-SSE-NEXT:    psrld $26, %xmm3
; X32-SSE-NEXT:    psrld $28, %xmm0
; X32-SSE-NEXT:    movsd {{.*#+}} xmm3 = xmm0[0],xmm3[1]
; X32-SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm3[0,2,2,3]
; X32-SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; X32-SSE-NEXT:    por %xmm2, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <4 x i32> %a, <i32 4, i32 5, i32 6, i32 7>
  %lshr = lshr <4 x i32> %a, <i32 28, i32 27, i32 26, i32 25>
  %or = or <4 x i32> %shl, %lshr
  ret <4 x i32> %or
}

define <8 x i16> @constant_rotate_v8i16(<8 x i16> %a) nounwind {
; SSE2-LABEL: constant_rotate_v8i16:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa {{.*#+}} xmm2 = [1,2,4,8,16,32,64,128]
; SSE2-NEXT:    pmullw %xmm0, %xmm2
; SSE2-NEXT:    movdqa {{.*#+}} xmm1 = [0,65535,65535,65535,65535,65535,65535,65535]
; SSE2-NEXT:    movdqa %xmm1, %xmm3
; SSE2-NEXT:    pandn %xmm0, %xmm3
; SSE2-NEXT:    psrlw $8, %xmm0
; SSE2-NEXT:    pand %xmm1, %xmm0
; SSE2-NEXT:    por %xmm3, %xmm0
; SSE2-NEXT:    movdqa {{.*#+}} xmm1 = [0,65535,65535,65535,65535,0,0,0]
; SSE2-NEXT:    movdqa %xmm1, %xmm3
; SSE2-NEXT:    pandn %xmm0, %xmm3
; SSE2-NEXT:    psrlw $4, %xmm0
; SSE2-NEXT:    pand %xmm1, %xmm0
; SSE2-NEXT:    por %xmm3, %xmm0
; SSE2-NEXT:    movdqa {{.*#+}} xmm1 = [0,65535,65535,0,0,65535,65535,0]
; SSE2-NEXT:    movdqa %xmm1, %xmm3
; SSE2-NEXT:    pandn %xmm0, %xmm3
; SSE2-NEXT:    psrlw $2, %xmm0
; SSE2-NEXT:    pand %xmm1, %xmm0
; SSE2-NEXT:    por %xmm3, %xmm0
; SSE2-NEXT:    movdqa {{.*#+}} xmm3 = [65535,0,65535,0,65535,0,65535,0]
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    pand %xmm3, %xmm1
; SSE2-NEXT:    psrlw $1, %xmm0
; SSE2-NEXT:    pandn %xmm0, %xmm3
; SSE2-NEXT:    por %xmm2, %xmm3
; SSE2-NEXT:    por %xmm3, %xmm1
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: constant_rotate_v8i16:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    movdqa {{.*#+}} xmm2 = [1,2,4,8,16,32,64,128]
; SSE41-NEXT:    pmullw %xmm1, %xmm2
; SSE41-NEXT:    movdqa %xmm1, %xmm3
; SSE41-NEXT:    psrlw $8, %xmm3
; SSE41-NEXT:    movaps {{.*#+}} xmm0 = [256,61680,57568,53456,49344,45232,41120,37008]
; SSE41-NEXT:    pblendvb %xmm3, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm3
; SSE41-NEXT:    psrlw $4, %xmm3
; SSE41-NEXT:    movaps {{.*#+}} xmm0 = [512,57824,49600,41376,33152,24928,16704,8480]
; SSE41-NEXT:    pblendvb %xmm3, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm3
; SSE41-NEXT:    psrlw $2, %xmm3
; SSE41-NEXT:    movaps {{.*#+}} xmm0 = [1024,50112,33664,17216,768,49856,33408,16960]
; SSE41-NEXT:    pblendvb %xmm3, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm3
; SSE41-NEXT:    psrlw $1, %xmm3
; SSE41-NEXT:    movaps {{.*#+}} xmm0 = [2048,34688,1792,34432,1536,34176,1280,33920]
; SSE41-NEXT:    pblendvb %xmm3, %xmm1
; SSE41-NEXT:    por %xmm2, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: constant_rotate_v8i16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpmullw {{.*}}(%rip), %xmm0, %xmm1
; AVX1-NEXT:    vpsrlw $8, %xmm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [256,61680,57568,53456,49344,45232,41120,37008]
; AVX1-NEXT:    vpblendvb %xmm3, %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpsrlw $4, %xmm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [512,57824,49600,41376,33152,24928,16704,8480]
; AVX1-NEXT:    vpblendvb %xmm3, %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpsrlw $2, %xmm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [1024,50112,33664,17216,768,49856,33408,16960]
; AVX1-NEXT:    vpblendvb %xmm3, %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpsrlw $1, %xmm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = [2048,34688,1792,34432,1536,34176,1280,33920]
; AVX1-NEXT:    vpblendvb %xmm3, %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: constant_rotate_v8i16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpmullw {{.*}}(%rip), %xmm0, %xmm1
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm2 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero
; AVX2-NEXT:    vpsrlvd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,4,5,8,9,12,13],zero,zero,zero,zero,zero,zero,zero,zero,ymm0[16,17,20,21,24,25,28,29],zero,zero,zero,zero,zero,zero,zero,zero
; AVX2-NEXT:    vpermq {{.*#+}} ymm0 = ymm0[0,2,2,3]
; AVX2-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; XOP-LABEL: constant_rotate_v8i16:
; XOP:       # BB#0:
; XOP-NEXT:    vpshlw {{.*}}(%rip), %xmm0, %xmm1
; XOP-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; XOP-NEXT:    vpsubw {{.*}}(%rip), %xmm2, %xmm2
; XOP-NEXT:    vpshlw %xmm2, %xmm0, %xmm0
; XOP-NEXT:    vpor %xmm0, %xmm1, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: constant_rotate_v8i16:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm2 = [1,2,4,8,16,32,64,128]
; X32-SSE-NEXT:    pmullw %xmm0, %xmm2
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm1 = [0,65535,65535,65535,65535,65535,65535,65535]
; X32-SSE-NEXT:    movdqa %xmm1, %xmm3
; X32-SSE-NEXT:    pandn %xmm0, %xmm3
; X32-SSE-NEXT:    psrlw $8, %xmm0
; X32-SSE-NEXT:    pand %xmm1, %xmm0
; X32-SSE-NEXT:    por %xmm3, %xmm0
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm1 = [0,65535,65535,65535,65535,0,0,0]
; X32-SSE-NEXT:    movdqa %xmm1, %xmm3
; X32-SSE-NEXT:    pandn %xmm0, %xmm3
; X32-SSE-NEXT:    psrlw $4, %xmm0
; X32-SSE-NEXT:    pand %xmm1, %xmm0
; X32-SSE-NEXT:    por %xmm3, %xmm0
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm1 = [0,65535,65535,0,0,65535,65535,0]
; X32-SSE-NEXT:    movdqa %xmm1, %xmm3
; X32-SSE-NEXT:    pandn %xmm0, %xmm3
; X32-SSE-NEXT:    psrlw $2, %xmm0
; X32-SSE-NEXT:    pand %xmm1, %xmm0
; X32-SSE-NEXT:    por %xmm3, %xmm0
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm3 = [65535,0,65535,0,65535,0,65535,0]
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    pand %xmm3, %xmm1
; X32-SSE-NEXT:    psrlw $1, %xmm0
; X32-SSE-NEXT:    pandn %xmm0, %xmm3
; X32-SSE-NEXT:    por %xmm2, %xmm3
; X32-SSE-NEXT:    por %xmm3, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <8 x i16> %a, <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>
  %lshr = lshr <8 x i16> %a, <i16 16, i16 15, i16 14, i16 13, i16 12, i16 11, i16 10, i16 9>
  %or = or <8 x i16> %shl, %lshr
  ret <8 x i16> %or
}

define <16 x i8> @constant_rotate_v16i8(<16 x i8> %a) nounwind {
; SSE2-LABEL: constant_rotate_v16i8:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa {{.*#+}} xmm3 = [0,1,2,3,4,5,6,7,8,7,6,5,4,3,2,1]
; SSE2-NEXT:    psllw $5, %xmm3
; SSE2-NEXT:    pxor %xmm2, %xmm2
; SSE2-NEXT:    pxor %xmm1, %xmm1
; SSE2-NEXT:    pcmpgtb %xmm3, %xmm1
; SSE2-NEXT:    movdqa %xmm0, %xmm4
; SSE2-NEXT:    psllw $4, %xmm4
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm4
; SSE2-NEXT:    pand %xmm1, %xmm4
; SSE2-NEXT:    pandn %xmm0, %xmm1
; SSE2-NEXT:    por %xmm4, %xmm1
; SSE2-NEXT:    paddb %xmm3, %xmm3
; SSE2-NEXT:    pxor %xmm4, %xmm4
; SSE2-NEXT:    pcmpgtb %xmm3, %xmm4
; SSE2-NEXT:    movdqa %xmm4, %xmm5
; SSE2-NEXT:    pandn %xmm1, %xmm5
; SSE2-NEXT:    psllw $2, %xmm1
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE2-NEXT:    pand %xmm4, %xmm1
; SSE2-NEXT:    por %xmm5, %xmm1
; SSE2-NEXT:    paddb %xmm3, %xmm3
; SSE2-NEXT:    pxor %xmm4, %xmm4
; SSE2-NEXT:    pcmpgtb %xmm3, %xmm4
; SSE2-NEXT:    movdqa %xmm4, %xmm3
; SSE2-NEXT:    pandn %xmm1, %xmm3
; SSE2-NEXT:    paddb %xmm1, %xmm1
; SSE2-NEXT:    pand %xmm4, %xmm1
; SSE2-NEXT:    movdqa {{.*#+}} xmm4 = [8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7]
; SSE2-NEXT:    psllw $5, %xmm4
; SSE2-NEXT:    pxor %xmm5, %xmm5
; SSE2-NEXT:    pcmpgtb %xmm4, %xmm5
; SSE2-NEXT:    movdqa %xmm5, %xmm6
; SSE2-NEXT:    pandn %xmm0, %xmm6
; SSE2-NEXT:    psrlw $4, %xmm0
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE2-NEXT:    pand %xmm5, %xmm0
; SSE2-NEXT:    por %xmm6, %xmm0
; SSE2-NEXT:    paddb %xmm4, %xmm4
; SSE2-NEXT:    pxor %xmm5, %xmm5
; SSE2-NEXT:    pcmpgtb %xmm4, %xmm5
; SSE2-NEXT:    movdqa %xmm5, %xmm6
; SSE2-NEXT:    pandn %xmm0, %xmm6
; SSE2-NEXT:    psrlw $2, %xmm0
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE2-NEXT:    pand %xmm5, %xmm0
; SSE2-NEXT:    por %xmm6, %xmm0
; SSE2-NEXT:    paddb %xmm4, %xmm4
; SSE2-NEXT:    pcmpgtb %xmm4, %xmm2
; SSE2-NEXT:    movdqa %xmm2, %xmm4
; SSE2-NEXT:    pandn %xmm0, %xmm4
; SSE2-NEXT:    psrlw $1, %xmm0
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE2-NEXT:    pand %xmm2, %xmm0
; SSE2-NEXT:    por %xmm4, %xmm0
; SSE2-NEXT:    por %xmm3, %xmm0
; SSE2-NEXT:    por %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: constant_rotate_v16i8:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    movdqa {{.*#+}} xmm0 = [0,1,2,3,4,5,6,7,8,7,6,5,4,3,2,1]
; SSE41-NEXT:    psllw $5, %xmm0
; SSE41-NEXT:    movdqa %xmm1, %xmm3
; SSE41-NEXT:    psllw $4, %xmm3
; SSE41-NEXT:    pand {{.*}}(%rip), %xmm3
; SSE41-NEXT:    movdqa %xmm1, %xmm2
; SSE41-NEXT:    pblendvb %xmm3, %xmm2
; SSE41-NEXT:    movdqa %xmm2, %xmm3
; SSE41-NEXT:    psllw $2, %xmm3
; SSE41-NEXT:    pand {{.*}}(%rip), %xmm3
; SSE41-NEXT:    paddb %xmm0, %xmm0
; SSE41-NEXT:    pblendvb %xmm3, %xmm2
; SSE41-NEXT:    movdqa %xmm2, %xmm3
; SSE41-NEXT:    paddb %xmm3, %xmm3
; SSE41-NEXT:    paddb %xmm0, %xmm0
; SSE41-NEXT:    pblendvb %xmm3, %xmm2
; SSE41-NEXT:    movdqa {{.*#+}} xmm0 = [8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7]
; SSE41-NEXT:    psllw $5, %xmm0
; SSE41-NEXT:    movdqa %xmm1, %xmm3
; SSE41-NEXT:    psrlw $4, %xmm3
; SSE41-NEXT:    pand {{.*}}(%rip), %xmm3
; SSE41-NEXT:    pblendvb %xmm3, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm3
; SSE41-NEXT:    psrlw $2, %xmm3
; SSE41-NEXT:    pand {{.*}}(%rip), %xmm3
; SSE41-NEXT:    paddb %xmm0, %xmm0
; SSE41-NEXT:    pblendvb %xmm3, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm3
; SSE41-NEXT:    psrlw $1, %xmm3
; SSE41-NEXT:    pand {{.*}}(%rip), %xmm3
; SSE41-NEXT:    paddb %xmm0, %xmm0
; SSE41-NEXT:    pblendvb %xmm3, %xmm1
; SSE41-NEXT:    por %xmm2, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: constant_rotate_v16i8:
; AVX:       # BB#0:
; AVX-NEXT:    vmovdqa {{.*#+}} xmm1 = [0,1,2,3,4,5,6,7,8,7,6,5,4,3,2,1]
; AVX-NEXT:    vpsllw $5, %xmm1, %xmm1
; AVX-NEXT:    vpsllw $4, %xmm0, %xmm2
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm2, %xmm2
; AVX-NEXT:    vpblendvb %xmm1, %xmm2, %xmm0, %xmm2
; AVX-NEXT:    vpsllw $2, %xmm2, %xmm3
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm3, %xmm3
; AVX-NEXT:    vpaddb %xmm1, %xmm1, %xmm1
; AVX-NEXT:    vpblendvb %xmm1, %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vpaddb %xmm2, %xmm2, %xmm3
; AVX-NEXT:    vpaddb %xmm1, %xmm1, %xmm1
; AVX-NEXT:    vpblendvb %xmm1, %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vmovdqa {{.*#+}} xmm2 = [8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7]
; AVX-NEXT:    vpsllw $5, %xmm2, %xmm2
; AVX-NEXT:    vpsrlw $4, %xmm0, %xmm3
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm3, %xmm3
; AVX-NEXT:    vpblendvb %xmm2, %xmm3, %xmm0, %xmm0
; AVX-NEXT:    vpsrlw $2, %xmm0, %xmm3
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm3, %xmm3
; AVX-NEXT:    vpaddb %xmm2, %xmm2, %xmm2
; AVX-NEXT:    vpblendvb %xmm2, %xmm3, %xmm0, %xmm0
; AVX-NEXT:    vpsrlw $1, %xmm0, %xmm3
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm3, %xmm3
; AVX-NEXT:    vpaddb %xmm2, %xmm2, %xmm2
; AVX-NEXT:    vpblendvb %xmm2, %xmm3, %xmm0, %xmm0
; AVX-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
;
; XOP-LABEL: constant_rotate_v16i8:
; XOP:       # BB#0:
; XOP-NEXT:    vpshlb {{.*}}(%rip), %xmm0, %xmm1
; XOP-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; XOP-NEXT:    vpsubb {{.*}}(%rip), %xmm2, %xmm2
; XOP-NEXT:    vpshlb %xmm2, %xmm0, %xmm0
; XOP-NEXT:    vpor %xmm0, %xmm1, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: constant_rotate_v16i8:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm3 = [0,1,2,3,4,5,6,7,8,7,6,5,4,3,2,1]
; X32-SSE-NEXT:    psllw $5, %xmm3
; X32-SSE-NEXT:    pxor %xmm2, %xmm2
; X32-SSE-NEXT:    pxor %xmm1, %xmm1
; X32-SSE-NEXT:    pcmpgtb %xmm3, %xmm1
; X32-SSE-NEXT:    movdqa %xmm0, %xmm4
; X32-SSE-NEXT:    psllw $4, %xmm4
; X32-SSE-NEXT:    pand .LCPI7_1, %xmm4
; X32-SSE-NEXT:    pand %xmm1, %xmm4
; X32-SSE-NEXT:    pandn %xmm0, %xmm1
; X32-SSE-NEXT:    por %xmm4, %xmm1
; X32-SSE-NEXT:    paddb %xmm3, %xmm3
; X32-SSE-NEXT:    pxor %xmm4, %xmm4
; X32-SSE-NEXT:    pcmpgtb %xmm3, %xmm4
; X32-SSE-NEXT:    movdqa %xmm4, %xmm5
; X32-SSE-NEXT:    pandn %xmm1, %xmm5
; X32-SSE-NEXT:    psllw $2, %xmm1
; X32-SSE-NEXT:    pand .LCPI7_2, %xmm1
; X32-SSE-NEXT:    pand %xmm4, %xmm1
; X32-SSE-NEXT:    por %xmm5, %xmm1
; X32-SSE-NEXT:    paddb %xmm3, %xmm3
; X32-SSE-NEXT:    pxor %xmm4, %xmm4
; X32-SSE-NEXT:    pcmpgtb %xmm3, %xmm4
; X32-SSE-NEXT:    movdqa %xmm4, %xmm3
; X32-SSE-NEXT:    pandn %xmm1, %xmm3
; X32-SSE-NEXT:    paddb %xmm1, %xmm1
; X32-SSE-NEXT:    pand %xmm4, %xmm1
; X32-SSE-NEXT:    movdqa {{.*#+}} xmm4 = [8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7]
; X32-SSE-NEXT:    psllw $5, %xmm4
; X32-SSE-NEXT:    pxor %xmm5, %xmm5
; X32-SSE-NEXT:    pcmpgtb %xmm4, %xmm5
; X32-SSE-NEXT:    movdqa %xmm5, %xmm6
; X32-SSE-NEXT:    pandn %xmm0, %xmm6
; X32-SSE-NEXT:    psrlw $4, %xmm0
; X32-SSE-NEXT:    pand .LCPI7_4, %xmm0
; X32-SSE-NEXT:    pand %xmm5, %xmm0
; X32-SSE-NEXT:    por %xmm6, %xmm0
; X32-SSE-NEXT:    paddb %xmm4, %xmm4
; X32-SSE-NEXT:    pxor %xmm5, %xmm5
; X32-SSE-NEXT:    pcmpgtb %xmm4, %xmm5
; X32-SSE-NEXT:    movdqa %xmm5, %xmm6
; X32-SSE-NEXT:    pandn %xmm0, %xmm6
; X32-SSE-NEXT:    psrlw $2, %xmm0
; X32-SSE-NEXT:    pand .LCPI7_5, %xmm0
; X32-SSE-NEXT:    pand %xmm5, %xmm0
; X32-SSE-NEXT:    por %xmm6, %xmm0
; X32-SSE-NEXT:    paddb %xmm4, %xmm4
; X32-SSE-NEXT:    pcmpgtb %xmm4, %xmm2
; X32-SSE-NEXT:    movdqa %xmm2, %xmm4
; X32-SSE-NEXT:    pandn %xmm0, %xmm4
; X32-SSE-NEXT:    psrlw $1, %xmm0
; X32-SSE-NEXT:    pand .LCPI7_6, %xmm0
; X32-SSE-NEXT:    pand %xmm2, %xmm0
; X32-SSE-NEXT:    por %xmm4, %xmm0
; X32-SSE-NEXT:    por %xmm3, %xmm0
; X32-SSE-NEXT:    por %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <16 x i8> %a, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1>
  %lshr = lshr <16 x i8> %a, <i8 8, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>
  %or = or <16 x i8> %shl, %lshr
  ret <16 x i8> %or
}

;
; Uniform Constant Rotates
;

define <2 x i64> @splatconstant_rotate_v2i64(<2 x i64> %a) nounwind {
; SSE-LABEL: splatconstant_rotate_v2i64:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psllq $14, %xmm1
; SSE-NEXT:    psrlq $50, %xmm0
; SSE-NEXT:    por %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: splatconstant_rotate_v2i64:
; AVX:       # BB#0:
; AVX-NEXT:    vpsllq $14, %xmm0, %xmm1
; AVX-NEXT:    vpsrlq $50, %xmm0, %xmm0
; AVX-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
;
; XOP-LABEL: splatconstant_rotate_v2i64:
; XOP:       # BB#0:
; XOP-NEXT:    vprotq $14, %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: splatconstant_rotate_v2i64:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    psllq $14, %xmm1
; X32-SSE-NEXT:    psrlq $50, %xmm0
; X32-SSE-NEXT:    por %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <2 x i64> %a, <i64 14, i64 14>
  %lshr = lshr <2 x i64> %a, <i64 50, i64 50>
  %or = or <2 x i64> %shl, %lshr
  ret <2 x i64> %or
}

define <4 x i32> @splatconstant_rotate_v4i32(<4 x i32> %a) nounwind {
; SSE-LABEL: splatconstant_rotate_v4i32:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    pslld $4, %xmm1
; SSE-NEXT:    psrld $28, %xmm0
; SSE-NEXT:    por %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: splatconstant_rotate_v4i32:
; AVX:       # BB#0:
; AVX-NEXT:    vpslld $4, %xmm0, %xmm1
; AVX-NEXT:    vpsrld $28, %xmm0, %xmm0
; AVX-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
;
; XOP-LABEL: splatconstant_rotate_v4i32:
; XOP:       # BB#0:
; XOP-NEXT:    vprotd $4, %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: splatconstant_rotate_v4i32:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    pslld $4, %xmm1
; X32-SSE-NEXT:    psrld $28, %xmm0
; X32-SSE-NEXT:    por %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <4 x i32> %a, <i32 4, i32 4, i32 4, i32 4>
  %lshr = lshr <4 x i32> %a, <i32 28, i32 28, i32 28, i32 28>
  %or = or <4 x i32> %shl, %lshr
  ret <4 x i32> %or
}

define <8 x i16> @splatconstant_rotate_v8i16(<8 x i16> %a) nounwind {
; SSE-LABEL: splatconstant_rotate_v8i16:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psllw $7, %xmm1
; SSE-NEXT:    psrlw $9, %xmm0
; SSE-NEXT:    por %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: splatconstant_rotate_v8i16:
; AVX:       # BB#0:
; AVX-NEXT:    vpsllw $7, %xmm0, %xmm1
; AVX-NEXT:    vpsrlw $9, %xmm0, %xmm0
; AVX-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
;
; XOP-LABEL: splatconstant_rotate_v8i16:
; XOP:       # BB#0:
; XOP-NEXT:    vprotw $7, %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: splatconstant_rotate_v8i16:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    psllw $7, %xmm1
; X32-SSE-NEXT:    psrlw $9, %xmm0
; X32-SSE-NEXT:    por %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <8 x i16> %a, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  %lshr = lshr <8 x i16> %a, <i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9>
  %or = or <8 x i16> %shl, %lshr
  ret <8 x i16> %or
}

define <16 x i8> @splatconstant_rotate_v16i8(<16 x i8> %a) nounwind {
; SSE-LABEL: splatconstant_rotate_v16i8:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psllw $4, %xmm1
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    psrlw $4, %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE-NEXT:    por %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: splatconstant_rotate_v16i8:
; AVX:       # BB#0:
; AVX-NEXT:    vpsllw $4, %xmm0, %xmm1
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
;
; XOP-LABEL: splatconstant_rotate_v16i8:
; XOP:       # BB#0:
; XOP-NEXT:    vprotb $4, %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: splatconstant_rotate_v16i8:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    psllw $4, %xmm1
; X32-SSE-NEXT:    pand .LCPI11_0, %xmm1
; X32-SSE-NEXT:    psrlw $4, %xmm0
; X32-SSE-NEXT:    pand .LCPI11_1, %xmm0
; X32-SSE-NEXT:    por %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <16 x i8> %a, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %lshr = lshr <16 x i8> %a, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %or = or <16 x i8> %shl, %lshr
  ret <16 x i8> %or
}

;
; Masked Uniform Constant Rotates
;

define <2 x i64> @splatconstant_rotate_mask_v2i64(<2 x i64> %a) nounwind {
; SSE-LABEL: splatconstant_rotate_mask_v2i64:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psllq $15, %xmm1
; SSE-NEXT:    psrlq $49, %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    por %xmm0, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: splatconstant_rotate_mask_v2i64:
; AVX:       # BB#0:
; AVX-NEXT:    vpsllq $15, %xmm0, %xmm1
; AVX-NEXT:    vpsrlq $49, %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
;
; XOP-LABEL: splatconstant_rotate_mask_v2i64:
; XOP:       # BB#0:
; XOP-NEXT:    vprotq $15, %xmm0, %xmm0
; XOP-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: splatconstant_rotate_mask_v2i64:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    psllq $15, %xmm1
; X32-SSE-NEXT:    psrlq $49, %xmm0
; X32-SSE-NEXT:    pand .LCPI12_0, %xmm0
; X32-SSE-NEXT:    pand .LCPI12_1, %xmm1
; X32-SSE-NEXT:    por %xmm0, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <2 x i64> %a, <i64 15, i64 15>
  %lshr = lshr <2 x i64> %a, <i64 49, i64 49>
  %rmask = and <2 x i64> %lshr, <i64 255, i64 127>
  %lmask = and <2 x i64> %shl, <i64 65, i64 33>
  %or = or <2 x i64> %lmask, %rmask
  ret <2 x i64> %or
}

define <4 x i32> @splatconstant_rotate_mask_v4i32(<4 x i32> %a) nounwind {
; SSE-LABEL: splatconstant_rotate_mask_v4i32:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    pslld $4, %xmm1
; SSE-NEXT:    psrld $28, %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    por %xmm0, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: splatconstant_rotate_mask_v4i32:
; AVX:       # BB#0:
; AVX-NEXT:    vpslld $4, %xmm0, %xmm1
; AVX-NEXT:    vpsrld $28, %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
;
; XOP-LABEL: splatconstant_rotate_mask_v4i32:
; XOP:       # BB#0:
; XOP-NEXT:    vprotd $4, %xmm0, %xmm0
; XOP-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: splatconstant_rotate_mask_v4i32:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    pslld $4, %xmm1
; X32-SSE-NEXT:    psrld $28, %xmm0
; X32-SSE-NEXT:    pand .LCPI13_0, %xmm0
; X32-SSE-NEXT:    pand .LCPI13_1, %xmm1
; X32-SSE-NEXT:    por %xmm0, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <4 x i32> %a, <i32 4, i32 4, i32 4, i32 4>
  %lshr = lshr <4 x i32> %a, <i32 28, i32 28, i32 28, i32 28>
  %rmask = and <4 x i32> %lshr, <i32 127, i32 255, i32 511, i32 1023>
  %lmask = and <4 x i32> %shl, <i32 1023, i32 511, i32 255, i32 127>
  %or = or <4 x i32> %lmask, %rmask
  ret <4 x i32> %or
}

define <8 x i16> @splatconstant_rotate_mask_v8i16(<8 x i16> %a) nounwind {
; SSE-LABEL: splatconstant_rotate_mask_v8i16:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psllw $5, %xmm1
; SSE-NEXT:    psrlw $11, %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    por %xmm0, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: splatconstant_rotate_mask_v8i16:
; AVX:       # BB#0:
; AVX-NEXT:    vpsllw $5, %xmm0, %xmm1
; AVX-NEXT:    vpsrlw $11, %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
;
; XOP-LABEL: splatconstant_rotate_mask_v8i16:
; XOP:       # BB#0:
; XOP-NEXT:    vprotw $5, %xmm0, %xmm0
; XOP-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: splatconstant_rotate_mask_v8i16:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    psllw $5, %xmm1
; X32-SSE-NEXT:    psrlw $11, %xmm0
; X32-SSE-NEXT:    pand .LCPI14_0, %xmm0
; X32-SSE-NEXT:    pand .LCPI14_1, %xmm1
; X32-SSE-NEXT:    por %xmm0, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <8 x i16> %a, <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  %lshr = lshr <8 x i16> %a, <i16 11, i16 11, i16 11, i16 11, i16 11, i16 11, i16 11, i16 11>
  %rmask = and <8 x i16> %lshr, <i16 55, i16 55, i16 55, i16 55, i16 55, i16 55, i16 55, i16 55>
  %lmask = and <8 x i16> %shl, <i16 33, i16 33, i16 33, i16 33, i16 33, i16 33, i16 33, i16 33>
  %or = or <8 x i16> %lmask, %rmask
  ret <8 x i16> %or
}

define <16 x i8> @splatconstant_rotate_mask_v16i8(<16 x i8> %a) nounwind {
; SSE-LABEL: splatconstant_rotate_mask_v16i8:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psllw $4, %xmm1
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    psrlw $4, %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    por %xmm0, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: splatconstant_rotate_mask_v16i8:
; AVX:       # BB#0:
; AVX-NEXT:    vpsllw $4, %xmm0, %xmm1
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vpsrlw $4, %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
;
; XOP-LABEL: splatconstant_rotate_mask_v16i8:
; XOP:       # BB#0:
; XOP-NEXT:    vprotb $4, %xmm0, %xmm0
; XOP-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; XOP-NEXT:    retq
;
; X32-SSE-LABEL: splatconstant_rotate_mask_v16i8:
; X32-SSE:       # BB#0:
; X32-SSE-NEXT:    movdqa %xmm0, %xmm1
; X32-SSE-NEXT:    psllw $4, %xmm1
; X32-SSE-NEXT:    pand .LCPI15_0, %xmm1
; X32-SSE-NEXT:    psrlw $4, %xmm0
; X32-SSE-NEXT:    pand .LCPI15_1, %xmm0
; X32-SSE-NEXT:    pand .LCPI15_2, %xmm0
; X32-SSE-NEXT:    pand .LCPI15_3, %xmm1
; X32-SSE-NEXT:    por %xmm0, %xmm1
; X32-SSE-NEXT:    movdqa %xmm1, %xmm0
; X32-SSE-NEXT:    retl
  %shl = shl <16 x i8> %a, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %lshr = lshr <16 x i8> %a, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %rmask = and <16 x i8> %lshr, <i8 55, i8 55, i8 55, i8 55, i8 55, i8 55, i8 55, i8 55, i8 55, i8 55, i8 55, i8 55, i8 55, i8 55, i8 55, i8 55>
  %lmask = and <16 x i8> %shl, <i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33, i8 33>
  %or = or <16 x i8> %lmask, %rmask
  ret <16 x i8> %or
}
