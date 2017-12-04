; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=CST --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+sse4.1 \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=CST --check-prefix=SSE --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=CST --check-prefix=AVX
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx2 \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=AVX2
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx512f \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512F
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx512vl \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512VL

; SSE2: [[MASKCSTADDR:.LCPI[0-9_]+]]:
; SSE2-NEXT: .long 65535 # 0xffff
; SSE2-NEXT: .long 65535 # 0xffff
; SSE2-NEXT: .long 65535 # 0xffff
; SSE2-NEXT: .long 65535 # 0xffff

; CST: [[FPMASKCSTADDR:.LCPI[0-9_]+]]:
; CST-NEXT: .long 1199570944 # float 65536
; CST-NEXT: .long 1199570944 # float 65536
; CST-NEXT: .long 1199570944 # float 65536
; CST-NEXT: .long 1199570944 # float 65536

; AVX2: [[FPMASKCSTADDR:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 1199570944 # float 65536

define <4 x float> @test_uitofp_v4i32_to_v4f32(<4 x i32> %arg) {
; SSE2-LABEL: test_uitofp_v4i32_to_v4f32:
; SSE2:       # %bb.0:
; SSE2-NEXT:    movaps {{.*#+}} xmm1 = [65535,65535,65535,65535]
; SSE2-NEXT:    andps %xmm0, %xmm1
; SSE2-NEXT:    cvtdq2ps %xmm1, %xmm1
; SSE2-NEXT:    psrld $16, %xmm0
; SSE2-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE2-NEXT:    mulps [[FPMASKCSTADDR]](%rip), %xmm0
; SSE2-NEXT:    addps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: test_uitofp_v4i32_to_v4f32:
; SSE41:       # %bb.0:
; SSE41-NEXT:    pxor %xmm1, %xmm1
; SSE41-NEXT:    pblendw {{.*#+}} xmm1 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; SSE41-NEXT:    cvtdq2ps %xmm1, %xmm1
; SSE41-NEXT:    psrld $16, %xmm0
; SSE41-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE41-NEXT:    mulps [[FPMASKCSTADDR]](%rip), %xmm0
; SSE41-NEXT:    addps %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX:       # %bb.0:
; AVX-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX-NEXT:    vpblendw {{.*#+}} xmm1 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX-NEXT:    vcvtdq2ps %xmm1, %xmm1
; AVX-NEXT:    vpsrld $16, %xmm0, %xmm0
; AVX-NEXT:    vcvtdq2ps %xmm0, %xmm0
; AVX-NEXT:    vmulps [[FPMASKCSTADDR]](%rip), %xmm0, %xmm0
; AVX-NEXT:    vaddps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
;
; AVX2-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vpsrld $16, %xmm0, %xmm1
; AVX2-NEXT:    vcvtdq2ps %xmm1, %xmm1
; AVX2-NEXT:    vbroadcastss [[FPMASKCSTADDR]](%rip), %xmm2
; AVX2-NEXT:    vmulps %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vxorps %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3],xmm0[4],xmm2[5],xmm0[6],xmm2[7]
; AVX2-NEXT:    vcvtdq2ps %xmm0, %xmm0
; AVX2-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    # kill
; AVX512F-NEXT:    vcvtudq2ps %zmm0, %zmm0
; AVX512F-NEXT:    # kill
; AVX512F-NEXT:    vzeroupper
; AVX512F-NEXT:    retq
;
; AVX512VL-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX512VL:       # %bb.0:
; AVX512VL-NEXT:    vcvtudq2ps %xmm0, %xmm0
; AVX512VL-NEXT:    retq
  %tmp = uitofp <4 x i32> %arg to <4 x float>
  ret <4 x float> %tmp
}

; AVX: [[FPMASKCSTADDR_v8:.LCPI[0-9_]+]]:
; AVX-NEXT: .long 1199570944 # float 65536
; AVX-NEXT: .long 1199570944 # float 65536
; AVX-NEXT: .long 1199570944 # float 65536
; AVX-NEXT: .long 1199570944 # float 65536

; AVX: [[MASKCSTADDR_v8:.LCPI[0-9_]+]]:
; AVX-NEXT: .long 65535 # 0xffff
; AVX-NEXT: .long 65535 # 0xffff
; AVX-NEXT: .long 65535 # 0xffff
; AVX-NEXT: .long 65535 # 0xffff

; AVX2: [[FPMASKCSTADDR_v8:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 1199570944 # float 65536

define <8 x float> @test_uitofp_v8i32_to_v8f32(<8 x i32> %arg) {
; SSE2-LABEL: test_uitofp_v8i32_to_v8f32:
; SSE2:       # %bb.0:
; SSE2-NEXT:    movdqa %xmm0, %xmm2
; SSE2-NEXT:    psrld $16, %xmm2
; SSE2-NEXT:    cvtdq2ps %xmm2, %xmm2
; SSE2-NEXT:    movaps {{.*#+}} xmm3 = [6.553600e+04,6.553600e+04,6.553600e+04,6.553600e+04]
; SSE2-NEXT:    mulps %xmm3, %xmm2
; SSE2-NEXT:    movdqa {{.*#+}} xmm4 = [65535,65535,65535,65535]
; SSE2-NEXT:    pand %xmm4, %xmm0
; SSE2-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE2-NEXT:    addps %xmm2, %xmm0
; SSE2-NEXT:    movdqa %xmm1, %xmm2
; SSE2-NEXT:    psrld $16, %xmm2
; SSE2-NEXT:    cvtdq2ps %xmm2, %xmm2
; SSE2-NEXT:    mulps %xmm3, %xmm2
; SSE2-NEXT:    pand %xmm4, %xmm1
; SSE2-NEXT:    cvtdq2ps %xmm1, %xmm1
; SSE2-NEXT:    addps %xmm2, %xmm1
; SSE2-NEXT:    retq
;
; SSE41-LABEL: test_uitofp_v8i32_to_v8f32:
; SSE41:       # %bb.0:
; SSE41-NEXT:    movdqa %xmm0, %xmm2
; SSE41-NEXT:    psrld $16, %xmm2
; SSE41-NEXT:    cvtdq2ps %xmm2, %xmm2
; SSE41-NEXT:    movaps {{.*#+}} xmm3 = [6.553600e+04,6.553600e+04,6.553600e+04,6.553600e+04]
; SSE41-NEXT:    mulps %xmm3, %xmm2
; SSE41-NEXT:    pxor %xmm4, %xmm4
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0],xmm4[1],xmm0[2],xmm4[3],xmm0[4],xmm4[5],xmm0[6],xmm4[7]
; SSE41-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE41-NEXT:    addps %xmm2, %xmm0
; SSE41-NEXT:    movdqa %xmm1, %xmm2
; SSE41-NEXT:    psrld $16, %xmm2
; SSE41-NEXT:    cvtdq2ps %xmm2, %xmm2
; SSE41-NEXT:    mulps %xmm3, %xmm2
; SSE41-NEXT:    pblendw {{.*#+}} xmm1 = xmm1[0],xmm4[1],xmm1[2],xmm4[3],xmm1[4],xmm4[5],xmm1[6],xmm4[7]
; SSE41-NEXT:    cvtdq2ps %xmm1, %xmm1
; SSE41-NEXT:    addps %xmm2, %xmm1
; SSE41-NEXT:    retq
;
; AVX-LABEL: test_uitofp_v8i32_to_v8f32:
; AVX:       # %bb.0:
; AVX-NEXT:    vpsrld $16, %xmm0, %xmm1
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX-NEXT:    vpsrld $16, %xmm2, %xmm2
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; AVX-NEXT:    vcvtdq2ps %ymm1, %ymm1
; AVX-NEXT:    vmulps [[FPMASKCSTADDR_v8]](%rip), %ymm1, %ymm1
; AVX-NEXT:    vandps [[MASKCSTADDR_v8]](%rip), %ymm0, %ymm0
; AVX-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; AVX-NEXT:    retq
;
; AVX2-LABEL: test_uitofp_v8i32_to_v8f32:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vpsrld $16, %ymm0, %ymm1
; AVX2-NEXT:    vcvtdq2ps %ymm1, %ymm1
; AVX2-NEXT:    vbroadcastss [[FPMASKCSTADDR_v8]](%rip), %ymm2
; AVX2-NEXT:    vmulps %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vxorps %xmm2, %xmm2, %xmm2
; AVX2-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm2[1],ymm0[2],ymm2[3],ymm0[4],ymm2[5],ymm0[6],ymm2[7],ymm0[8],ymm2[9],ymm0[10],ymm2[11],ymm0[12],ymm2[13],ymm0[14],ymm2[15]
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: test_uitofp_v8i32_to_v8f32:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    # kill
; AVX512F-NEXT:    vcvtudq2ps %zmm0, %zmm0
; AVX512F-NEXT:    # kill
; AVX512F-NEXT:    retq
;
; AVX512VL-LABEL: test_uitofp_v8i32_to_v8f32:
; AVX512VL:       # %bb.0:
; AVX512VL-NEXT:    vcvtudq2ps %ymm0, %ymm0
; AVX512VL-NEXT:    retq
  %tmp = uitofp <8 x i32> %arg to <8 x float>
  ret <8 x float> %tmp
}
