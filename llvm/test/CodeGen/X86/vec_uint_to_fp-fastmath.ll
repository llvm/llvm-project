; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=SSE --check-prefix=CST
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+sse4.1 \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=SSE --check-prefix=CST
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=AVX --check-prefix=CST
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx2 \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=AVX2
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx512f \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512F
; RUN: llc < %s -mtriple=x86_64 -enable-unsafe-fp-math -mattr=+avx512vl \
; RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512VL

; CST: [[MASKCSTADDR:.LCPI[0-9_]+]]:
; CST-NEXT: .long 65535 # 0xffff
; CST-NEXT: .long 65535 # 0xffff
; CST-NEXT: .long 65535 # 0xffff
; CST-NEXT: .long 65535 # 0xffff

; CST: [[FPMASKCSTADDR:.LCPI[0-9_]+]]:
; CST-NEXT: .long 1199570944 # float 65536
; CST-NEXT: .long 1199570944 # float 65536
; CST-NEXT: .long 1199570944 # float 65536
; CST-NEXT: .long 1199570944 # float 65536

; AVX2: [[FPMASKCSTADDR:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 1199570944 # float 65536

; AVX2: [[MASKCSTADDR:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 65535 # 0xffff

define <4 x float> @test_uitofp_v4i32_to_v4f32(<4 x i32> %arg) {
; SSE-LABEL: test_uitofp_v4i32_to_v4f32:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm1 = [65535,65535,65535,65535]
; SSE-NEXT:    andps %xmm0, %xmm1
; SSE-NEXT:    cvtdq2ps %xmm1, %xmm1
; SSE-NEXT:    psrld $16, %xmm0
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    mulps [[FPMASKCSTADDR]](%rip), %xmm0
; SSE-NEXT:    addps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX:       # BB#0:
; AVX-NEXT:    vandps [[MASKCSTADDR]](%rip), %xmm0, %xmm1
; AVX-NEXT:    vcvtdq2ps %xmm1, %xmm1
; AVX-NEXT:    vpsrld $16, %xmm0, %xmm0
; AVX-NEXT:    vcvtdq2ps %xmm0, %xmm0
; AVX-NEXT:    vmulps [[FPMASKCSTADDR]](%rip), %xmm0, %xmm0
; AVX-NEXT:    vaddps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
;
; AVX2-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsrld $16, %xmm0, %xmm1
; AVX2-NEXT:    vcvtdq2ps %xmm1, %xmm1
; AVX2-NEXT:    vbroadcastss [[FPMASKCSTADDR]](%rip), %xmm2
; AVX2-NEXT:    vmulps %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpbroadcastd [[MASKCSTADDR]](%rip), %xmm2
; AVX2-NEXT:    vpand %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vcvtdq2ps %xmm0, %xmm0
; AVX2-NEXT:    vaddps %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vcvtudq2ps %zmm0, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512VL-LABEL: test_uitofp_v4i32_to_v4f32:
; AVX512VL:       # BB#0:
; AVX512VL-NEXT:    vcvtudq2ps %xmm0, %xmm0
; AVX512VL-NEXT:    retq
  %tmp = uitofp <4 x i32> %arg to <4 x float>
  ret <4 x float> %tmp
}

; AVX: [[MASKCSTADDR_v8:.LCPI[0-9_]+]]:
; AVX-NEXT: .long 65535 # 0xffff
; AVX-NEXT: .long 65535 # 0xffff
; AVX-NEXT: .long 65535 # 0xffff
; AVX-NEXT: .long 65535 # 0xffff

; AVX: [[FPMASKCSTADDR_v8:.LCPI[0-9_]+]]:
; AVX-NEXT: .long 1199570944 # float 65536
; AVX-NEXT: .long 1199570944 # float 65536
; AVX-NEXT: .long 1199570944 # float 65536
; AVX-NEXT: .long 1199570944 # float 65536

; AVX2: [[FPMASKCSTADDR_v8:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 1199570944 # float 65536

; AVX2: [[MASKCSTADDR_v8:.LCPI[0-9_]+]]:
; AVX2-NEXT: .long 65535 # 0xffff

define <8 x float> @test_uitofp_v8i32_to_v8f32(<8 x i32> %arg) {
; SSE-LABEL: test_uitofp_v8i32_to_v8f32:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    psrld $16, %xmm2
; SSE-NEXT:    cvtdq2ps %xmm2, %xmm2
; SSE-NEXT:    movaps {{.*#+}} xmm3 = [6.553600e+04,6.553600e+04,6.553600e+04,6.553600e+04]
; SSE-NEXT:    mulps %xmm3, %xmm2
; SSE-NEXT:    movdqa {{.*#+}} xmm4 = [65535,65535,65535,65535]
; SSE-NEXT:    pand %xmm4, %xmm0
; SSE-NEXT:    cvtdq2ps %xmm0, %xmm0
; SSE-NEXT:    addps %xmm2, %xmm0
; SSE-NEXT:    movdqa %xmm1, %xmm2
; SSE-NEXT:    psrld $16, %xmm2
; SSE-NEXT:    cvtdq2ps %xmm2, %xmm2
; SSE-NEXT:    mulps %xmm3, %xmm2
; SSE-NEXT:    pand %xmm4, %xmm1
; SSE-NEXT:    cvtdq2ps %xmm1, %xmm1
; SSE-NEXT:    addps %xmm2, %xmm1
; SSE-NEXT:    retq
;
; AVX-LABEL: test_uitofp_v8i32_to_v8f32:
; AVX:       # BB#0:
; AVX-NEXT:    vandps [[MASKCSTADDR_v8]](%rip), %ymm0, %ymm1
; AVX-NEXT:    vcvtdq2ps %ymm1, %ymm1
; AVX-NEXT:    vpsrld $16, %xmm0, %xmm2
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX-NEXT:    vpsrld $16, %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; AVX-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX-NEXT:    vmulps [[FPMASKCSTADDR_v8]](%rip), %ymm0, %ymm0
; AVX-NEXT:    vaddps %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
;
; AVX2-LABEL: test_uitofp_v8i32_to_v8f32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsrld $16, %ymm0, %ymm1
; AVX2-NEXT:    vcvtdq2ps %ymm1, %ymm1
; AVX2-NEXT:    vbroadcastss [[FPMASKCSTADDR_v8]](%rip), %ymm2
; AVX2-NEXT:    vmulps %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpbroadcastd [[MASKCSTADDR_v8]](%rip), %ymm2
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vcvtdq2ps %ymm0, %ymm0
; AVX2-NEXT:    vaddps %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: test_uitofp_v8i32_to_v8f32:
; AVX512F:       # BB#0:
; AVX512F-NEXT:    vcvtudq2ps %zmm0, %zmm0
; AVX512F-NEXT:    retq
;
; AVX512VL-LABEL: test_uitofp_v8i32_to_v8f32:
; AVX512VL:       # BB#0:
; AVX512VL-NEXT:    vcvtudq2ps %ymm0, %ymm0
; AVX512VL-NEXT:    retq
  %tmp = uitofp <8 x i32> %arg to <8 x float>
  ret <8 x float> %tmp
}
