; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512cd -mattr=+avx512vl| FileCheck %s --check-prefix=AVX512VLCD --check-prefix=ALL --check-prefix=AVX512
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512cd | FileCheck %s --check-prefix=AVX512CD --check-prefix=ALL --check-prefix=AVX512

define <4 x i64> @testv4i64(<4 x i64> %in) nounwind {
; AVX1-LABEL: testv4i64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrq $1, %xmm1, %rax
; AVX1-NEXT:    bsrq %rax, %rax
; AVX1-NEXT:    movl $127, %ecx
; AVX1-NEXT:    cmoveq %rcx, %rax
; AVX1-NEXT:    xorq $63, %rax
; AVX1-NEXT:    vmovq %rax, %xmm2
; AVX1-NEXT:    vmovq %xmm1, %rax
; AVX1-NEXT:    bsrq %rax, %rax
; AVX1-NEXT:    cmoveq %rcx, %rax
; AVX1-NEXT:    xorq $63, %rax
; AVX1-NEXT:    vmovq %rax, %xmm1
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX1-NEXT:    vpextrq $1, %xmm0, %rax
; AVX1-NEXT:    bsrq %rax, %rax
; AVX1-NEXT:    cmoveq %rcx, %rax
; AVX1-NEXT:    xorq $63, %rax
; AVX1-NEXT:    vmovq %rax, %xmm2
; AVX1-NEXT:    vmovq %xmm0, %rax
; AVX1-NEXT:    bsrq %rax, %rax
; AVX1-NEXT:    cmoveq %rcx, %rax
; AVX1-NEXT:    xorq $63, %rax
; AVX1-NEXT:    vmovq %rax, %xmm0
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv4i64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrq $1, %xmm1, %rax
; AVX2-NEXT:    bsrq %rax, %rax
; AVX2-NEXT:    movl $127, %ecx
; AVX2-NEXT:    cmoveq %rcx, %rax
; AVX2-NEXT:    xorq $63, %rax
; AVX2-NEXT:    vmovq %rax, %xmm2
; AVX2-NEXT:    vmovq %xmm1, %rax
; AVX2-NEXT:    bsrq %rax, %rax
; AVX2-NEXT:    cmoveq %rcx, %rax
; AVX2-NEXT:    xorq $63, %rax
; AVX2-NEXT:    vmovq %rax, %xmm1
; AVX2-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX2-NEXT:    vpextrq $1, %xmm0, %rax
; AVX2-NEXT:    bsrq %rax, %rax
; AVX2-NEXT:    cmoveq %rcx, %rax
; AVX2-NEXT:    xorq $63, %rax
; AVX2-NEXT:    vmovq %rax, %xmm2
; AVX2-NEXT:    vmovq %xmm0, %rax
; AVX2-NEXT:    bsrq %rax, %rax
; AVX2-NEXT:    cmoveq %rcx, %rax
; AVX2-NEXT:    xorq $63, %rax
; AVX2-NEXT:    vmovq %rax, %xmm0
; AVX2-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512VLCD-LABEL: testv4i64:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vplzcntq %ymm0, %ymm0
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: testv4i64:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vplzcntq %zmm0, %zmm0
; AVX512CD-NEXT:    retq

  %out = call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %in, i1 0)
  ret <4 x i64> %out
}

define <4 x i64> @testv4i64u(<4 x i64> %in) nounwind {
; AVX1-LABEL: testv4i64u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrq $1, %xmm1, %rax
; AVX1-NEXT:    bsrq %rax, %rax
; AVX1-NEXT:    xorq $63, %rax
; AVX1-NEXT:    vmovq %rax, %xmm2
; AVX1-NEXT:    vmovq %xmm1, %rax
; AVX1-NEXT:    bsrq %rax, %rax
; AVX1-NEXT:    xorq $63, %rax
; AVX1-NEXT:    vmovq %rax, %xmm1
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX1-NEXT:    vpextrq $1, %xmm0, %rax
; AVX1-NEXT:    bsrq %rax, %rax
; AVX1-NEXT:    xorq $63, %rax
; AVX1-NEXT:    vmovq %rax, %xmm2
; AVX1-NEXT:    vmovq %xmm0, %rax
; AVX1-NEXT:    bsrq %rax, %rax
; AVX1-NEXT:    xorq $63, %rax
; AVX1-NEXT:    vmovq %rax, %xmm0
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv4i64u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrq $1, %xmm1, %rax
; AVX2-NEXT:    bsrq %rax, %rax
; AVX2-NEXT:    xorq $63, %rax
; AVX2-NEXT:    vmovq %rax, %xmm2
; AVX2-NEXT:    vmovq %xmm1, %rax
; AVX2-NEXT:    bsrq %rax, %rax
; AVX2-NEXT:    xorq $63, %rax
; AVX2-NEXT:    vmovq %rax, %xmm1
; AVX2-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX2-NEXT:    vpextrq $1, %xmm0, %rax
; AVX2-NEXT:    bsrq %rax, %rax
; AVX2-NEXT:    xorq $63, %rax
; AVX2-NEXT:    vmovq %rax, %xmm2
; AVX2-NEXT:    vmovq %xmm0, %rax
; AVX2-NEXT:    bsrq %rax, %rax
; AVX2-NEXT:    xorq $63, %rax
; AVX2-NEXT:    vmovq %rax, %xmm0
; AVX2-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512VLCD-LABEL: testv4i64u:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vplzcntq %ymm0, %ymm0
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: testv4i64u:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vplzcntq %zmm0, %zmm0
; AVX512CD-NEXT:    retq

  %out = call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %in, i1 -1)
  ret <4 x i64> %out
}

define <8 x i32> @testv8i32(<8 x i32> %in) nounwind {
; AVX1-LABEL: testv8i32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrd $1, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %ecx
; AVX1-NEXT:    movl $63, %eax
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $31, %ecx
; AVX1-NEXT:    vmovd %xmm1, %edx
; AVX1-NEXT:    bsrl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    xorl $31, %edx
; AVX1-NEXT:    vmovd %edx, %xmm2
; AVX1-NEXT:    vpinsrd $1, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $2, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $31, %ecx
; AVX1-NEXT:    vpinsrd $2, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $3, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $31, %ecx
; AVX1-NEXT:    vpinsrd $3, %ecx, %xmm2, %xmm1
; AVX1-NEXT:    vpextrd $1, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $31, %ecx
; AVX1-NEXT:    vmovd %xmm0, %edx
; AVX1-NEXT:    bsrl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    xorl $31, %edx
; AVX1-NEXT:    vmovd %edx, %xmm2
; AVX1-NEXT:    vpinsrd $1, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $2, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $31, %ecx
; AVX1-NEXT:    vpinsrd $2, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $3, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $31, %ecx
; AVX1-NEXT:    vpinsrd $3, %ecx, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv8i32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrd $1, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %ecx
; AVX2-NEXT:    movl $63, %eax
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $31, %ecx
; AVX2-NEXT:    vmovd %xmm1, %edx
; AVX2-NEXT:    bsrl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    xorl $31, %edx
; AVX2-NEXT:    vmovd %edx, %xmm2
; AVX2-NEXT:    vpinsrd $1, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $2, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $31, %ecx
; AVX2-NEXT:    vpinsrd $2, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $3, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $31, %ecx
; AVX2-NEXT:    vpinsrd $3, %ecx, %xmm2, %xmm1
; AVX2-NEXT:    vpextrd $1, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $31, %ecx
; AVX2-NEXT:    vmovd %xmm0, %edx
; AVX2-NEXT:    bsrl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    xorl $31, %edx
; AVX2-NEXT:    vmovd %edx, %xmm2
; AVX2-NEXT:    vpinsrd $1, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $2, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $31, %ecx
; AVX2-NEXT:    vpinsrd $2, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $3, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $31, %ecx
; AVX2-NEXT:    vpinsrd $3, %ecx, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512VLCD-LABEL: testv8i32:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vplzcntd %ymm0, %ymm0
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: testv8i32:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512CD-NEXT:    retq

  %out = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %in, i1 0)
  ret <8 x i32> %out
}

define <8 x i32> @testv8i32u(<8 x i32> %in) nounwind {
; AVX1-LABEL: testv8i32u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrd $1, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $31, %eax
; AVX1-NEXT:    vmovd %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    xorl $31, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $2, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $31, %eax
; AVX1-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $3, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $31, %eax
; AVX1-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm1
; AVX1-NEXT:    vpextrd $1, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $31, %eax
; AVX1-NEXT:    vmovd %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    xorl $31, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $2, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $31, %eax
; AVX1-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $3, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $31, %eax
; AVX1-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv8i32u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrd $1, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $31, %eax
; AVX2-NEXT:    vmovd %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    xorl $31, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $2, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $31, %eax
; AVX2-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $3, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $31, %eax
; AVX2-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm1
; AVX2-NEXT:    vpextrd $1, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $31, %eax
; AVX2-NEXT:    vmovd %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    xorl $31, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $2, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $31, %eax
; AVX2-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $3, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $31, %eax
; AVX2-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512VLCD-LABEL: testv8i32u:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vplzcntd %ymm0, %ymm0
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: testv8i32u:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512CD-NEXT:    retq

  %out = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %in, i1 -1)
  ret <8 x i32> %out
}

define <16 x i16> @testv16i16(<16 x i16> %in) nounwind {
; AVX1-LABEL: testv16i16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrw $1, %xmm1, %eax
; AVX1-NEXT:    bsrw %ax, %cx
; AVX1-NEXT:    movw $31, %ax
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vmovd %xmm1, %edx
; AVX1-NEXT:    bsrw %dx, %dx
; AVX1-NEXT:    cmovew %ax, %dx
; AVX1-NEXT:    xorl $15, %edx
; AVX1-NEXT:    vmovd %edx, %xmm2
; AVX1-NEXT:    vpinsrw $1, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $2, %xmm1, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $2, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $3, %xmm1, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $3, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $4, %xmm1, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $4, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $5, %xmm1, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $5, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $6, %xmm1, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $6, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $7, %xmm1, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $7, %ecx, %xmm2, %xmm1
; AVX1-NEXT:    vpextrw $1, %xmm0, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vmovd %xmm0, %edx
; AVX1-NEXT:    bsrw %dx, %dx
; AVX1-NEXT:    cmovew %ax, %dx
; AVX1-NEXT:    xorl $15, %edx
; AVX1-NEXT:    vmovd %edx, %xmm2
; AVX1-NEXT:    vpinsrw $1, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $2, %xmm0, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $2, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $3, %xmm0, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $3, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $4, %xmm0, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $4, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $5, %xmm0, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $5, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $6, %xmm0, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $6, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $7, %xmm0, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vpinsrw $7, %ecx, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv16i16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrw $1, %xmm1, %eax
; AVX2-NEXT:    bsrw %ax, %cx
; AVX2-NEXT:    movw $31, %ax
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vmovd %xmm1, %edx
; AVX2-NEXT:    bsrw %dx, %dx
; AVX2-NEXT:    cmovew %ax, %dx
; AVX2-NEXT:    xorl $15, %edx
; AVX2-NEXT:    vmovd %edx, %xmm2
; AVX2-NEXT:    vpinsrw $1, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $2, %xmm1, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $2, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $3, %xmm1, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $3, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $4, %xmm1, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $4, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $5, %xmm1, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $5, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $6, %xmm1, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $6, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $7, %xmm1, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $7, %ecx, %xmm2, %xmm1
; AVX2-NEXT:    vpextrw $1, %xmm0, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vmovd %xmm0, %edx
; AVX2-NEXT:    bsrw %dx, %dx
; AVX2-NEXT:    cmovew %ax, %dx
; AVX2-NEXT:    xorl $15, %edx
; AVX2-NEXT:    vmovd %edx, %xmm2
; AVX2-NEXT:    vpinsrw $1, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $2, %xmm0, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $2, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $3, %xmm0, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $3, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $4, %xmm0, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $4, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $5, %xmm0, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $5, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $6, %xmm0, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $6, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $7, %xmm0, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vpinsrw $7, %ecx, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512VLCD-LABEL: testv16i16:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vpmovzxwd %ymm0, %zmm0
; AVX512VLCD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512VLCD-NEXT:    vpmovdw %zmm0, %ymm0
; AVX512VLCD-NEXT:    vpsubw {{.*}}(%rip), %ymm0, %ymm0
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: testv16i16:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vpmovzxwd %ymm0, %zmm0
; AVX512CD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512CD-NEXT:    vpmovdw %zmm0, %ymm0
; AVX512CD-NEXT:    vpsubw {{.*}}(%rip), %ymm0, %ymm0
; AVX512CD-NEXT:    retq
  %out = call <16 x i16> @llvm.ctlz.v16i16(<16 x i16> %in, i1 0)
  ret <16 x i16> %out
}

define <16 x i16> @testv16i16u(<16 x i16> %in) nounwind {
; AVX1-LABEL: testv16i16u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrw $1, %xmm1, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vmovd %xmm1, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $2, %xmm1, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $3, %xmm1, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $4, %xmm1, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $5, %xmm1, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $6, %xmm1, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $7, %xmm1, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm1
; AVX1-NEXT:    vpextrw $1, %xmm0, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vmovd %xmm0, %ecx
; AVX1-NEXT:    bsrw %cx, %cx
; AVX1-NEXT:    xorl $15, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $2, %xmm0, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $3, %xmm0, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $4, %xmm0, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $5, %xmm0, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $6, %xmm0, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $7, %xmm0, %eax
; AVX1-NEXT:    bsrw %ax, %ax
; AVX1-NEXT:    xorl $15, %eax
; AVX1-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv16i16u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrw $1, %xmm1, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vmovd %xmm1, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $2, %xmm1, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $3, %xmm1, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $4, %xmm1, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $5, %xmm1, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $6, %xmm1, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $7, %xmm1, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm1
; AVX2-NEXT:    vpextrw $1, %xmm0, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vmovd %xmm0, %ecx
; AVX2-NEXT:    bsrw %cx, %cx
; AVX2-NEXT:    xorl $15, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $2, %xmm0, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $3, %xmm0, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $4, %xmm0, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $5, %xmm0, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $6, %xmm0, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $7, %xmm0, %eax
; AVX2-NEXT:    bsrw %ax, %ax
; AVX2-NEXT:    xorl $15, %eax
; AVX2-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512VLCD-LABEL: testv16i16u:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vpmovzxwd %ymm0, %zmm0
; AVX512VLCD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512VLCD-NEXT:    vpmovdw %zmm0, %ymm0
; AVX512VLCD-NEXT:    vpsubw {{.*}}(%rip), %ymm0, %ymm0
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: testv16i16u:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vpmovzxwd %ymm0, %zmm0
; AVX512CD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512CD-NEXT:    vpmovdw %zmm0, %ymm0
; AVX512CD-NEXT:    vpsubw {{.*}}(%rip), %ymm0, %ymm0
; AVX512CD-NEXT:    retq
  %out = call <16 x i16> @llvm.ctlz.v16i16(<16 x i16> %in, i1 -1)
  ret <16 x i16> %out
}

define <32 x i8> @testv32i8(<32 x i8> %in) nounwind {
; AVX1-LABEL: testv32i8:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrb $1, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %ecx
; AVX1-NEXT:    movl $15, %eax
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpextrb $0, %xmm1, %edx
; AVX1-NEXT:    bsrl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    xorl $7, %edx
; AVX1-NEXT:    vmovd %edx, %xmm2
; AVX1-NEXT:    vpinsrb $1, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $2, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $2, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $3, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $3, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $4, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $4, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $5, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $5, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $6, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $6, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $7, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $7, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $8, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $8, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $9, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $9, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $10, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $10, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $11, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $11, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $12, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $12, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $13, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $13, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $14, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $14, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $15, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $15, %ecx, %xmm2, %xmm1
; AVX1-NEXT:    vpextrb $1, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpextrb $0, %xmm0, %edx
; AVX1-NEXT:    bsrl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    xorl $7, %edx
; AVX1-NEXT:    vmovd %edx, %xmm2
; AVX1-NEXT:    vpinsrb $1, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $2, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $2, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $3, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $3, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $4, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $4, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $5, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $5, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $6, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $6, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $7, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $7, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $8, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $8, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $9, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $9, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $10, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $10, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $11, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $11, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $12, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $12, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $13, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $13, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $14, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $14, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $15, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vpinsrb $15, %ecx, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv32i8:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrb $1, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %ecx
; AVX2-NEXT:    movl $15, %eax
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpextrb $0, %xmm1, %edx
; AVX2-NEXT:    bsrl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    xorl $7, %edx
; AVX2-NEXT:    vmovd %edx, %xmm2
; AVX2-NEXT:    vpinsrb $1, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $2, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $2, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $3, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $3, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $4, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $4, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $5, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $5, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $6, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $6, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $7, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $7, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $8, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $8, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $9, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $9, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $10, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $10, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $11, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $11, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $12, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $12, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $13, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $13, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $14, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $14, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $15, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $15, %ecx, %xmm2, %xmm1
; AVX2-NEXT:    vpextrb $1, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpextrb $0, %xmm0, %edx
; AVX2-NEXT:    bsrl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    xorl $7, %edx
; AVX2-NEXT:    vmovd %edx, %xmm2
; AVX2-NEXT:    vpinsrb $1, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $2, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $2, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $3, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $3, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $4, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $4, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $5, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $5, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $6, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $6, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $7, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $7, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $8, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $8, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $9, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $9, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $10, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $10, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $11, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $11, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $12, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $12, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $13, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $13, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $14, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $14, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $15, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vpinsrb $15, %ecx, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512VLCD-LABEL: testv32i8:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX512VLCD-NEXT:    vpmovzxbd %xmm1, %zmm1
; AVX512VLCD-NEXT:    vplzcntd %zmm1, %zmm1
; AVX512VLCD-NEXT:    vpmovdb %zmm1, %xmm1
; AVX512VLCD-NEXT:    vmovdqa64 {{.*#+}} xmm2 = [24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24]
; AVX512VLCD-NEXT:    vpsubb %xmm2, %xmm1, %xmm1
; AVX512VLCD-NEXT:    vpmovzxbd %xmm0, %zmm0
; AVX512VLCD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512VLCD-NEXT:    vpmovdb %zmm0, %xmm0
; AVX512VLCD-NEXT:    vpsubb %xmm2, %xmm0, %xmm0
; AVX512VLCD-NEXT:    vinserti32x4 $1, %xmm1, %ymm0, %ymm0
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: testv32i8:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX512CD-NEXT:    vpmovzxbd %xmm1, %zmm1
; AVX512CD-NEXT:    vplzcntd %zmm1, %zmm1
; AVX512CD-NEXT:    vpmovdb %zmm1, %xmm1
; AVX512CD-NEXT:    vmovdqa {{.*#+}} xmm2 = [24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24]
; AVX512CD-NEXT:    vpsubb %xmm2, %xmm1, %xmm1
; AVX512CD-NEXT:    vpmovzxbd %xmm0, %zmm0
; AVX512CD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512CD-NEXT:    vpmovdb %zmm0, %xmm0
; AVX512CD-NEXT:    vpsubb %xmm2, %xmm0, %xmm0
; AVX512CD-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX512CD-NEXT:    retq
  %out = call <32 x i8> @llvm.ctlz.v32i8(<32 x i8> %in, i1 0)
  ret <32 x i8> %out
}

define <32 x i8> @testv32i8u(<32 x i8> %in) nounwind {
; AVX1-LABEL: testv32i8u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrb $1, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpextrb $0, %xmm1, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $2, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $3, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $4, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $5, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $6, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $7, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $8, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $9, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $10, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $11, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $12, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $13, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $14, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $15, %xmm1, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm1
; AVX1-NEXT:    vpextrb $1, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX1-NEXT:    bsrl %ecx, %ecx
; AVX1-NEXT:    xorl $7, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $2, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $3, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $4, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $5, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $6, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $7, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $8, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $9, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $10, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $11, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $12, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $13, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $14, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $15, %xmm0, %eax
; AVX1-NEXT:    bsrl %eax, %eax
; AVX1-NEXT:    xorl $7, %eax
; AVX1-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv32i8u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrb $1, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpextrb $0, %xmm1, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $2, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $3, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $4, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $5, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $6, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $7, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $8, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $9, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $10, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $11, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $12, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $13, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $14, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $15, %xmm1, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm1
; AVX2-NEXT:    vpextrb $1, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX2-NEXT:    bsrl %ecx, %ecx
; AVX2-NEXT:    xorl $7, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $2, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $3, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $4, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $5, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $6, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $7, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $8, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $9, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $10, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $11, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $12, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $13, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $14, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $15, %xmm0, %eax
; AVX2-NEXT:    bsrl %eax, %eax
; AVX2-NEXT:    xorl $7, %eax
; AVX2-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
;
; AVX512VLCD-LABEL: testv32i8u:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX512VLCD-NEXT:    vpmovzxbd %xmm1, %zmm1
; AVX512VLCD-NEXT:    vplzcntd %zmm1, %zmm1
; AVX512VLCD-NEXT:    vpmovdb %zmm1, %xmm1
; AVX512VLCD-NEXT:    vmovdqa64 {{.*#+}} xmm2 = [24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24]
; AVX512VLCD-NEXT:    vpsubb %xmm2, %xmm1, %xmm1
; AVX512VLCD-NEXT:    vpmovzxbd %xmm0, %zmm0
; AVX512VLCD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512VLCD-NEXT:    vpmovdb %zmm0, %xmm0
; AVX512VLCD-NEXT:    vpsubb %xmm2, %xmm0, %xmm0
; AVX512VLCD-NEXT:    vinserti32x4 $1, %xmm1, %ymm0, %ymm0
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: testv32i8u:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX512CD-NEXT:    vpmovzxbd %xmm1, %zmm1
; AVX512CD-NEXT:    vplzcntd %zmm1, %zmm1
; AVX512CD-NEXT:    vpmovdb %zmm1, %xmm1
; AVX512CD-NEXT:    vmovdqa {{.*#+}} xmm2 = [24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24]
; AVX512CD-NEXT:    vpsubb %xmm2, %xmm1, %xmm1
; AVX512CD-NEXT:    vpmovzxbd %xmm0, %zmm0
; AVX512CD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512CD-NEXT:    vpmovdb %zmm0, %xmm0
; AVX512CD-NEXT:    vpsubb %xmm2, %xmm0, %xmm0
; AVX512CD-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX512CD-NEXT:    retq
  %out = call <32 x i8> @llvm.ctlz.v32i8(<32 x i8> %in, i1 -1)
  ret <32 x i8> %out
}

define <4 x i64> @foldv4i64() nounwind {
; AVX-LABEL: foldv4i64:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [55,0,64,56]
; AVX-NEXT:    retq
;
; AVX512VLCD-LABEL: foldv4i64:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vmovdqa64 {{.*#+}} ymm0 = [55,0,64,56]
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: foldv4i64:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vmovaps {{.*#+}} ymm0 = [55,0,64,56]
; AVX512CD-NEXT:    retq
  %out = call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> <i64 256, i64 -1, i64 0, i64 255>, i1 0)
  ret <4 x i64> %out
}

define <4 x i64> @foldv4i64u() nounwind {
; AVX-LABEL: foldv4i64u:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [55,0,64,56]
; AVX-NEXT:    retq
;
; AVX512VLCD-LABEL: foldv4i64u:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vmovdqa64 {{.*#+}} ymm0 = [55,0,64,56]
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: foldv4i64u:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vmovaps {{.*#+}} ymm0 = [55,0,64,56]
; AVX512CD-NEXT:    retq
  %out = call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> <i64 256, i64 -1, i64 0, i64 255>, i1 -1)
  ret <4 x i64> %out
}

define <8 x i32> @foldv8i32() nounwind {
; AVX-LABEL: foldv8i32:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [23,0,32,24,0,29,27,25]
; AVX-NEXT:    retq
;
; AVX512VLCD-LABEL: foldv8i32:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vmovdqa32 {{.*#+}} ymm0 = [23,0,32,24,0,29,27,25]
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: foldv8i32:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vmovaps {{.*#+}} ymm0 = [23,0,32,24,0,29,27,25]
; AVX512CD-NEXT:    retq
  %out = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> <i32 256, i32 -1, i32 0, i32 255, i32 -65536, i32 7, i32 24, i32 88>, i1 0)
  ret <8 x i32> %out
}

define <8 x i32> @foldv8i32u() nounwind {
; AVX-LABEL: foldv8i32u:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [23,0,32,24,0,29,27,25]
; AVX-NEXT:    retq
;
; AVX512VLCD-LABEL: foldv8i32u:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vmovdqa32 {{.*#+}} ymm0 = [23,0,32,24,0,29,27,25]
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: foldv8i32u:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vmovaps {{.*#+}} ymm0 = [23,0,32,24,0,29,27,25]
; AVX512CD-NEXT:    retq
  %out = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> <i32 256, i32 -1, i32 0, i32 255, i32 -65536, i32 7, i32 24, i32 88>, i1 -1)
  ret <8 x i32> %out
}

define <16 x i16> @foldv16i16() nounwind {
; AVX-LABEL: foldv16i16:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [7,0,16,8,16,13,11,9,0,8,15,14,13,12,11,10]
; AVX-NEXT:    retq
;
; AVX512VLCD-LABEL: foldv16i16:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vmovdqa64 {{.*#+}} ymm0 = [7,0,16,8,16,13,11,9,0,8,15,14,13,12,11,10]
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: foldv16i16:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vmovaps {{.*#+}} ymm0 = [7,0,16,8,16,13,11,9,0,8,15,14,13,12,11,10]
; AVX512CD-NEXT:    retq
  %out = call <16 x i16> @llvm.ctlz.v16i16(<16 x i16> <i16 256, i16 -1, i16 0, i16 255, i16 -65536, i16 7, i16 24, i16 88, i16 -2, i16 254, i16 1, i16 2, i16 4, i16 8, i16 16, i16 32>, i1 0)
  ret <16 x i16> %out
}

define <16 x i16> @foldv16i16u() nounwind {
; AVX-LABEL: foldv16i16u:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [7,0,16,8,16,13,11,9,0,8,15,14,13,12,11,10]
; AVX-NEXT:    retq
;
; AVX512VLCD-LABEL: foldv16i16u:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vmovdqa64 {{.*#+}} ymm0 = [7,0,16,8,16,13,11,9,0,8,15,14,13,12,11,10]
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: foldv16i16u:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vmovaps {{.*#+}} ymm0 = [7,0,16,8,16,13,11,9,0,8,15,14,13,12,11,10]
; AVX512CD-NEXT:    retq
  %out = call <16 x i16> @llvm.ctlz.v16i16(<16 x i16> <i16 256, i16 -1, i16 0, i16 255, i16 -65536, i16 7, i16 24, i16 88, i16 -2, i16 254, i16 1, i16 2, i16 4, i16 8, i16 16, i16 32>, i1 -1)
  ret <16 x i16> %out
}

define <32 x i8> @foldv32i8() nounwind {
; AVX-LABEL: foldv32i8:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,8,0,8,5,3,1,0,0,7,6,5,4,3,2,1,0,8,8,0,0,0,0,0,0,0,0,6,5,5,1]
; AVX-NEXT:    retq
;
; AVX512VLCD-LABEL: foldv32i8:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vmovdqa64 {{.*#+}} ymm0 = [8,0,8,0,8,5,3,1,0,0,7,6,5,4,3,2,1,0,8,8,0,0,0,0,0,0,0,0,6,5,5,1]
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: foldv32i8:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,8,0,8,5,3,1,0,0,7,6,5,4,3,2,1,0,8,8,0,0,0,0,0,0,0,0,6,5,5,1]
; AVX512CD-NEXT:    retq
  %out = call <32 x i8> @llvm.ctlz.v32i8(<32 x i8> <i8 256, i8 -1, i8 0, i8 255, i8 -65536, i8 7, i8 24, i8 88, i8 -2, i8 254, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 128, i8 256, i8 -256, i8 -128, i8 -64, i8 -32, i8 -16, i8 -8, i8 -4, i8 -2, i8 -1, i8 3, i8 5, i8 7, i8 127>, i1 0)
  ret <32 x i8> %out
}

define <32 x i8> @foldv32i8u() nounwind {
; AVX-LABEL: foldv32i8u:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,8,0,8,5,3,1,0,0,7,6,5,4,3,2,1,0,8,8,0,0,0,0,0,0,0,0,6,5,5,1]
; AVX-NEXT:    retq
;
; AVX512VLCD-LABEL: foldv32i8u:
; AVX512VLCD:       ## BB#0:
; AVX512VLCD-NEXT:    vmovdqa64 {{.*#+}} ymm0 = [8,0,8,0,8,5,3,1,0,0,7,6,5,4,3,2,1,0,8,8,0,0,0,0,0,0,0,0,6,5,5,1]
; AVX512VLCD-NEXT:    retq
;
; AVX512CD-LABEL: foldv32i8u:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,8,0,8,5,3,1,0,0,7,6,5,4,3,2,1,0,8,8,0,0,0,0,0,0,0,0,6,5,5,1]
; AVX512CD-NEXT:    retq
  %out = call <32 x i8> @llvm.ctlz.v32i8(<32 x i8> <i8 256, i8 -1, i8 0, i8 255, i8 -65536, i8 7, i8 24, i8 88, i8 -2, i8 254, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 128, i8 256, i8 -256, i8 -128, i8 -64, i8 -32, i8 -16, i8 -8, i8 -4, i8 -2, i8 -1, i8 3, i8 5, i8 7, i8 127>, i1 -1)
  ret <32 x i8> %out
}

declare <4 x i64> @llvm.ctlz.v4i64(<4 x i64>, i1)
declare <8 x i32> @llvm.ctlz.v8i32(<8 x i32>, i1)
declare <16 x i16> @llvm.ctlz.v16i16(<16 x i16>, i1)
declare <32 x i8> @llvm.ctlz.v32i8(<32 x i8>, i1)
