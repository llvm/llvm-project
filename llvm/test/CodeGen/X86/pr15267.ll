; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=corei7-avx | FileCheck %s

define <4 x i3> @test1(<4 x i3>* %in) nounwind {
; CHECK-LABEL: test1:
; CHECK:       # BB#0:
; CHECK-NEXT:    movzwl (%rdi), %eax
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $3, %ecx
; CHECK-NEXT:    andl $7, %ecx
; CHECK-NEXT:    movl %eax, %edx
; CHECK-NEXT:    andl $7, %edx
; CHECK-NEXT:    vmovd %edx, %xmm0
; CHECK-NEXT:    vpinsrd $1, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $6, %ecx
; CHECK-NEXT:    andl $7, %ecx
; CHECK-NEXT:    vpinsrd $2, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    shrl $9, %eax
; CHECK-NEXT:    andl $7, %eax
; CHECK-NEXT:    vpinsrd $3, %eax, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %ret = load <4 x i3>, <4 x i3>* %in, align 1
  ret <4 x i3> %ret
}

define <4 x i1> @test2(<4 x i1>* %in) nounwind {
; CHECK-LABEL: test2:
; CHECK:       # BB#0:
; CHECK-NEXT:    movzbl (%rdi), %eax
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl %ecx
; CHECK-NEXT:    andl $1, %ecx
; CHECK-NEXT:    movl %eax, %edx
; CHECK-NEXT:    andl $1, %edx
; CHECK-NEXT:    vmovd %edx, %xmm0
; CHECK-NEXT:    vpinsrd $1, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $2, %ecx
; CHECK-NEXT:    andl $1, %ecx
; CHECK-NEXT:    vpinsrd $2, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    shrl $3, %eax
; CHECK-NEXT:    andl $1, %eax
; CHECK-NEXT:    vpinsrd $3, %eax, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %ret = load <4 x i1>, <4 x i1>* %in, align 1
  ret <4 x i1> %ret
}

define <4 x i64> @test3(<4 x i1>* %in) nounwind {
; CHECK-LABEL: test3:
; CHECK:       # BB#0:
; CHECK-NEXT:    movzbl (%rdi), %eax
; CHECK-NEXT:    movq %rax, %rcx
; CHECK-NEXT:    shlq $62, %rcx
; CHECK-NEXT:    sarq $63, %rcx
; CHECK-NEXT:    movq %rax, %rdx
; CHECK-NEXT:    shlq $63, %rdx
; CHECK-NEXT:    sarq $63, %rdx
; CHECK-NEXT:    vmovd %edx, %xmm0
; CHECK-NEXT:    vpinsrd $1, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movq %rax, %rcx
; CHECK-NEXT:    shlq $61, %rcx
; CHECK-NEXT:    sarq $63, %rcx
; CHECK-NEXT:    vpinsrd $2, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    shlq $60, %rax
; CHECK-NEXT:    sarq $63, %rax
; CHECK-NEXT:    vpinsrd $3, %eax, %xmm0, %xmm0
; CHECK-NEXT:    vpmovsxdq %xmm0, %xmm1
; CHECK-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; CHECK-NEXT:    vpmovsxdq %xmm0, %xmm0
; CHECK-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; CHECK-NEXT:    retq
  %wide.load35 = load <4 x i1>, <4 x i1>* %in, align 1
  %sext = sext <4 x i1> %wide.load35 to <4 x i64>
  ret <4 x i64> %sext
}

define <16 x i4> @test4(<16 x i4>* %in) nounwind {
; CHECK-LABEL: test4:
; CHECK:       # BB#0:
; CHECK-NEXT:    movq (%rdi), %rax
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $4, %ecx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    movl %eax, %edx
; CHECK-NEXT:    andl $15, %edx
; CHECK-NEXT:    vmovd %edx, %xmm0
; CHECK-NEXT:    vpinsrb $1, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $8, %ecx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $2, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $12, %ecx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $3, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $16, %ecx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $4, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $20, %ecx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $5, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $24, %ecx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $6, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    shrl $28, %ecx
; CHECK-NEXT:    vpinsrb $7, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movq %rax, %rcx
; CHECK-NEXT:    shrq $32, %rcx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $8, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movq %rax, %rcx
; CHECK-NEXT:    shrq $36, %rcx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $9, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movq %rax, %rcx
; CHECK-NEXT:    shrq $40, %rcx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $10, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movq %rax, %rcx
; CHECK-NEXT:    shrq $44, %rcx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $11, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movq %rax, %rcx
; CHECK-NEXT:    shrq $48, %rcx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $12, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movq %rax, %rcx
; CHECK-NEXT:    shrq $52, %rcx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $13, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    movq %rax, %rcx
; CHECK-NEXT:    shrq $56, %rcx
; CHECK-NEXT:    andl $15, %ecx
; CHECK-NEXT:    vpinsrb $14, %ecx, %xmm0, %xmm0
; CHECK-NEXT:    shrq $60, %rax
; CHECK-NEXT:    vpinsrb $15, %eax, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %ret = load <16 x i4>, <16 x i4>* %in, align 1
  ret <16 x i4> %ret
}
