// RUN: llvm-mc -triple x86_64_lfi %s | FileCheck %s

movq (%rax), %rdi
// CHECK: movq %gs:(%eax), %rdi

movq %rcx, (%rax)
// CHECK: movq %rcx, %gs:(%eax)

movq 8(%rax), %rdi
// CHECK: movq %gs:8(%eax), %rdi

movq -8(%rax), %rdi
// CHECK: movq %gs:-8(%eax), %rdi

movq (%rax,%rdi), %rcx
// CHECK: movq %gs:(%eax,%edi), %rcx

movq 16(%rax,%rdi,4), %rcx
// CHECK: movq %gs:16(%eax,%edi,4), %rcx

movq (,%rdi,8), %rcx
// CHECK: movq %gs:(,%edi,8), %rcx

movl (%rax), %edi
// CHECK: movl %gs:(%eax), %edi

movw (%rax), %di
// CHECK: movw %gs:(%eax), %di

movb (%rax), %dil
// CHECK: movb %gs:(%eax), %dil

movb (%rax), %ah
// CHECK: movb %gs:(%eax), %ah

// An address that is already 32-bit keeps its registers.

movl (%edi), %eax
// CHECK: movl %gs:(%edi), %eax

addq $1, (%rax)
// CHECK: addq $1, %gs:(%eax)

incq (%rax)
// CHECK: incq %gs:(%eax)

xchgq %rax, (%rcx)
// CHECK: xchgq %rax, %gs:(%ecx)

lock incq (%rax)
// CHECK: lock {{.*}}incq %gs:(%eax)

lock
addq $1, (%rax)
// CHECK:      lock
// CHECK-NEXT: addq $1, %gs:(%eax)

movq (%rsp), %rax
// CHECK: movq (%rsp), %rax

movq 8(%rsp), %rax
// CHECK: movq 8(%rsp), %rax

movq %rax, -8(%rsp)
// CHECK: movq %rax, -8(%rsp)

movq foo(%rip), %rax
// CHECK: movq foo(%rip), %rax

movq %rax, foo(%rip)
// CHECK: movq %rax, foo(%rip)

movq (%r14), %rax
// CHECK: movq (%r14), %rax

movq 8(%r14), %rax
// CHECK: movq 8(%r14), %rax

movq (%rsp,%rax), %rcx
// CHECK: movq %gs:(%esp,%eax), %rcx

movq (%r14,%rax), %rcx
// CHECK: movq %gs:(%r14d,%eax), %rcx

movq 4096, %rax
// CHECK: movq 4096(%r14), %rax

vextractf64x2 $1, %zmm0, (%rax)
// CHECK: vextractf64x2 $1, %zmm0, %gs:(%eax)

vextracti32x8 $1, %zmm0, 16(%rax,%rcx,4)
// CHECK: vextracti32x8 $1, %zmm0, %gs:16(%eax,%ecx,4)

tileloaddrs 291(%rbp,%rax,4), %tmm3
// CHECK: tileloaddrs %gs:291(%ebp,%eax,4), %tmm3

movq (%rax,%riz,1), %rbx
// CHECK: movq %gs:(%eax,%riz), %rbx

movq (,%riz,1), %rbx
// CHECK: movq %gs:(%r14d,%riz), %rbx

leaq (%rax), %rdi
// CHECK: leaq (%rax), %rdi

leaq 8(%rax,%rcx,4), %rdi
// CHECK: leaq 8(%rax,%rcx,4), %rdi

leal 4(%eax), %edx
// CHECK: leal 4(%eax), %edx

nopw (%rax)
// CHECK: nopw (%rax)

nopl (%rax,%rcx,4)
// CHECK: nopl (%rax,%rcx,4)

movaps (%rdi), %xmm0
// CHECK: movaps %gs:(%edi), %xmm0

movups %xmm1, 16(%rsi,%rdx,4)
// CHECK: movups %xmm1, %gs:16(%esi,%edx,4)

flds (%rax)
// CHECK: flds %gs:(%eax)

fstpl 8(%rsp)
// CHECK: fstpl 8(%rsp)

vgatherdpd %xmm3, (%rax,%xmm2,8), %xmm0
// CHECK: vgatherdpd %xmm3, %gs:(%eax,%xmm2,8), %xmm0

vgatherqps %xmm3, (%rdi,%xmm2,4), %xmm0
// CHECK: vgatherqps %xmm3, %gs:(%edi,%xmm2,4), %xmm0

vpgatherdd (%rax,%ymm2,4), %ymm0 {%k1}
// CHECK: vpgatherdd %gs:(%eax,%ymm2,4), %ymm0 {%k1}

vpscatterdd %ymm0, 16(%rax,%ymm2,4) {%k1}
// CHECK: vpscatterdd %ymm0, %gs:16(%eax,%ymm2,4) {%k1}

vgatherpf0dps (%rax,%zmm2,4) {%k1}
// CHECK: vgatherpf0dps %gs:(%eax,%zmm2,4) {%k1}

vgatherdpd %xmm3, (,%xmm2,8), %xmm0
// CHECK: vgatherdpd %xmm3, %gs:(%r14d,%xmm2,8), %xmm0

vgatherdpd %xmm3, (%r14,%xmm2,8), %xmm0
// CHECK: vgatherdpd %xmm3, %gs:(%r14d,%xmm2,8), %xmm0

pushq %rax
// CHECK: pushq %rax

pushq (%rax)
// CHECK: pushq %gs:(%eax)

popq (%rax)
// CHECK: popq %gs:(%eax)
