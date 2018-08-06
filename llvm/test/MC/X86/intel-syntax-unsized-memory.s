// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s | FileCheck %s

// Check that we deduce unsized memory operands in the general, unambiguous, case.
// We can't deduce xword memory operands, because there is no instruction
// unambiguously accessing 80-bit memory.

// CHECK: movb %al, (%rax)
mov [rax], al

// CHECK: movw %ax, (%rax)
mov [rax], ax

// CHECK: movl %eax, (%rax)
mov [rax], eax

// CHECK: movq %rax, (%rax)
mov [rax], rax

// CHECK: movdqa %xmm0, (%rax)
movdqa [rax], xmm0

// CHECK: vmovdqa %ymm0, (%rax)
vmovdqa [rax], ymm0

// CHECK: vaddps (%rax), %zmm1, %zmm1
vaddps zmm1, zmm1, [rax]

// CHECK: leal 1(%r15d), %r9d
lea r9d, [r15d+1]
