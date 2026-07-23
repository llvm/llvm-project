// RUN: llvm-mc -triple x86_64_lfi %s | FileCheck %s

jmpq *%rax
// CHECK:      andl $-32, %eax
// CHECK-NEXT: addq %r14, %rax
// CHECK-NEXT: jmpq *%rax

// The scratch register may be used as a branch target.
jmpq *%r11
// CHECK:      andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11

jmpq *(%rdi)
// CHECK:      movq (%rdi), %r11
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11

jmpq *8(%rdi,%rsi,4)
// CHECK:      movq 8(%rdi,%rsi,4), %r11
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11

jmpq *foo(%rip)
// CHECK:      movq foo(%rip), %r11
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11

// The target load is itself rewritten, so an %fs-relative branch target is
// resolved against the virtual thread pointer.
jmpq *%fs:(%rdi)
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: movq (%r11,%rdi), %r11
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11

notrack jmpq *%rax
// CHECK:      andl $-32, %eax
// CHECK-NEXT: addq %r14, %rax
// CHECK-NEXT: notrack jmpq *%rax

notrack callq *(%rdx)
// CHECK:      movq (%rdx), %r11
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: notrack callq *%r11

callq *%rcx
// CHECK:      andl $-32, %ecx
// CHECK-NEXT: addq %r14, %rcx
// CHECK-NEXT: callq *%rcx

callq *(%rdx)
// CHECK:      movq (%rdx), %r11
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: callq *%r11

ret
// CHECK:      popq %r11
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11

rep ret
// CHECK:      popq %r11
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11

retq $16
// CHECK:      popq %r11
// CHECK-NEXT: addq $16, %rsp
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11

callq foo
// CHECK: callq foo

jmp foo
// CHECK: jmp foo

je foo
// CHECK: je foo
