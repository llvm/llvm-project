// RUN: llvm-mc -triple x86_64_lfi %s | FileCheck %s

movq %rdi, %rsp
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %edi, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

addq $8, %rsp
// CHECK:      .bundle_lock
// CHECK-NEXT: addl $8, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

subq $16, %rsp
// CHECK:      .bundle_lock
// CHECK-NEXT: subl $16, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

addq %rax, %rsp
// CHECK:      .bundle_lock
// CHECK-NEXT: addl %eax, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

andq $-16, %rsp
// CHECK:      .bundle_lock
// CHECK-NEXT: andl $-16, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

orq $8, %rsp
// CHECK:      .bundle_lock
// CHECK-NEXT: orl $8, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

leaq 8(%rax), %rsp
// CHECK:      .bundle_lock
// CHECK-NEXT: leal 8(%rax), %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

movq (%rax), %rsp
// CHECK:      movq %gs:(%eax), %r11
// CHECK-NEXT: .bundle_lock
// CHECK-NEXT: movl %r11d, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

addq 8(%rdi), %rsp
// CHECK:      movq %gs:8(%edi), %r11
// CHECK-NEXT: .bundle_lock
// CHECK-NEXT: addl %r11d, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

andq (%rsp), %rsp
// CHECK:      movq (%rsp), %r11
// CHECK-NEXT: .bundle_lock
// CHECK-NEXT: andl %r11d, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

movl %eax, %esp
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %eax, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

popq %rsp
// CHECK:      popq %r11
// CHECK-NEXT: .bundle_lock
// CHECK-NEXT: movl %r11d, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock

leave
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %ebp, %esp
// CHECK-NEXT: leaq (%rsp,%r14), %rsp
// CHECK-NEXT: .bundle_unlock
// CHECK-NEXT: popq %rbp

pushq %rax
// CHECK: pushq %rax

popq %rax
// CHECK: popq %rax

pushq %rsp
// CHECK: pushq %rsp
