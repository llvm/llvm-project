// RUN: llvm-mc -triple x86_64_lfi %s | FileCheck %s

jmpq *%rax
// CHECK:      .bundle_lock
// CHECK-NEXT: andl $-32, %eax
// CHECK-NEXT: addq %r14, %rax
// CHECK-NEXT: jmpq *%rax
// CHECK-NEXT: .bundle_unlock

jmpq *(%rdi)
// CHECK:      movq (%rdi), %r11
// CHECK-NEXT: .bundle_lock
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11
// CHECK-NEXT: .bundle_unlock

jmpq *8(%rdi,%rsi,4)
// CHECK:      movq 8(%rdi,%rsi,4), %r11
// CHECK-NEXT: .bundle_lock
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11
// CHECK-NEXT: .bundle_unlock

jmpq *foo(%rip)
// CHECK:      movq foo(%rip), %r11
// CHECK-NEXT: .bundle_lock
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11
// CHECK-NEXT: .bundle_unlock

// The target load is itself rewritten, so an %fs-relative branch target is
// resolved against the virtual thread pointer.
jmpq *%fs:(%rdi)
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: movq (%r11,%rdi), %r11
// CHECK-NEXT: .bundle_lock
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11
// CHECK-NEXT: .bundle_unlock

notrack jmpq *%rax
// CHECK:      .bundle_lock
// CHECK-NEXT: andl $-32, %eax
// CHECK-NEXT: addq %r14, %rax
// CHECK-NEXT: notrack jmpq *%rax
// CHECK-NEXT: .bundle_unlock

notrack callq *(%rdx)
// CHECK:      movq (%rdx), %r11
// CHECK-NEXT: .bundle_lock align_to_end
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: notrack callq *%r11
// CHECK-NEXT: .bundle_unlock

callq *%rcx
// CHECK:      .bundle_lock align_to_end
// CHECK-NEXT: andl $-32, %ecx
// CHECK-NEXT: addq %r14, %rcx
// CHECK-NEXT: callq *%rcx
// CHECK-NEXT: .bundle_unlock

callq *(%rdx)
// CHECK:      movq (%rdx), %r11
// CHECK-NEXT: .bundle_lock align_to_end
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: callq *%r11
// CHECK-NEXT: .bundle_unlock

ret
// CHECK:      popq %r11
// CHECK-NEXT: .bundle_lock
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11
// CHECK-NEXT: .bundle_unlock

// The rep prefix has no effect on ret and the return is fully replaced, so the
// prefix is dropped.
rep ret
// CHECK-NOT:  rep
// CHECK:      popq %r11
// CHECK-NEXT: .bundle_lock
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11
// CHECK-NEXT: .bundle_unlock

retq $16
// CHECK:      popq %r11
// CHECK-NEXT: addq $16, %rsp
// CHECK-NEXT: .bundle_lock
// CHECK-NEXT: andl $-32, %r11d
// CHECK-NEXT: addq %r14, %r11
// CHECK-NEXT: jmpq *%r11
// CHECK-NEXT: .bundle_unlock

callq foo
// CHECK:      .bundle_lock align_to_end
// CHECK-NEXT: callq foo
// CHECK-NEXT: .bundle_unlock

jmp foo
// CHECK: jmp foo

je foo
// CHECK: je foo
