// RUN: llvm-mc -triple x86_64_lfi %s | FileCheck %s

movq %fs:0, %rax
// CHECK: movq 16(%r15), %rax

movq %fs:0, %rdi
// CHECK: movq 16(%r15), %rdi

movq %fs:0, %rcx
// CHECK: movq 16(%r15), %rcx

addq %fs:0, %rax
// CHECK: addq 16(%r15), %rax

movq %fs:(%rdi), %rax
// CHECK:      movq 16(%r15), %rax
// CHECK-NEXT: movq %gs:(%eax,%edi), %rax

movq %fs:(%rcx), %rdx
// CHECK:      movq 16(%r15), %rdx
// CHECK-NEXT: movq %gs:(%edx,%ecx), %rdx

// base == dest, falls back to %r11
movq %fs:(%rax), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: movq %gs:(%r11d,%eax), %rax

movq %rax, %fs:(%rdi)
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: movq %rax, %gs:(%r11d,%edi)

movq %fs:8(%rdi,%rsi,2), %rax
// CHECK:      movq 16(%r15), %rax
// CHECK-NEXT: leaq (%rax,%rdi), %rax
// CHECK-NEXT: movq %gs:8(%eax,%esi,2), %rax

movq %fs:(%rax,%rbx,4), %rcx
// CHECK:      movq 16(%r15), %rcx
// CHECK-NEXT: leaq (%rcx,%rax), %rcx
// CHECK-NEXT: movq %gs:(%ecx,%ebx,4), %rcx

movq %rax, %fs:8(%rdi,%rsi,2)
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: leaq (%r11,%rdi), %r11
// CHECK-NEXT: movq %rax, %gs:8(%r11d,%esi,2)

movq %fs:foo, %rax
// CHECK:      movq 16(%r15), %rax
// CHECK-NEXT: movq %gs:foo(%eax), %rax

movq %fs:foo@TPOFF, %rax
// CHECK:      movq 16(%r15), %rax
// CHECK-NEXT: movq %gs:foo@TPOFF(%eax), %rax

movq %fs:foo(%rdi), %rax
// CHECK:      movq 16(%r15), %rax
// CHECK-NEXT: movq %gs:foo(%eax,%edi), %rax

movq %fs:foo@TPOFF(%rdi,%rsi,2), %rax
// CHECK:      movq 16(%r15), %rax
// CHECK-NEXT: leaq (%rax,%rdi), %rax
// CHECK-NEXT: movq %gs:foo@TPOFF(%eax,%esi,2), %rax

movq %rcx, %fs:foo
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: movq %rcx, %gs:foo(%r11d)

cmpq %fs:(%rdi), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: cmpq %gs:(%r11d,%edi), %rax

cmpq %fs:(%rbx,%rcx), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: leaq (%r11,%rbx), %r11
// CHECK-NEXT: cmpq %gs:(%r11d,%ecx), %rax

cmpq %fs:8(%rdi,%rsi,2), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: leaq (%r11,%rdi), %r11
// CHECK-NEXT: cmpq %gs:8(%r11d,%esi,2), %rax

subq %fs:(%rdi), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: subq %gs:(%r11d,%edi), %rax

cmoveq %fs:(%rdi), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: cmoveq %gs:(%r11d,%edi), %rax

andnq %fs:(%rdi), %rax, %rbx
// CHECK:      movq 16(%r15), %rbx
// CHECK-NEXT: andnq %gs:(%ebx,%edi), %rax, %rbx

andnq %fs:(%rdi), %rax, %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: andnq %gs:(%r11d,%edi), %rax, %rax
