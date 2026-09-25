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
// CHECK-NEXT: movq (%rax,%rdi), %rax

movq %fs:(%rcx), %rdx
// CHECK:      movq 16(%r15), %rdx
// CHECK-NEXT: movq (%rdx,%rcx), %rdx

// base == dest, falls back to %r11
movq %fs:(%rax), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: movq (%r11,%rax), %rax

movq %rax, %fs:(%rdi)
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: movq %rax, (%r11,%rdi)

movq %fs:8(%rdi,%rsi,2), %rax
// CHECK:      movq 16(%r15), %rax
// CHECK-NEXT: leaq (%rax,%rdi), %rax
// CHECK-NEXT: movq 8(%rax,%rsi,2), %rax

movq %fs:(%rax,%rbx,4), %rcx
// CHECK:      movq 16(%r15), %rcx
// CHECK-NEXT: leaq (%rcx,%rax), %rcx
// CHECK-NEXT: movq (%rcx,%rbx,4), %rcx

movq %rax, %fs:8(%rdi,%rsi,2)
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: leaq (%r11,%rdi), %r11
// CHECK-NEXT: movq %rax, 8(%r11,%rsi,2)

movq %fs:foo, %rax
// CHECK:      movq 16(%r15), %rax
// CHECK-NEXT: movq foo(%rax), %rax

movq %fs:foo@TPOFF, %rax
// CHECK:      movq 16(%r15), %rax
// CHECK-NEXT: movq foo@TPOFF(%rax), %rax

movq %fs:foo(%rdi), %rax
// CHECK:      movq 16(%r15), %rax
// CHECK-NEXT: movq foo(%rax,%rdi), %rax

movq %fs:foo@TPOFF(%rdi,%rsi,2), %rax
// CHECK:      movq 16(%r15), %rax
// CHECK-NEXT: leaq (%rax,%rdi), %rax
// CHECK-NEXT: movq foo@TPOFF(%rax,%rsi,2), %rax

movq %rcx, %fs:foo
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: movq %rcx, foo(%r11)

cmpq %fs:(%rdi), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: cmpq (%r11,%rdi), %rax

cmpq %fs:(%rbx,%rcx), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: leaq (%r11,%rbx), %r11
// CHECK-NEXT: cmpq (%r11,%rcx), %rax

cmpq %fs:8(%rdi,%rsi,2), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: leaq (%r11,%rdi), %r11
// CHECK-NEXT: cmpq 8(%r11,%rsi,2), %rax

subq %fs:(%rdi), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: subq (%r11,%rdi), %rax

cmoveq %fs:(%rdi), %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: cmoveq (%r11,%rdi), %rax

andnq %fs:(%rdi), %rax, %rbx
// CHECK:      movq 16(%r15), %rbx
// CHECK-NEXT: andnq (%rbx,%rdi), %rax, %rbx

andnq %fs:(%rdi), %rax, %rax
// CHECK:      movq 16(%r15), %r11
// CHECK-NEXT: andnq (%r11,%rdi), %rax, %rax
