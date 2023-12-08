// RUN: llvm-mc -masm-integers -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s | FileCheck %s
// rdar://12470373

// Checks to make sure we parse the hexadecimal suffix properly.
// CHECK: movl $10, %eax
  mov eax, 10
// CHECK: movl $16, %eax
  mov eax, 10h
// CHECK: movl $16, %eax
  mov eax, 10H
// CHECK: movl $4294967295, %eax
  mov eax, 0ffffffffh
// CHECK: movl $4294967295, %eax
  mov eax, 0xffffffff
// CHECK: movl $4294967295, %eax
  mov eax, 0xffffffffh
// CHECK: movl $15, %eax
  mov eax, 0fh
// CHECK: movl $162, %eax
  mov eax, 0a2h
// CHECK: movl $162, %eax
  mov eax, 0xa2
// CHECK: movl $162, %eax
  mov eax, 0xa2h
// CHECK: movl $674, %eax
  mov eax, 2a2h
