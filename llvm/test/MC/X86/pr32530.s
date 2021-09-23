// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s | FileCheck %s

.text
// CHECK: movq    $msg, %rsi
// CHECK: movq    $msg+314159, %rax
// CHECK: movq    $msg-89793, msg-6535(%rax,%rbx,2)
  mov rsi, offset msg
  mov rax, offset "msg" + 314159
  mov qword ptr [rax + 2*rbx + offset msg - 6535], offset msg - 89793
.data
msg:
  .ascii "Hello, world!\n"

