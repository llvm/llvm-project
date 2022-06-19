@ RUN: llvm-mc -triple armv7-windows-gnu -filetype obj -o %t.obj %s
@ RUN: llvm-objdump -d %t.obj | FileCheck %s
@ RUN: llvm-mc -triple armv7-windows-msvc -filetype obj -o %t.obj %s
@ RUN: llvm-objdump -d %t.obj | FileCheck %s

  .syntax unified
  .thumb

  .text

  .global function
  .thumb_func
function:
  @ this is a comment
  mov r0, #42 @ this # was not
  nop; nop @ This retains both instructions
  bx lr

@ CHECK:  0: 4f f0 2a 00   mov.w   r0, #42
@ CHECK:  4: 00 bf         nop
@ CHECK:  6: 00 bf         nop
@ CHECK:  8: 70 47         bx      lr
