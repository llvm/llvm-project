; RUN: llvm-mc -triple armv7-windows-msvc -filetype obj -o - %s

  .syntax unified
  .thumb

  .text

  .global function
  .thumb_func
function:
  ; this is a comment
  mov r0, #42 ; this # was not
  bx lr
