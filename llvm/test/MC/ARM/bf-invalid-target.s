@ RUN: not llvm-mc -triple arm-none-eabi -mcpu=cortex-m55 -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

.text
bf label, __myfunc
label:
  b __myfunc

.section .text.foo, "ax", %progbits
.type __myfunc, %function
.global __myfunc
__myfunc:
  nop

@ CHECK: error: out of range pc-relative fixup value
