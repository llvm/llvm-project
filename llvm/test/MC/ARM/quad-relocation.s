@ RUN: not llvm-mc -triple arm-arm-none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

	.align 3
symbol:
  .quad(symbol)

@ CHECK: error: unsupported relocation on symbol
@ CHECK-NEXT:   .quad(symbol)
@ CHECK-NEXT:        ^
