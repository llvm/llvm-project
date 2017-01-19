@ RUN: not llvm-mc -triple arm-arm-none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

	.align 3
symbol:
  .quad(symbol)

@ CHECK: error: bad relocation fixup type
@ CHECK-NEXT:   .quad(symbol)
@ CHECK-NEXT:        ^
