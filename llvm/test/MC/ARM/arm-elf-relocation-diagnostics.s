@ RUN: not llvm-mc -triple armv7-eabi -filetype obj -o - %s 2>&1 \
@ RUN:     | FileCheck %s
@ RUN: not llvm-mc -triple thumbv7-eabi -filetype obj -o - %s 2>&1 \
@ RUN:     | FileCheck %s

	.byte target(sbrel)
@ CHECK: error: relocated expression must be 32-bit
@ CHECK: .byte target(sbrel)
@ CHECK:       ^

	.hword target(sbrel)
@ CHECK: error: relocated expression must be 32-bit
@ CHECK: .hword target(sbrel)
@ CHECK:        ^

	.short target(sbrel)
@ CHECK: error: relocated expression must be 32-bit
@ CHECK: .short target(sbrel)
@ CHECK:        ^

	.quad target(sbrel)
@ CHECK: error: relocated expression must be 32-bit
@ CHECK: .quad target(sbrel)
@ CHECK:        ^


