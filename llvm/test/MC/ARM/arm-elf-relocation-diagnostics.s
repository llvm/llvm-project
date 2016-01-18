@ RUN: not llvm-mc -triple armv7-eabi -filetype obj -o - %s 2>&1 \
@ RUN:     | FileCheck %s
@ RUN: not llvm-mc -triple thumbv7-eabi -filetype obj -o - %s 2>&1 \
@ RUN:     | FileCheck %s

	.byte target(sbrel)
@ CHECK: error: relocated expression must be 32-bit
@ CHECK: .byte target(sbrel)
@ CHECK:       ^

@ TODO: enable these negative test cases
@ 	.hword target(sbrel)
@ @ CHECK-SBREL-HWORD: error: relocated expression must be 32-bit
@ @ CHECK-SBREL-HWORD: .hword target(sbrel)
@ @ CHECK-SBREL-HWORD:        ^
@
@ 	.short target(sbrel)
@ @ CHECK-SBREL-SHORT: error: relocated expression must be 32-bit
@ @ CHECK-SBREL-SHORT: .short target(sbrel)
@ @ CHECK-SBREL-SHORT:        ^
@
@ 	.quad target(sbrel)
@ @ CHECK-SBREL-SHORT: error: relocated expression must be 32-bit
@ @ CHECK-SBREL-SHORT: .quad target(sbrel)
@ @ CHECK-SBREL-SHORT:        ^


