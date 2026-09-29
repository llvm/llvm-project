@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null %s 2>&1 \
@ RUN:   | FileCheck %s

	.syntax unified

	.object_arch i686

@ CHECK: error: unknown architecture 'i686'
@ CHECK: 	.object_arch i686
@ CHECK:                     ^

	.object_arch armv4!

@ CHECK: error: unexpected token
@ CHECK: 	.object_arch armv4!
@ CHECK:                          ^

	.object_arch, invalid

@ CHECK: error: unexpected token
@ CHECK: 	.object_arch, invalid
@ CHECK:                    ^

