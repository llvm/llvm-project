@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null %s 2>&1 \
@ RUN:    | FileCheck %s -check-prefix CHECK-EABI

@ NOTE: this test ensures that both forms are accepted for MachO
@ RUN: llvm-mc -triple armv7-darwin -filetype asm -o /dev/null %s

	.syntax unified

	.thumb_func
no_suffix:
	bx lr

	.thumb_func suffix
suffix:
	bx lr

// CHECK-EABI: error: unexpected token in '.thumb_func' directive
// CHECK-EABI: 	.thumb_func suffix
// CHECK-EABI:              ^

// CHECK-EABI-NOT: error: invalid instruction

