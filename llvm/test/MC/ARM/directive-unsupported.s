@ RUN: not llvm-mc -triple thumbv7-windows -filetype asm -o /dev/null %s 2>&1 \
@ RUN:     | FileCheck %s

@ RUN: not llvm-mc -triple armv7-darwin -filetype asm -o /dev/null %s 2>&1 \
@ RUN:    | FileCheck %s

	.syntax unified

	.arch armv7

// CHECK: error: unknown directive
// CHECK: .arch armv7
// CHECK: ^

	.cpu cortex-a7

// CHECK: error: unknown directive
// CHECK: .cpu cortex-a7
// CHECK: ^

	.fpu neon

// CHECK: error: unknown directive
// CHECK: .fpu neon
// CHECK: ^

	.eabi_attribute 0, 0

// CHECK: error: unknown directive
// CHECK: .eabi_attribute 0, 0
// CHECK: ^

	.object_arch armv7

// CHECK: error: unknown directive
// CHECK: .object_arch armv7
// CHECK: ^

	.tlsdescseq undefined

// CHECK: error: unknown directive
// CHECK: .tlsdescseq undefined
// CHECK: ^

	.fnstart

// CHECK: error: unknown directive
// CHECK: .fnstart
// CHECK: ^

