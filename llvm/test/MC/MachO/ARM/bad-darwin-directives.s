@ RUN: not llvm-mc -n -triple armv7-apple-darwin10 %s -filetype asm -o /dev/null 2>&1 \
@ RUN:  | FileCheck --check-prefix CHECK-ERROR %s

@ RUN: not llvm-mc -n -triple armv7-apple-darwin10 %s -filetype obj -o /dev/null 2>&1 \
@ RUN:  | FileCheck --check-prefix CHECK-ERROR %s

@ rdar://16335232

.eabi_attribute 8, 1
@ CHECK-ERROR: error: unknown directive

.cpu
@ CHECK-ERROR: error: unknown directive

.fpu neon
@ CHECK-ERROR: error: unknown directive

.arch armv7
@ CHECK-ERROR: error: unknown directive

.fnstart
@ CHECK-ERROR: error: unknown directive

.tlsdescseq
@ CHECK-ERROR: error: unknown directive

.object_arch armv7
@ CHECK-ERROR: error: unknown directive

