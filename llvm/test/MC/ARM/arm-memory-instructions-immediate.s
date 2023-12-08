// RUN: llvm-mc -triple=armv7 -filetype=obj %s | llvm-objdump --no-print-imm-hex --triple=armv7 -d - | FileCheck %s
// RUN: not llvm-mc -triple=armv7 -filetype=obj --defsym=ERR=1 < %s -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s

    .syntax unified
// Check that the assembler accepts the result of symbolic expressions as the
// immediate operand in load and stores.
0:
// CHECK-LABEL: foo
    .space 1024
1:
foo:
    ldr r0, [r1, #(1b - 0b)]
// CHECK-NEXT: ldr r0, [r1, #1024]
    ldr r0, [r1, #(0b - 1b)]
// CHECK-NEXT: ldr r0, [r1, #-1024]
    ldrb r0, [r1, #(1b-0b)]
// CHECK-NEXT: ldrb r0, [r1, #1024]
    str r0, [r1, #(1b-0b)]
// CHECK-NEXT: str r0, [r1, #1024]
    strb r0, [r1, #(1b-0b)]
// CHECK-NEXT: strb r0, [r1, #1024]
.ifdef ERR
    str r0, [r1, 1b]
// ERR:[[#@LINE-1]]:5: error: unsupported relocation type
.endif
