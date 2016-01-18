@ RUN: not llvm-mc -triple thumbv6m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

        .global func1
_func1:
        ldr r0, L_far
        .space 1024

        .p2align 2
L_far:
        .word 42

@ CHECK-ERROR: out of range pc-relative fixup value

