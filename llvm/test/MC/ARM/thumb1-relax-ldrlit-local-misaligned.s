@ RUN: not llvm-mc -triple thumbv6m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

        .global func1
_func1:
        ldr r0, L_misaligned
L_misaligned:
        .word 42

@ CHECK-ERROR: misaligned pc-relative fixup value

