@ RUN: not llvm-mc -triple thumbv6m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

        .global func1
        adr r0, Lmisaligned
Lmisaligned:
        .word 42

@ CHECK-ERROR: misaligned pc-relative fixup value

