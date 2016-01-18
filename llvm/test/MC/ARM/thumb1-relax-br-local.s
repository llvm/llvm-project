@ RUN: not llvm-mc -triple thumbv6m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

        b Lfar
        .space 2050

Lfar:

@ CHECK-ERROR: out of range pc-relative fixup value
