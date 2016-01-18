@ RUN: not llvm-mc -triple thumbv6m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

Lhere:
        ldr r0, Lhere

@ CHECK-ERROR: out of range pc-relative fixup value

