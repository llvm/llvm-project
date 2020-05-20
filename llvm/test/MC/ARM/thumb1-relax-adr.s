@ RUN: not llvm-mc -triple thumbv6m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
@ RUN: not llvm-mc -triple thumbv7m-none-macho -filetype=obj -o /dev/null %s 2>&1  | FileCheck --check-prefix=CHECK-ERROR %s
@ RUN: not llvm-mc -triple thumbv7m-none-eabi -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

        .global func1
_func1:
        adr r0, _func2
@ CHECK-ERROR: unsupported relocation on symbol
