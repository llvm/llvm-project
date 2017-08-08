@ RUN: not llvm-mc -triple thumbv6m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
@ RUN: not llvm-mc -triple thumbv7m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
@ RUN: llvm-mc -triple thumbv7m-none-eabi -filetype=obj -o %t %s
@ RUN:    llvm-objdump -d -r -triple thumbv7m-none-eabi %t | FileCheck --check-prefix=CHECK-ELF %s

        .global func1
_func1:
        bne _func2
@ CHECK-ERROR: unsupported relocation on symbol

@ CHECK-ELF: 7f f4 fe af        bne.w #-4
@ CHECK-ELF-NEXT: R_ARM_THM_JUMP19 _func2
