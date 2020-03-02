@ RUN: not llvm-mc -triple thumbv6m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
@ RUN: not llvm-mc -triple thumbv7m-none-macho -filetype=obj -o /dev/null %s 2>&1  | FileCheck --check-prefix=CHECK-ERROR %s
@ RUN: llvm-mc -triple thumbv6m-none-eabi -filetype=obj %s -o - | llvm-readobj --relocs | FileCheck --check-prefix=CHECK-ELF-T1 %s
@ RUN: llvm-mc -triple thumbv7m-none-eabi -filetype=obj %s -o - | llvm-readobj --relocs | FileCheck --check-prefix=CHECK-ELF-T2 %s

        .global func1
_func1:
        adr r0, _func2
@ CHECK-ERROR: unsupported relocation on symbol
@ CHECK-ELF-T1: 0x0 R_ARM_THM_PC8 _func2 0x0
@ CHECK-ELF-T2: 0x0 R_ARM_THM_ALU_PREL_11_0 _func2 0x0
