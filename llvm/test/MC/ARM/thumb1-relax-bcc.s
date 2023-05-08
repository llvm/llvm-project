@ RUN: not llvm-mc -triple thumbv6m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
@ RUN: not llvm-mc -triple thumbv7m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
@ RUN: llvm-mc -triple thumbv7m-none-eabi -filetype=obj -o %t %s
@ RUN:    llvm-objdump --no-print-imm-hex -d -r --triple=thumbv7m-none-eabi %t | FileCheck --check-prefix=CHECK-ELF %s

        .global func1
_func1:
@ CHECK-ERROR: :[[#@LINE+1]]:9: error: unsupported relocation type
        bne _func2

@ CHECK-ELF: f47f affe          bne.w {{.+}} @ imm = #-4
@ CHECK-ELF-NEXT: R_ARM_THM_JUMP19 _func2
