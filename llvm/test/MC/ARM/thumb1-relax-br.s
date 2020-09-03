@ RUN: not llvm-mc -triple thumbv6m-none-macho -filetype=obj -o /dev/null %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
@ RUN: llvm-mc -triple thumbv7m-none-macho -filetype=obj -o %t %s
@ RUN:    llvm-objdump -d -r --triple=thumbv7m-none-macho %t | FileCheck --check-prefix=CHECK-MACHO %s
@ RUN: llvm-mc -triple thumbv7m-none-eabi -filetype=obj -o %t %s
@ RUN:    llvm-objdump -d -r --triple=thumbv7m-none-eabi %t | FileCheck --check-prefix=CHECK-ELF %s

        .global func1
_func1:
        @ There is no MachO relocation for Thumb1's unconditional branch, so
        @ this is unrepresentable. FIXME: I think ELF could represent this.
        b _func2

@ CHECK-ERROR: unsupported relocation on symbol

@ CHECK-MACHO: ff f7 fe bf        b.w #-4
@ CHECK-MACHO-NEXT: ARM_THUMB_RELOC_BR22

@ CHECK-ELF: ff f7 fe bf        b.w #-4
@ CHECK-ELF-NEXT: R_ARM_THM_JUMP24 _func2
