@ RUN: llvm-mc -triple armv7a-unknown-linux-gnueabihf -filetype=obj -o - %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=HARD

@ RUN: llvm-mc -triple armv7a-unknown-linux-gnueabi -filetype=obj -o - %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=SOFT

@ RUN: llvm-mc -triple thumbv7-pc-windows-elf -filetype=obj -o - %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=HARD

@ HARD:        Tag: 28
@ HARD-NEXT:   Value: 1
@ HARD-NEXT:   TagName: ABI_VFP_args
@ HARD-NEXT:   Description: AAPCS VFP

@ SOFT-NOT: TagName: ABI_VFP_args

.syntax unified
.text
.global _start
_start:
    bx lr
