@ RUN: llvm-mc -triple armv7a-unknown-linux-gnueabihf -filetype=obj -o - %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=EXPLICIT-SOFT

@ EXPLICIT-SOFT:        Tag: 28
@ EXPLICIT-SOFT-NEXT:   Value: 0
@ EXPLICIT-SOFT-NEXT:   TagName: ABI_VFP_args
@ EXPLICIT-SOFT-NEXT:   Description: AAPCS

.syntax unified
.eabi_attribute 28, 0 @ Explicitly declare Base AAPCS (soft-float/softfp)
.text
.global _start
_start:
    bx lr
