@ RUN: llvm-mc -triple=armv7-linux-gnueabi -filetype=obj < %s | llvm-objdump -t - | FileCheck %s

        .text
        add r0, r0, r0

@ .wibble should *not* inherit .text's mapping symbol. It's a completely different section.
        .section .wibble
        add r0, r0, r0

@ A section should be able to start with a $t
        .section .starts_thumb
        .thumb
        adds r0, r0, r0

@ A setion should be able to start with a $d
        .section .starts_data
        .word 42

@ Changing back to .text should not emit a redundant $a
        .text
        .arm
        add r0, r0, r0

@ With all those constraints, we want:
@   + .text to have $a at 0 and no others
@   + .wibble to have $a at 0
@   + .starts_thumb to have $t at 0
@   + .starts_data to have $d at 0

@ CHECK: 00000000 .text 00000000 $a
@ CHECK-NEXT: 00000000 .wibble 00000000 $a
@ CHECK-NEXT: 00000000 .starts_thumb 00000000 $t
@ CHECK-NOT: ${{[adt]}}

