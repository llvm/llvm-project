@ RUN: llvm-mc -triple=armv7-linux-gnueabi -filetype=obj < %s | llvm-objdump -t - | FileCheck %s --match-full-lines

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

@ Similarly no $t if we change back .starts_thumb using .pushsection
        .pushsection .starts_thumb
        .thumb
        adds r0, r0, r0

@ When we change back to .text using .popsection .thumb is still active, so we
@ should emit a $t
        .popsection
        add r0, r0, r0

@ .ident does a push then pop of the .comment section, so the .word should
@ cause $d to appear in the .text section
        .ident "ident"
        .word 0

@ With all those constraints, we want:
@   + .text to have $a at 0, $t at 8, $d at 12
@   + .wibble to have $a at 0
@   + .starts_thumb to have $t at 0
@   + .starts_data to have $d at 0

@ CHECK:      00000000 l .text 00000000 $a
@ CHECK-NEXT: 00000000 l .wibble 00000000 $a
@ CHECK-NEXT: 00000000 l .starts_thumb 00000000 $t
@ CHECK-NEXT: 00000008 l .text 00000000 $t
@ CHECK-NEXT: 0000000a l .text 00000000 $d
@ CHECK-NOT: ${{[adt]}}

