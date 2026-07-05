@ RUN: llvm-mc -filetype=obj -o %t %s -triple=armv7
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=armv7 %t | FileCheck %s --check-prefix=ARM_OFFSET

@ RUN: llvm-mc -filetype=obj -o %t %s -triple=armebv7
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=armebv7  %t | FileCheck %s --check-prefix=ARM_OFFSET

@ RUN: llvm-mc -filetype=obj -o %t %s -triple=thumbv7
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=thumbv7  %t | FileCheck %s --check-prefix=THUMB_OFFSET

@ RUN: llvm-mc -filetype=obj -o %t %s -triple=thumbebv7
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=thumbebv7 %t | FileCheck %s --check-prefix=THUMB_OFFSET

baz:
    .word 0x11223344, 0x55667788
label:

    ldrd r0, r1, foo      @ arm_pcrel_10_unscaled / t2_pcrel_10
    ldrd r0, r1, bar-8    @ arm_pcrel_10_unscaled / t2_pcrel_10

    ldrd r0, r1, baz      @ arm_pcrel_10_unscaled / t2_pcrel_10
    ldrd r0, r1, label-8  @ arm_pcrel_10_unscaled / t2_pcrel_10
foo:
    .word 0x11223344, 0x55667788
bar:

@ RELOC: There are no relocations in this file.

@ ARM_OFFSET:   ldrd	r0, r1, [pc, #8]        @ 0x18 <foo>
@ ARM_OFFSET:   ldrd	r0, r1, [pc, #4]        @ 0x18 <foo>
@ ARM_OFFSET:   ldrd	r0, r1, [pc, #-24]      @ 0x0 <baz>
@ ARM_OFFSET:   ldrd	r0, r1, [pc, #-28]      @ 0x0 <baz>
@ THUMB_OFFSET: ldrd	r0, r1, [pc, #12]       @ 0x18 <foo>
@ THUMB_OFFSET: ldrd	r0, r1, [pc, #8]        @ 0x18 <foo>
@ THUMB_OFFSET: ldrd	r0, r1, [pc, #-20]      @ 0x0 <baz>
@ THUMB_OFFSET: ldrd	r0, r1, [pc, #-24]      @ 0x0 <baz>
