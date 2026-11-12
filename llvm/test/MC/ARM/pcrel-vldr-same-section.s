@ RUN: llvm-mc -filetype=obj -o %t %s -triple=armv8.2a-eabi
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=armv8.2a-eabi      --mattr=+fullfp16 %t | FileCheck %s --check-prefix=ARM_OFFSET
@ RUN: llvm-mc -filetype=obj -o %t %s -triple=armebv8.2a-eabi
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=armebv8.2a-eabi    --mattr=+fullfp16 %t | FileCheck %s --check-prefix=ARM_OFFSET
@ RUN: llvm-mc -filetype=obj -o %t %s -triple=thumbv8.2a-eabi
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=thumbv8.2a-eabi    --mattr=+fullfp16 %t | FileCheck %s --check-prefix=THUMB_OFFSET
@ RUN: llvm-mc -filetype=obj -o %t %s -triple=thumbebv8.2a-eabi
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=thumbebv8.2a-eabi  --mattr=+fullfp16 %t | FileCheck %s --check-prefix=THUMB_OFFSET

         .arch_extension fp16
baz:
    .word 0x11223344, 0x55667788
label:

    vldr    s0, foo     @ arm_pcrel_10 / t2_pcrel_10
    vldr    d0, foo     @ arm_pcrel_10 / t2_pcrel_10
    vldr.16 s0, foo     @ arm_pcrel_9  / t2_pcrel_9
    vldr    s0, bar-8
    vldr    d0, bar-8
    vldr.16 s0, bar-8
    vldr    s0, baz
    vldr    d0, baz
    vldr.16 s0, baz
    vldr    s0, label-8
    vldr    d0, label-8
    vldr.16 s0, label-8

foo:
    .word 0x11223344, 0x55667788
bar:

@ RELOC: There are no relocations in this file.

@ ARM_OFFSET:   vldr    s0, [pc, #40]           @ 0x38 <foo>
@ ARM_OFFSET:   vldr    d0, [pc, #36]           @ 0x38 <foo>
@ ARM_OFFSET:   vldr.16 s0, [pc, #32]           @ 0x38 <foo>
@ ARM_OFFSET:   vldr    s0, [pc, #28]           @ 0x38 <foo>
@ ARM_OFFSET:   vldr    d0, [pc, #24]           @ 0x38 <foo>
@ ARM_OFFSET:   vldr.16 s0, [pc, #20]           @ 0x38 <foo>
@ ARM_OFFSET:   vldr    s0, [pc, #-40]          @ 0x0 <baz>
@ ARM_OFFSET:   vldr    d0, [pc, #-44]          @ 0x0 <baz>
@ ARM_OFFSET:   vldr.16 s0, [pc, #-48]          @ 0x0 <baz>
@ ARM_OFFSET:   vldr    s0, [pc, #-52]          @ 0x0 <baz>
@ ARM_OFFSET:   vldr    d0, [pc, #-56]          @ 0x0 <baz>
@ ARM_OFFSET:   vldr.16 s0, [pc, #-60]          @ 0x0 <baz>
@ THUMB_OFFSET: vldr    s0, [pc, #44]           @ 0x38 <foo>
@ THUMB_OFFSET: vldr    d0, [pc, #40]           @ 0x38 <foo>
@ THUMB_OFFSET: vldr.16 s0, [pc, #36]           @ 0x38 <foo>
@ THUMB_OFFSET: vldr    s0, [pc, #32]           @ 0x38 <foo>
@ THUMB_OFFSET: vldr    d0, [pc, #28]           @ 0x38 <foo>
@ THUMB_OFFSET: vldr.16 s0, [pc, #24]           @ 0x38 <foo>
@ THUMB_OFFSET: vldr    s0, [pc, #-36]          @ 0x0 <baz>
@ THUMB_OFFSET: vldr    d0, [pc, #-40]          @ 0x0 <baz>
@ THUMB_OFFSET: vldr.16 s0, [pc, #-44]          @ 0x0 <baz>
@ THUMB_OFFSET: vldr    s0, [pc, #-48]          @ 0x0 <baz>
@ THUMB_OFFSET: vldr    d0, [pc, #-52]          @ 0x0 <baz>
@ THUMB_OFFSET: vldr.16 s0, [pc, #-56]          @ 0x0 <baz>
