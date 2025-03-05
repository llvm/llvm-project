@ RUN: llvm-mc -triple thumbv7s-apple-ios9.0 %s -filetype obj -o %t.o
@ RUN: llvm-readobj -r %t.o | FileCheck %s

        .thumb
        movw r0, :lower16:_x
        movt r0, :upper16:_x

        movw r0, :lower16:_x+4
        movt r0, :upper16:_x+4

        movw r0, :lower16:_x+0x1000
        movt r0, :upper16:_x+0x1000

        .arm
        movw r0, :lower16:_x
        movt r0, :upper16:_x

        movw r0, :lower16:_x+4
        movt r0, :upper16:_x+4

        movw r0, :lower16:_x+0x1000
        movt r0, :upper16:_x+0x1000

@ Enter the bizarre world of MachO relocations. First, they're in reverse order
@ to the actual instructions

@ First column on the second line is the "other half" of the addend, its partner
@ being in the instruction itself.

@ Third column identifies ARM/Thumb & HI/LO.

@ CHECK: 0x2C 0 1 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x1000 0 1 0 ARM_RELOC_PAIR 0 -

@ CHECK: 0x28 0 0 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x0 0 0 0 ARM_RELOC_PAIR 0 -

@ CHECK: 0x24 0 1 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x4 0 1 0 ARM_RELOC_PAIR 0 -

@ CHECK: 0x20 0 0 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x0 0 0 0 ARM_RELOC_PAIR 0 -

@ CHECK: 0x1C 0 1 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x0 0 1 0 ARM_RELOC_PAIR 0 -

@ CHECK: 0x18 0 0 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x0 0 0 0 ARM_RELOC_PAIR 0 -

@ CHECK: 0x14 0 3 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x1000 0 3 0 ARM_RELOC_PAIR 0 -

@ CHECK: 0x10 0 2 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x0 0 2 0 ARM_RELOC_PAIR 0 -

@ CHECK: 0xC 0 3 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x4 0 3 0 ARM_RELOC_PAIR 0 -

@ CHECK: 0x8 0 2 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x0 0 2 0 ARM_RELOC_PAIR 0 -

@ CHECK: 0x4 0 3 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x0 0 3 0 ARM_RELOC_PAIR 0 -

@ CHECK: 0x0 0 2 1 ARM_RELOC_HALF 0 _x
@ CHECK: 0x0 0 2 0 ARM_RELOC_PAIR 0 -
