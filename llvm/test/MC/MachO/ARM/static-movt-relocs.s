@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumbv7-apple-darwin10 -filetype=obj -o - < %s | llvm-readobj -r --expand-relocs - | FileCheck %s
        .thumb
        .thumb_func foo
foo:
        movw r0, :lower16:(bar + 16)
        movt r0, :upper16:(bar + 16)
        bx r0

@ CHECK:      Relocations [
@ CHECK-NEXT:   Section __text {
@ CHECK-NEXT:     Relocation {
@ CHECK-NEXT:       Offset: 0x4
@ CHECK-NEXT:       PCRel: 0
@ CHECK-NEXT:       Length: 3
@ CHECK-NEXT:       Type: ARM_RELOC_HALF (8)
@ CHECK-NEXT:       Symbol: bar
@ CHECK-NEXT:     }
@ CHECK-NEXT:     Relocation {
@ CHECK-NEXT:       Offset: 0x10
@ CHECK-NEXT:       PCRel: 0
@ CHECK-NEXT:       Length: 3
@ CHECK-NEXT:       Type: ARM_RELOC_PAIR (1)
@ CHECK-NEXT:       Section: -
@ CHECK-NEXT:     }
@ CHECK-NEXT:     Relocation {
@ CHECK-NEXT:       Offset: 0x0
@ CHECK-NEXT:       PCRel: 0
@ CHECK-NEXT:       Length: 2
@ CHECK-NEXT:       Type: ARM_RELOC_HALF (8)
@ CHECK-NEXT:       Symbol: bar
@ CHECK-NEXT:     }
@ CHECK-NEXT:     Relocation {
@ CHECK-NEXT:       Offset: 0x0
@ CHECK-NEXT:       PCRel: 0
@ CHECK-NEXT:       Length: 2
@ CHECK-NEXT:       Type: ARM_RELOC_PAIR (1)
@ CHECK-NEXT:       Section: -
@ CHECK-NEXT:     }
@ CHECK-NEXT:   }
@ CHECK-NEXT: ]
