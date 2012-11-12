@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumbv7-apple-darwin10 -filetype=obj -o - < %s | macho-dump | FileCheck %s
        .thumb
        .thumb_func foo
foo:
        movw r0, :lower16:(bar + 16)
        movt r0, :upper16:(bar + 16)
        bx r0


@ CHECK:  ('_relocations', [
@ CHECK:    # Relocation 0
@ CHECK:    (('word-0', 0x4),
@ CHECK:     ('word-1', 0x8e000001)),
@ CHECK:    # Relocation 1
@ CHECK:    (('word-0', 0x10),
@ CHECK:     ('word-1', 0x16ffffff)),
@ CHECK:    # Relocation 2
@ CHECK:    (('word-0', 0x0),
@ CHECK:     ('word-1', 0x8c000001)),
@ CHECK:    # Relocation 3
@ CHECK:    (('word-0', 0x0),
@ CHECK:     ('word-1', 0x14ffffff)),
@ CHECK:  ])
