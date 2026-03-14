@ RUN: llvm-mc -n -triple thumbv7-apple-darwin10 %s -filetype=obj -o %t.obj
@ RUN: llvm-readobj -S --sd - < %t.obj > %t.dump
@ RUN: FileCheck < %t.dump %s

	.syntax unified
        .text
	.thumb
	.thumb_func _foo
_foo:
        ldr r2, (_foo - 4)

@ CHECK:  RelocationCount: 0
@ CHECK:  Type: Regular (0x0)
@ CHECK:  Attributes [ (0x800004)
@ CHECK:    PureInstructions (0x800000)
@ CHECK:    SomeInstructions (0x4)
@ CHECK:  ]
@ CHECK:  Reserved1: 0x0
@ CHECK:  Reserved2: 0x0
@ CHECK:  SectionData (
@ CHECK:    0000: 5FF80820                             |_.. |
@ CHECK:  )
