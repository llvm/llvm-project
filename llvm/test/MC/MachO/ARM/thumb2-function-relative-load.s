@ RUN: llvm-mc -n -triple thumbv7-apple-darwin10 %s -filetype=obj -o %t.obj
@ RUN: llvm-readobj -S --sd < %t.obj > %t.dump
@ RUN: FileCheck < %t.dump %s
        .syntax unified
        .text
	.thumb
        .thumb_func _foo
_foo:
	ldr lr, (_foo - 4)

        .subsections_via_symbols

@ CHECK:  SectionData (
@ CHECK:    0000: 5FF808E0                             |_...|
@ CHECK:  )
