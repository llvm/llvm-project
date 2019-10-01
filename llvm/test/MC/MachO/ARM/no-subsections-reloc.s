@ RUN: llvm-mc -n -triple thumbv7-apple-darwin10 %s -filetype=obj -o %t.obj
@ RUN: llvm-readobj -S --sd < %t.obj > %t.dump
@ RUN: FileCheck < %t.dump %s

@ When not using subsections-via-symbols, references to non-local symbols
@ in the same section can be resolved at assembly time w/o relocations.

 .syntax unified
 .text
 .thumb
 .thumb_func _foo
_foo:
    ldr r3, bar
bar:
    .long 0

@ CHECK: RelocationCount: 0
@ CHECK: SectionData (
@ CHECK:   0000: DFF80030 00000000                    |...0....|
@ CHECK: )
