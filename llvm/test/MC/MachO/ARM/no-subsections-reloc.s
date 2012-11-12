@ RUN: llvm-mc -n -triple thumbv7-apple-darwin10 %s -filetype=obj -o %t.obj
@ RUN: macho-dump --dump-section-data < %t.obj > %t.dump
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

@ CHECK: 'num_reloc', 0
@ CHECK: '_section_data', 'dff80030 00000000'
