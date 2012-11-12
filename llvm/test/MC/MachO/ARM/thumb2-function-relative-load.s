@ RUN: llvm-mc -n -triple thumbv7-apple-darwin10 %s -filetype=obj -o %t.obj
@ RUN: macho-dump --dump-section-data < %t.obj > %t.dump
@ RUN: FileCheck < %t.dump %s
        .syntax unified
        .text
	.thumb
        .thumb_func _foo
_foo:
	ldr lr, (_foo - 4)

        .subsections_via_symbols

@ CHECK: ('_section_data', '5ff808e0')
