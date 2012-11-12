@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumbv7-apple-darwin10 -filetype=obj -o - < %s | macho-dump | FileCheck %s

@ rdar://10038370

	.syntax unified
  .text
	.align	2
	.code	16           
	.thumb_func	_foo
  movw	r2, :lower16:L1
	movt	r2, :upper16:L1
  movw	r12, :lower16:L2
	movt	r12, :upper16:L2
  .space 70000
  
  .data
L1: .long 0
L2: .long 0

@ CHECK:  ('_relocations', [
@ CHECK:    # Relocation 0
@ CHECK:    (('word-0', 0xc),
@ CHECK:     ('word-1', 0x86000002)),
@ CHECK:    # Relocation 1
@ CHECK:    (('word-0', 0x1184),
@ CHECK:     ('word-1', 0x16ffffff)),
@ CHECK:    # Relocation 2
@ CHECK:    (('word-0', 0x8),
@ CHECK:     ('word-1', 0x84000002)),
@ CHECK:    # Relocation 3
@ CHECK:    (('word-0', 0x1),
@ CHECK:     ('word-1', 0x14ffffff)),
@ CHECK:    # Relocation 4
@ CHECK:    (('word-0', 0x4),
@ CHECK:     ('word-1', 0x86000002)),
@ CHECK:    # Relocation 5
@ CHECK:    (('word-0', 0x1180),
@ CHECK:     ('word-1', 0x16ffffff)),
@ CHECK:    # Relocation 6
@ CHECK:    (('word-0', 0x0),
@ CHECK:     ('word-1', 0x84000002)),
@ CHECK:    # Relocation 7
@ CHECK:    (('word-0', 0x1),
@ CHECK:     ('word-1', 0x14ffffff)),
